import os
import sys
import time
import json
import socket
import logging
import struct
import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
from PIL import Image
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Dict
import tyro
import pickle
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.append(os.getcwd())
sys.path.append("/mnt/public/user/MFVLA")

# Import necessary utils matching deploy_policy.py
from experiments.robot.openvla_utils import get_processor, get_vla_latent_action
from experiments.robot.robot_utils import get_model as get_openvla_model
from prismatic.models.policy.transformer_utils import MAPBlock

# Sane Defaults
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size=5, hidden_dim=512, action_dim=14, proprio_dim=14):
        super().__init__()
        self.attn_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
                                nn.Linear(proprio_dim, hidden_dim), 
                                nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim)
                            )
        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim * 2, window_size * action_dim), 
                    )
        self.window_size = window_size
        self.action_dim = action_dim

    def forward(self, latent_action_tokens, visual_embed, proprio):
        proprio = self.proprio_proj(proprio.to(torch.float))
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(torch.cat([
            self.attn_pool(latent_action_tokens.to(torch.float), init_embed=visual_embed), 
            proprio
        ], dim=-1))
        return action

@dataclass
class EvalConfig:
    # Model arguments
    model_family: str = "openvla"
    pretrained_checkpoint: str = "/MFVLA/models/0121pickbottle"
    action_decoder_path: str = "/MFVLA/models/0121pickbottle/action_decoder-20000.pt" 
    using_dino: bool = True
    dino_proj_ckpt: str = "/MFVLA/models/0121pickbottle/dino_proj-20000.pt"

    data_root: str = "/mnt/public/datasets/x2robot/pickbottle"
    dataset_stats_path: str = "/MFVLA/models/0121pickbottle/dataset_stats.pkl"
    
    instruction: str = "pick up the bottle and place it on the plate"
    window_size: int = 20
    action_horizon: int = 10
    action_dim: int = 7
    proprio_dim: int = 7

    center_crop: bool = False
    unnorm_key: Optional[str] = None
    device: str = "cuda"

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    task_suite_name: str = "pickbottle"

    ip: str = ''
    port: int = 57770


class Policy:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        print(f"Loading OpenVLA model from {cfg.pretrained_checkpoint}...")
        self.model = get_openvla_model(cfg).to(self.device).eval()
        self.processor = get_processor(cfg)
        
        # Load dataset stats for normalization
        self.dataset_stats = None
        if cfg.dataset_stats_path and os.path.isfile(cfg.dataset_stats_path):
            try:
                with open(cfg.dataset_stats_path, 'rb') as f:
                    self.dataset_stats = pickle.load(f)
                print(f"Loaded dataset stats from {cfg.dataset_stats_path}")
            except Exception as e:
                print(f"WARNING: Failed to load dataset stats: {e}")

        print(f"Loading ActionDecoder (Action Dim: {cfg.action_dim}, Proprio Dim: {cfg.proprio_dim})...")
        self.action_decoder = ActionDecoder(
            window_size=cfg.window_size, 
            action_dim=cfg.action_dim, 
            proprio_dim=cfg.proprio_dim
        ).to(self.device).eval()
        
        if cfg.action_decoder_path and os.path.exists(cfg.action_decoder_path):
            print(f"Loading ActionDecoder weights from {cfg.action_decoder_path}")
            # Use strict=False just in case, though structure should match
            self.action_decoder.load_state_dict(torch.load(cfg.action_decoder_path), strict=False)
            self.action_decoder.eval()
        else:
            print(f"WARNING: ActionDecoder path {cfg.action_decoder_path} invalid or not provided.")

        if cfg.using_dino:
            print("Loading DINOv2 model and projector...")
            self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(self.device).eval()
            self.dino_proj = nn.Linear(768, 4096).to(self.device)
            if cfg.dino_proj_ckpt and os.path.exists(cfg.dino_proj_ckpt):
                 print(f"Loading DINO projector weights from {cfg.dino_proj_ckpt}")
                 self.dino_proj.load_state_dict(torch.load(cfg.dino_proj_ckpt))
                 self.dino_proj.eval()
            else:
                 print(f"WARNING: DINO projector checkpoint {cfg.dino_proj_ckpt} invalid or not provided!")

        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()

        self.latent_action_detokenize = [f'<ACT_{i}>' for i in range(32)]
        self.reset()
    
    def reset(self):
        self.prev_hist_action = deque(maxlen=4)
        self.prev_hist_action.append('')
             
        self.current_instruction = self.cfg.instruction
        self.current_obs = None

    def update_obs(self, obs):
        self.current_obs = obs
        if 'instruction' in obs:
             self.current_instruction = obs['instruction']

    def get_action(self):
        if self.current_obs is None:
            return np.zeros((self.cfg.action_horizon, self.cfg.action_dim))

        image_numpy = self.current_obs['image']
        image_left_numpy = self.current_obs['image_left']
        image_right_numpy = self.current_obs['image_right']
        
        # Proprio processing
        proprio = torch.from_numpy(self.current_obs['proprio']).float().to(self.device).unsqueeze(0)
        
        # Normalize proprio
        if self.dataset_stats is not None:
             try:
                proprio_mean = self.dataset_stats["qpos_mean"]
                proprio_std  = self.dataset_stats["qpos_std"]
                
                if not isinstance(proprio_mean, torch.Tensor):
                    proprio_mean = torch.as_tensor(proprio_mean, device=proprio.device, dtype=proprio.dtype)
                else:
                    proprio_mean = proprio_mean.to(device=proprio.device, dtype=proprio.dtype)

                if not isinstance(proprio_std, torch.Tensor):
                    proprio_std = torch.as_tensor(proprio_std, device=proprio.device, dtype=proprio.dtype)
                else:
                    proprio_std = proprio_std.to(device=proprio.device, dtype=proprio.dtype)

                proprio = (proprio - proprio_mean) / (proprio_std + 1e-8)
             except Exception as e:
                print(f"Proprio norm error: {e}")

        instruction = self.current_instruction
        current_hist = self.prev_hist_action[-1]

        # Images
        image_pil_for_vla = Image.fromarray(image_numpy)
        image_left_pil = Image.fromarray(image_left_numpy)
        image_right_pil = Image.fromarray(image_right_numpy)

        # Resize to 224x224 match training
        resized_pil = self.resize(image_pil_for_vla)
        resized_left_pil = self.resize(image_left_pil)
        resized_right_pil = self.resize(image_right_pil)

        vla_image_np = np.asarray(resized_pil)
        vla_obs = {"full_image": vla_image_np}

        # VLA Inference - THIS IS THE FLOW REQUESTED
        latent_action, visual_embed, generated_ids = get_vla_latent_action(
            self.model, self.processor, self.cfg.pretrained_checkpoint, vla_obs, instruction, self.cfg.unnorm_key,
            center_crop=self.cfg.center_crop, hist_action=current_hist
        )

        if self.cfg.using_dino:
            image_t = self.to_tensor(resized_pil)
            image_left_t = self.to_tensor(resized_left_pil)
            image_right_t = self.to_tensor(resized_right_pil)
            
            all_views = torch.stack([image_t, image_left_t, image_right_t], dim=0) # [3, 3, 224, 224]
            pixel_values = self.dino_transform(all_views).to(self.device)

            with torch.no_grad():
                dino_out = self.dino_encoder.forward_features(pixel_values)
                dino_patches = dino_out["x_norm_patchtokens"] 
                visual_embed_dino = self.dino_proj(dino_patches).to(torch.float)
                
                # Reshape: [3, 256, 4096] -> [1, 768, 4096]
                head, left, right = visual_embed_dino[0:1], visual_embed_dino[1:2], visual_embed_dino[2:3]
                visual_embed = torch.cat([head, left, right], dim=1) # Overwrite visual_embed from VLA with DINO one if using_dino

        # Update History
        new_hist_action = ''
        if generated_ids is not None:
            for token_id in generated_ids[0]:
                idx = token_id.item() - 32001
                if 0 <= idx < 32:
                    new_hist_action += self.latent_action_detokenize[idx]
        self.prev_hist_action.append(new_hist_action)

        # Decode Action
        with torch.no_grad():
             pred_action = self.action_decoder(latent_action, visual_embed, proprio)
             pred_action = pred_action.reshape(1, self.cfg.window_size, self.cfg.action_dim)
             pred_action = pred_action.cpu().numpy()[0]
        # Un-normalize Actions
        if self.dataset_stats is not None:
            try:
                action_mean = self.dataset_stats.get("action_mean")
                action_std = self.dataset_stats.get("action_std")
             
                if action_mean is not None and action_std is not None:
                    if isinstance(action_mean, torch.Tensor): action_mean = action_mean.detach().cpu().numpy()
                    if isinstance(action_std, torch.Tensor): action_std = action_std.detach().cpu().numpy()
                    action_prediction = pred_action * action_std + action_mean
            except Exception as e:
                print(f"Action denorm error: {e}")
        return action_prediction.tolist() # Return list of floats

def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def read_img(conn):
    image_size = struct.unpack('<L', conn.recv(4))[0]
    image = recv_all(conn, image_size)
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB
    return image
def interpolates_actions(actions, target_num_actions = 80, action_dim=7):
    num_actions = actions.shape[0]
    original_indices = np.linspace(0, num_actions - 1, num_actions)
    target_indices = np.linspace(0, num_actions - 1, target_num_actions)
    interpolated_actions = np.zeros((target_num_actions, action_dim))
    if action_dim == 2: 
        for i in range(action_dim):
            interpolated_actions[:, i] = np.interp(target_indices, original_indices, actions[:, i])
        return interpolated_actions

    for i in range(3):
        interpolated_actions[:, i] = np.interp(target_indices, original_indices, actions[:, i])
    interpolated_actions[:, -1] = np.interp(target_indices, original_indices, actions[:, -1])
    quaternions = R.from_euler('xyz', actions[:, 3:6]).as_quat()  # shape: [num_actions, 4]
    interpolated_quats = np.zeros((target_num_actions, 4))
    for i in range(4): 
        interpolated_quats[:, i] = np.interp(target_indices, original_indices, quaternions[:, i])
    interpolated_quats = interpolated_quats / np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
    interpolated_eulers = R.from_quat(interpolated_quats).as_euler('xyz')  # shape: [target_num_actions, 3]
    interpolated_actions[:, 3:6] = interpolated_eulers
    return interpolated_actions
def main(

) -> None:
    
    cfg = EvalConfig()
    
    print("Initializing Policy...")
    policy = Policy(cfg)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(True)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((cfg.ip, cfg.port))
    sock.listen(1)
    print(f"Server is listening on {cfg.ip}:{cfg.port}")

    while True:
        try:
            print("Waiting for connection...")
            conn, addr = sock.accept()
            print(f"Connection from {addr}")
            policy.reset() # Reset history on new connection
            
            while True:
                head = conn.recv(4)
                if not head: break
                data_size_tuple = struct.unpack('<L', head)
                data_size = data_size_tuple[0]
                data = recv_all(conn, data_size)
                if not data: break
                action_data = json.loads(data.decode('utf8'))
                                                                             
                left_agent_data = action_data['follow1_pos'] 
                right_agent_data = action_data['follow2_pos'] 
                
                # Assume images are sent in order: Left, Face, Right
                image1 = read_img(conn) # Left
                image1[...] = 0.0
                image2 = read_img(conn) # Face
                image3 = read_img(conn) # Right
                
                # Make explicit float32 array to match downstream expectations
                curr_slave = np.array(right_agent_data, dtype=np.float32)
                
                obs = {
                    'image': image2,       # Face -> head_camera
                    'image_left': image1,  # Left -> left_camera
                    'image_right': image3, # Right -> right_camera
                    'proprio': curr_slave, # 28-dim
                    'instruction': cfg.instruction
                }
                
                policy.update_obs(obs)
                t0 = time.time()
                action = policy.get_action()
                t1 = time.time()
                print(f"Step time: {t1-t0:.3f}s. Action: {action[-1]}...")

                # Send action back
                # Protocol: action string "a,b,c..."
                # Previous protocol used json dumps, but deployment.py (attached) used struct pack for response.
                # Let's match the attached deployment.py RESPONSE format
                
                # Prepare follow1_pos (7 dims) and follow2_pos (7 dims) 
                # action is 14 dim flattened
                action_arr = np.array(action)
                follow1_pos = np.zeros((cfg.action_horizon,7))

                follow2_pos = action_arr

                follow1 = interpolates_actions(actions=follow1_pos, target_num_actions=100, action_dim=7)  # left eef
                follow2 = interpolates_actions(actions=follow2_pos[:cfg.action_horizon], target_num_actions=100, action_dim=7)  # right eef

       
                

                
                data_dir ={
                    "follow1_pos": follow1.tolist(),
                    "follow2_pos": follow2.tolist(), 
                }
    
                data_str = json.dumps(data_dir)
                data_bytes = data_str.encode('utf-8') 
                conn.sendall(struct.pack('<L', len(data_bytes)))
                conn.sendall(data_bytes)
                
        except Exception as e:
            print(f"Error in connection loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try: conn.close()
            except: pass

if __name__ == "__main__":
    tyro.cli(main)
