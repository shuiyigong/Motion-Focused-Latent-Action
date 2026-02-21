import numpy as np
import torch
import os
import glob
import h5py
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import fnmatch
import subprocess
import pickle
import re
from datetime import datetime
import cv2
import logging
from PIL import Image
from einops import rearrange, repeat
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision
import random
import json
from dataclasses import dataclass
from typing import Dict, Sequence, List
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]

        initial_pixel_values_hist, target_pixel_values_hist = [], []
        with_hist = []
        for instance in instances:
            if instance["initial_pixel_values_hist"] is not None:
                initial_pixel_values_hist.append(instance["initial_pixel_values_hist"])
                target_pixel_values_hist.append(instance["target_pixel_values_hist"])
                with_hist.append(torch.tensor(True))
            else:
                with_hist.append(torch.tensor(False))     



        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None


        # For low-level policy training
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions, dim=0)

        proprio = [instance["proprio"] for instance in instances]
        proprio = torch.stack(proprio, dim=0)

        instructions = [instance["lang"] for instance in instances]


        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        pixel_values = torch.stack(pixel_values)
        initial_pixel_values = torch.stack(initial_pixel_values)
        target_pixel_values = torch.stack(target_pixel_values)
        initial_pixel_values_hist = torch.stack(initial_pixel_values_hist) if len(initial_pixel_values_hist) > 0 else []
        target_pixel_values_hist = torch.stack(target_pixel_values_hist) if len(target_pixel_values_hist) > 0 else []
        with_hist = torch.stack(with_hist)

        # Handle wrist cameras if present
        extra_keys = {}
        for k in ["initial_pixel_values_left_wrist", "initial_pixel_values_right_wrist"]:
            if k in instances[0]:
                extra_keys[k] = torch.stack([instance[k] for instance in instances])

        output = dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist,
            target_pixel_values_hist=target_pixel_values_hist,
            instructions=instructions,
            with_hist=with_hist,
            actions=actions,
            proprio=proprio,
            **extra_keys
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


def load_data_univla(action_dim, proprio_dim, dataset_paths, camera_names, batch_size_train, action_tokenizer, processor, window_size,     
        min_window_size, max_window_size, image_transform, instruction=None):

    norm_stats = get_generic_realworld_norm_stats(dataset_paths,action_dim,proprio_dim)
    train_dataset = GenericVideoJsonDataset(
            action_dim, proprio_dim,
            dataset_paths, camera_names, norm_stats,
            window_size = window_size,
            min_window_size = min_window_size,
            max_window_size = max_window_size,
            image_transform = image_transform,
            instruction=instruction
        )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=8, prefetch_factor=2, collate_fn=collator,)


    return train_dataloader, norm_stats


def find_all_episodes(dataset_dir):
    data_dirs = []
    for root, dirs, files in os.walk(dataset_dir):
        dirname = os.path.basename(root)
        if os.path.exists(os.path.join(root, dirname + '.json')) and os.path.exists(os.path.join(root, 'faceImg.mp4')):
             data_dirs.append(root)
    data_dirs = sorted(data_dirs)
    if len(data_dirs) >= 50:
        print(f'Found {len(data_dirs)} episodes')
        return data_dirs[0:50]

def get_generic_realworld_norm_stats(dataset_dirs, action_dim, proprio_dim):
    # Try to load cached stats if available
    if len(dataset_dirs) > 0:
        common_root = os.path.dirname(dataset_dirs[0])   
        stats_path = os.path.join(common_root, 'dataset_stats.pkl')
        if os.path.exists(stats_path):
            print(f"Loading existing dataset stats from {stats_path}")
            try:
                with open(stats_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load stats: {e}, recomputing...")
    
    all_qpos_data = []
    all_action_data = []
    print("Computing normalization stats for generic realworld dataset...")
    for ep_dir in dataset_dirs:
        ep_name = os.path.basename(ep_dir)
        json_path = os.path.join(ep_dir, ep_name + '.json')
        if not os.path.exists(json_path):
             continue
        with open(json_path, 'r') as f:
             data = json.load(f)['data']
        
        curr_qpos = []
        curr_actions = []
        for entry in data:
            # Proprio
            try:
                fl_pos = entry['follow_left_position']
                fl_rot = entry['follow_left_rotation']
                fl_grip = [entry['follow_left_gripper']]
                fr_pos = entry['follow_right_position']
                fr_rot = entry['follow_right_rotation']
                fr_grip = [entry['follow_right_gripper']]
                if proprio_dim == 7:
                    p = np.concatenate([
                        fr_pos, fr_rot, fr_grip
                    ])
                elif proprio_dim == 14:
                    p = np.concatenate([
                        fl_pos, fl_rot, fl_grip,
                        fr_pos, fr_rot, fr_grip
                    ])

                # Action (unchanged): master positions + follow grippers -> 14 dim
                # Note: actions still reference master positions if present in json
                ml_pos = entry.get('master_left_position', fl_pos)
                ml_rot = entry.get('master_left_rotation', fl_rot)
                mr_pos = entry.get('master_right_position', fr_pos)
                mr_rot = entry.get('master_right_rotation', fr_rot)
                ml_grip = [entry.get('master_left_gripper', fl_grip[0])]
                mr_grip = [entry.get('master_right_gripper', fr_grip[0])]
                if action_dim == 7:
                    a = np.concatenate([
                        mr_pos, mr_rot, mr_grip 
                    ])
                elif action_dim == 14:
                    a = np.concatenate([
                        ml_pos, ml_rot, ml_grip,
                        mr_pos, mr_rot, mr_grip
                    ])
                curr_qpos.append(p)
                curr_actions.append(a)
            except KeyError as e:
                print(f"KeyError in {ep_dir}: {e}")
                continue
        
        if len(curr_qpos) > 0:
            all_qpos_data.append(torch.tensor(np.array(curr_qpos), dtype=torch.float32))
            all_action_data.append(torch.tensor(np.array(curr_actions), dtype=torch.float32))

    if len(all_qpos_data) == 0:
         print("No data found for stats!")
         return None

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # Robust normalization using percentiles (q1 / q99) -> use median as center and (q99 - q1)/2 as scale
    try:
        # Actions
        action_q01 = torch.quantile(all_action_data, 0.01, dim=0, keepdim=True)
        action_q99 = torch.quantile(all_action_data, 0.99, dim=0, keepdim=True)
        action_median = torch.quantile(all_action_data, 0.5, dim=0, keepdim=True)
        action_scale = (action_q99 - action_q01) / 2.0
        action_scale = torch.clip(action_scale, 1e-3, np.inf)

        # Qpos (proprio)
        qpos_q01 = torch.quantile(all_qpos_data, 0.01, dim=0, keepdim=True)
        qpos_q99 = torch.quantile(all_qpos_data, 0.99, dim=0, keepdim=True)
        qpos_median = torch.quantile(all_qpos_data, 0.5, dim=0, keepdim=True)
        qpos_scale = (qpos_q99 - qpos_q01) / 2.0
        qpos_scale = torch.clip(qpos_scale, 1e-3, np.inf)
    except Exception as e:
        raise RuntimeError(f"Failed to compute quantiles for normalization stats: {e}")
    
    stats = {
        # Keep keys same for compatibility but values now reflect median/robust-scale (q99-based)
        "action_mean": action_median.numpy().squeeze(),
        "action_std": action_scale.numpy().squeeze(),
        "qpos_mean": qpos_median.numpy().squeeze(),
        "qpos_std": qpos_scale.numpy().squeeze(),
        "example_qpos": all_qpos_data[0].numpy(),
    }
    print("Generic Realworld Stats computed.")

    # Plot distributions for each action / qpos dimension and save next to stats file
    try:
        common_root = None
        if len(dataset_dirs) > 0:
            common_root = os.path.dirname(dataset_dirs[0])
        else:
            # fallback to current working dir
            common_root = os.getcwd()

        # Actions distribution plot
        action_np = all_action_data.numpy()
        action_med = action_median.numpy().squeeze()
        n_actions = action_np.shape[1]
        ncols = 4
        nrows = (n_actions + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * max(1, nrows)))
        axes = axes.reshape(-1)
        for i in range(n_actions):
            ax = axes[i]
            ax.hist(action_np[:, i], bins=100, color='C0', alpha=0.8)
            ax.axvline(action_med[i], color='r', linestyle='--', label=f'median={action_med[i]:.4f}')
            ax.set_title(f'action_dim_{i}')
            ax.legend(fontsize='small')
        # hide extra axes
        for j in range(n_actions, len(axes)):
            axes[j].axis('off')
        actions_plot_path = os.path.join(common_root, 'action_distributions.png')
        fig.tight_layout()
        fig.savefig(actions_plot_path)
        plt.close(fig)

        # Qpos (proprio) distribution plot
        qpos_np = all_qpos_data.numpy()
        qpos_med = qpos_median.numpy().squeeze()
        n_qpos = qpos_np.shape[1]
        ncols = 4
        nrows = (n_qpos + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * max(1, nrows)))
        axes = axes.reshape(-1)
        for i in range(n_qpos):
            ax = axes[i]
            ax.hist(qpos_np[:, i], bins=100, color='C1', alpha=0.8)
            ax.axvline(qpos_med[i], color='r', linestyle='--', label=f'median={qpos_med[i]:.4f}')
            ax.set_title(f'qpos_dim_{i}')
            ax.legend(fontsize='small')
        for j in range(n_qpos, len(axes)):
            axes[j].axis('off')
        qpos_plot_path = os.path.join(common_root, 'qpos_distributions.png')
        fig.tight_layout()
        fig.savefig(qpos_plot_path)
        plt.close(fig)

        print(f"Saved distribution plots to: {actions_plot_path} and {qpos_plot_path}")
    except Exception as e:
        print(f"Failed to save distribution plots: {e}")
    # Save computed stats to cache
    if len(dataset_dirs) > 0:
        common_root = os.path.dirname(dataset_dirs[0])
        stats_path = os.path.join(common_root, 'dataset_stats.pkl')
        print(f"Saving computed dataset stats to {stats_path}")
        try:
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
        except Exception as e:
            print(f"Failed to save stats cache: {e}")
            
    return stats


class GenericVideoJsonDataset(torch.utils.data.Dataset):
    def __init__(self, action_dim, proprio_dim, dataset_dirs, 
                 camera_names, 
                 norm_stats, 
                 window_size=16,
                 min_window_size=16,
                 max_window_size=16,
                 image_transform=None,
                 other_config=(),
                 instruction=None) -> None:
        
        super().__init__()
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.dataset_dirs = dataset_dirs
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = window_size
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.image_transform = image_transform
        
        self.resize_img = torchvision.transforms.Resize((224, 224))
        self.image_transform_lam = torchvision.transforms.ToTensor()
        self.color_aug = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        
        # Load all episodes
        self.image_dict, self.qpos, self.action, self.instructions, self.episode_lens = self.load_all_episodes(dataset_dirs, instruction)
        self.indices = []
        for ep_idx, ep_len in enumerate(self.episode_lens):
           if ep_len > 1:
               for start_idx in range(ep_len - 1):
                   self.indices.append((ep_idx, start_idx))

    def __len__(self):
        return len(self.indices)

    def load_all_episodes(self, dataset_dirs, instruction):
        image_dict = {cam: [] for cam in self.camera_names}
        qpos_list = []
        actions_list = []
        instructions = []
        episode_lens = []

        # Map camera names to filenames
        # Default mapping based on user description
        cam_file_map = {
            "left_wrist": "leftImg.mp4",
            "right_wrist": "rightImg.mp4",
            "face": "faceImg.mp4"
        }

        for ep_dir in dataset_dirs:
            print(f"Loading {ep_dir}")
            ep_name = os.path.basename(ep_dir)
            json_path = os.path.join(ep_dir, ep_name + '.json')
            
            if not os.path.exists(json_path):
                print(f"Skipping {ep_dir}, json not found")
                continue
                
            with open(json_path, 'r') as f:
                data = json.load(f)['data']
            
            ep_len = len(data)
            episode_lens.append(ep_len)
            instructions.append(instruction)
        
            curr_qpos = []
            curr_actions = []
            
            for entry in data:
                fl_pos = entry['follow_left_position']
                fl_rot = entry['follow_left_rotation']
                fl_grip = [entry['follow_left_gripper']]
                fr_pos = entry['follow_right_position']
                fr_rot = entry['follow_right_rotation']
                fr_grip = [entry['follow_right_gripper']]
                
                if self.proprio_dim == 7:
                    p = np.concatenate([
                        fr_pos, fr_rot, fr_grip
                    ])
                elif self.proprio_dim == 14:
                    p = np.concatenate([
                        fl_pos, fl_rot, fl_grip,
                        fr_pos, fr_rot, fr_grip
                    ])
                curr_qpos.append(p)
                
                # Action
                ml_pos = entry.get('master_left_position')
                ml_rot = entry.get('master_left_rotation')
                mr_pos = entry.get('master_right_position')
                mr_rot = entry.get('master_right_rotation')
                ml_grip = [entry.get('master_left_gripper', fl_grip[0])]
                mr_grip = [entry.get('master_right_gripper', fr_grip[0])]
                if self.action_dim == 7:
                    a = np.concatenate([
                        mr_pos, mr_rot, mr_grip
                    ])
                elif self.action_dim == 14:
                    a = np.concatenate([
                        ml_pos, ml_rot, ml_grip,
                        mr_pos, mr_rot, mr_grip
                    ])
                curr_actions.append(a)
            
            qpos_list.append(torch.tensor(np.array(curr_qpos), dtype=torch.float32))
            actions_list.append(torch.tensor(np.array(curr_actions), dtype=torch.float32))

            # Load Images
            for cam_name in self.camera_names:
                filename = cam_file_map.get(cam_name, None)                
                vid_path = os.path.join(ep_dir, filename)
                frame_list = []
                
                if os.path.exists(vid_path):
                    cap = cv2.VideoCapture(vid_path)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Resize
                        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
                        frame_list.append(torch.from_numpy(frame)) # HWC
                    cap.release()
                else:
                    print(f"Warning: {filename} not found in {ep_dir}")
                    frame_list = [torch.zeros(3, 224, 224) for _ in range(ep_len)]

                # Check length consistency
                if len(frame_list) != ep_len:
                    print(f"Warning: Frame count mismatch in {ep_dir} for {cam_name}: video frames = {len(frame_list)}, json entries = {ep_len}")
                    # Handle mismatch (video vs json length)
                    min_len = min(len(frame_list), ep_len)
                    frame_list = frame_list[:min_len]
                    if len(frame_list) < ep_len:
                         # Truncate others
                         qpos_list[-1] = qpos_list[-1][:min_len]
                         actions_list[-1] = actions_list[-1][:min_len]
                         episode_lens[-1] = min_len
                         ep_len = min_len
                
                image_dict[cam_name].append(torch.stack(frame_list))

        return image_dict, qpos_list, actions_list, instructions, episode_lens

    def __getitem__(self, index):
        ep_idx, start_frame = self.indices[index]
        ep_len = self.episode_lens[ep_idx] 
        w_size = self.window_size


        action_chunk = self.action[ep_idx][start_frame+1 : start_frame+1 + w_size]
        actions_chunking = action_chunk
        
        # Pad actions if near the end of the episode (repeat last action)
        if len(action_chunk) < w_size:
            pad_len = w_size - len(action_chunk)
            last_action = action_chunk[-1].unsqueeze(0)
            padding = last_action.repeat(pad_len, 1)
            actions_chunking = torch.cat([action_chunk, padding], dim=0)  
 
        qpos_chunking = self.qpos[ep_idx][start_frame]
        
        ret_dict = {}
        
        # Images
        if "face" in self.camera_names:
            face = self.image_dict["face"][ep_idx][start_frame] 
            face_pil = Image.fromarray(face.numpy().astype(np.uint8))
            face_aug = self.color_aug(face_pil)

            target = self.image_dict["face"][ep_idx][min(start_frame + self.window_size - 1, ep_len - 1)]
            target_pil = Image.fromarray(target.numpy().astype(np.uint8))
            target_aug = self.color_aug(target_pil)
            ret_dict["pixel_values"] = self.image_transform(face_aug)
            ret_dict["initial_pixel_values"] = self.image_transform_lam(self.resize_img(face_aug))
            ret_dict["target_pixel_values"] = self.image_transform_lam(self.resize_img(target_aug))
        
        # Wrists
        for cam in ["left_wrist", "right_wrist"]:
            if cam in self.camera_names:
                wrist_t = self.image_dict[cam][ep_idx][start_frame]
                wrist_pil = Image.fromarray(wrist_t.numpy().astype(np.uint8))
                wrist_aug = self.color_aug(wrist_pil)
                ret_dict[f"initial_pixel_values_{cam}"] = self.image_transform_lam(self.resize_img(wrist_aug))
            else:
                ret_dict[f"initial_pixel_values_{cam}"] = torch.zeros(3, 224, 224)

        # Normalize Actions/Proprio
        action_tensor = actions_chunking.float()
        action_tensor = (action_tensor - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        
        qpos_tensor = qpos_chunking.float()
        qpos_tensor = (qpos_tensor - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        ret_dict["dataset_name"] = "pickbottle"
        ret_dict["actions"] = action_tensor
        ret_dict["lang"] = self.instructions[ep_idx]
        ret_dict["proprio"] = qpos_tensor
        
        # Hist placeholders for existing collator compatibility
        ret_dict["initial_pixel_values_hist"] = None
        ret_dict["target_pixel_values_hist"] = None
        
        return ret_dict


