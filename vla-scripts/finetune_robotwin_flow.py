import os
from collections import deque, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Sequence
import cv2
import json
import glob
import math
import draccus
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import tqdm
import pickle
from PIL import Image
import numpy as np
from ema_pytorch import EMA
from accelerate import PartialState, Accelerator
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


os.environ["TOKENIZERS_PARALLELISM"] = "false"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
import h5py


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        initial_pixel_values_left = [instance["initial_pixel_values_left"] for instance in instances]
        initial_pixel_values_right = [instance["initial_pixel_values_right"] for instance in instances]
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

        # For low-level policy training
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions, dim=0)

        proprio = [instance["proprio"] for instance in instances]
        proprio = torch.stack(proprio, dim=0)

        instructions = [instance["lang"] for instance in instances]

        # Stack all `pixel_values`
        pixel_values = torch.stack(pixel_values)
        initial_pixel_values = torch.stack(initial_pixel_values)
        initial_pixel_values_left = torch.stack(initial_pixel_values_left)
        initial_pixel_values_right = torch.stack(initial_pixel_values_right)
        target_pixel_values = torch.stack(target_pixel_values)
        initial_pixel_values_hist = torch.stack(initial_pixel_values_hist) if len(initial_pixel_values_hist) > 0 else []
        target_pixel_values_hist = torch.stack(target_pixel_values_hist) if len(target_pixel_values_hist) > 0 else []
        with_hist = torch.stack(with_hist)

        output = dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            initial_pixel_values_left=initial_pixel_values_left,
            initial_pixel_values_right=initial_pixel_values_right,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist,
            target_pixel_values_hist=target_pixel_values_hist,
            instructions=instructions,
            with_hist=with_hist,
            actions=actions,
            proprio=proprio
        )
        return output


class RoboTwinDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, instruction_dir, camera_names, norm_stats,
                 window_size=16, min_window_size=16, max_window_size=16,
                 image_transform=None):
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.instruction_dir = instruction_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.image_transform = image_transform

        self.resize_img = transforms.Resize((224, 224))
        self.image_transform_lam = transforms.ToTensor()
        self.color_aug = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

        self.episodes = self.load_episodes()

        self.indices = []
        for ep_idx, ep_data in enumerate(self.episodes):
            ep_len = ep_data['len']
            # Modified: Allow sampling up to the end of episode.
            # We need at least 1 future frame for action (at t+1).
            # So start_idx can go up to ep_len - 2.
            if ep_len > 1:
                for start_idx in range(ep_len - 1):
                    self.indices.append((ep_idx, start_idx))

    def load_episodes(self):
        hdf5_files = sorted(glob.glob(os.path.join(self.dataset_dir, '*.hdf5')))

        episodes = []

        for h5_path in hdf5_files:
            file_name = os.path.basename(h5_path)
            ep_name = os.path.splitext(file_name)[0]

            json_path = os.path.join(self.instruction_dir, f"{ep_name}.json")
            if not os.path.exists(json_path):
                instruction = ""
            else:
                with open(json_path, 'r') as f:
                    instr_data = json.load(f)
                instruction = instr_data.get('seen', [""])[0] if instr_data.get('seen') else ""

            with h5py.File(h5_path, 'r') as f:
                rgb_ds = f['observation/head_camera/rgb']
                rgb_ds_left = f['observation/left_camera/rgb']
                rgb_ds_right = f['observation/right_camera/rgb']
                num_frames = rgb_ds.shape[0]
                images = []
                images_left = []
                images_right = []
                for i in range(num_frames):
                    img_data = rgb_ds[i]
                    if isinstance(img_data, np.ndarray):
                        img_data = img_data.tobytes()
                    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(Image.fromarray(img))

                    img_data_left = rgb_ds_left[i]
                    if isinstance(img_data_left, np.ndarray):
                        img_data_left = img_data_left.tobytes()
                    img_left = cv2.imdecode(np.frombuffer(img_data_left, np.uint8), cv2.IMREAD_COLOR)
                    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                    images_left.append(Image.fromarray(img_left))

                    img_data_right = rgb_ds_right[i]
                    if isinstance(img_data_right, np.ndarray):
                        img_data_right = img_data_right.tobytes()
                    img_right = cv2.imdecode(np.frombuffer(img_data_right, np.uint8), cv2.IMREAD_COLOR)
                    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                    images_right.append(Image.fromarray(img_right))

                actions = f['joint_action/vector'][:]
                qpos = actions

            episodes.append({
                'images': images,
                'images_left': images_left,
                'images_right': images_right,
                'actions': torch.tensor(actions, dtype=torch.float32),
                'proprio': torch.tensor(qpos, dtype=torch.float32),
                'instruction': instruction,
                'len': num_frames
            })
        print(f"Loaded {len(episodes)} episodes.")
        return episodes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start_frame = self.indices[idx]
        episode = self.episodes[ep_idx]

        w_size = self.window_size

        # action is t+1
        action_chunk = episode['actions'][start_frame+1: start_frame+1 + w_size]

        # Pad actions if near the end of the episode (repeat last action)
        if len(action_chunk) < w_size:
            pad_len = w_size - len(action_chunk)
            # Repeat the last action
            last_action = action_chunk[-1].unsqueeze(0)
            padding = last_action.repeat(pad_len, 1)
            action_chunk = torch.cat([action_chunk, padding], dim=0)

        proprio_chunk = episode['proprio'][start_frame]

        image_vla_pil = self.resize_img(episode['images'][start_frame])
        image_vla_pil_aug = self.color_aug(image_vla_pil)

        image_left_pil = self.resize_img(episode['images_left'][start_frame])
        image_left_pil_aug = self.color_aug(image_left_pil)

        image_right_pil = self.resize_img(episode['images_right'][start_frame])
        image_right_pil_aug = self.color_aug(image_right_pil)

        target_image_pil = self.resize_img(episode['images'][min(start_frame + w_size - 1, episode['len']-1)])
        target_image_pil_aug = self.color_aug(target_image_pil)

        pixel_values = self.image_transform(image_vla_pil_aug)

        initial_pixel_values = self.image_transform_lam(image_vla_pil_aug)
        initial_pixel_values_left = self.image_transform_lam(image_left_pil_aug)
        initial_pixel_values_right = self.image_transform_lam(image_right_pil_aug)
        target_pixel_values = self.image_transform_lam(target_image_pil_aug)

        initial_pixel_values_hist = None
        target_pixel_values_hist = None

        action_tensor = (action_chunk - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        proprio_tensor = (proprio_chunk - self.norm_stats["proprio_mean"]) / self.norm_stats["proprio_std"]

        task_instr = episode['instruction']

        return dict(pixel_values=pixel_values, initial_pixel_values=initial_pixel_values,
                    initial_pixel_values_left=initial_pixel_values_left, initial_pixel_values_right=initial_pixel_values_right,
                    target_pixel_values=target_pixel_values,
                    initial_pixel_values_hist=initial_pixel_values_hist, target_pixel_values_hist=target_pixel_values_hist,
                    dataset_name='robotwin', actions=action_tensor, lang=task_instr, proprio=proprio_tensor)


def get_norm_stats_robotwin(dataset_dir, instruction_dir):
    hdf5_files = sorted(glob.glob(os.path.join(dataset_dir, '*.hdf5')))
    all_actions = []
    all_proprios = []

    for h5_path in hdf5_files:
        with h5py.File(h5_path, 'r') as f:
            actions = f['joint_action/vector'][:]
            all_actions.append(torch.from_numpy(actions))
            # proprio is same as action (joint_action)
            all_proprios.append(torch.from_numpy(actions))

    all_actions = torch.cat(all_actions, dim=0)
    all_proprios = torch.cat(all_proprios, dim=0)

    action_mean = all_actions.mean(dim=0, keepdim=True)
    action_std = all_actions.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    proprio_mean = all_proprios.mean(dim=0, keepdim=True)
    proprio_std = all_proprios.std(dim=0, keepdim=True)
    proprio_std = torch.clip(proprio_std, 1e-2, np.inf)

    return {
        "action_mean": action_mean, "action_std": action_std,
        "proprio_mean": proprio_mean, "proprio_std": proprio_std
    }


def load_data_robotwin(dataset_dir, instruction_dir, camera_names, batch_size_train, action_tokenizer, processor, window_size,
                       min_window_size, max_window_size, image_transform):

    norm_stats = get_norm_stats_robotwin(dataset_dir, instruction_dir)

    train_dataset = RoboTwinDataset(None, dataset_dir, instruction_dir, camera_names, norm_stats,
        window_size=window_size,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        image_transform=image_transform,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=8, prefetch_factor=2, collate_fn=collator)

    return train_dataloader, norm_stats


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    """1D 正余弦位置编码，来自 MAE。"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim: int, grid_sizes):
    """ND 网格位置编码，逐维累加。"""
    emb = np.zeros(grid_sizes + (embed_dim,))
    for size_idx, grid_size in enumerate(grid_sizes):
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [embed_dim]
        posemb_shape[size_idx] = -1
        emb += get_1d_sincos_pos_embed_from_grid(embed_dim, pos).reshape(posemb_shape)
    return emb


def get_multimodal_pos_embed(embed_dim: int, mm_lens: OrderedDict):
    tot_len = 0
    for modality, cond_len in mm_lens.items():
        if modality == "image" and (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            tot_len += np.prod([abs(x) for x in cond_len])
        else:
            tot_len += abs(cond_len)

    num_modalities = len(mm_lens)
    modality_pos_embed = None
    if num_modalities > 1:
        modality_pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(num_modalities))

    pos_emb = np.zeros((tot_len, embed_dim))
    start_pos = 0
    for idx, (modality, cond_len) in enumerate(mm_lens.items()):
        if modality == "image" and (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            pos_embed_i_ = get_nd_sincos_pos_embed_from_grid(embed_dim, embed_grid_sizes)
            pos_embed_i = np.zeros(all_grid_sizes + (embed_dim,))
            pos_embed_i += pos_embed_i_
            pos_embed_i = pos_embed_i.reshape((-1, embed_dim))
        else:
            pos_embed_i_ = get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(cond_len)) if cond_len > 1 else 0
            pos_embed_i = np.zeros((abs(cond_len), embed_dim))
            pos_embed_i += pos_embed_i_

        if modality_pos_embed is not None:
            pos_embed_i += modality_pos_embed[idx]

        pos_emb[start_pos:start_pos + len(pos_embed_i)] = pos_embed_i
        start_pos += len(pos_embed_i)

    return pos_emb


def modulate(x, shift, scale):
    """AdaLN 调制：x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (与 HRDT 一致)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """自注意力（支持 FlashAttn / GQA），对齐 HRDT 版本。"""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, norm_eps: float, use_flash_attn: bool):
        super().__init__()
        self.n_heads = num_heads
        self.n_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_heads should be divisible by num_kv_heads")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = hidden_size
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads")
        self.head_size = self.hidden_size // self.n_heads

        self.wq = nn.Linear(self.hidden_size, self.n_heads * self.head_size, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_size * 2, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_size, self.hidden_size, bias=False)

        self.norm_q = RMSNorm(self.head_size, eps=norm_eps)
        self.norm_k = RMSNorm(self.head_size, eps=norm_eps)
        self.use_flash_attn = use_flash_attn
        self.attn_scale = 1.0 / math.sqrt(self.head_size)

    def forward(self, x: torch.Tensor):
        bs, seq_len, _ = x.shape
        xq = self.wq(x).view(bs, seq_len, self.n_heads, self.head_size)
        xkv = self.wkv(x).view(bs, seq_len, self.n_kv_heads, self.head_size, 2)
        xk, xv = xkv.unbind(-1)

        xq, xk = self.norm_q(xq), self.norm_k(xk)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.use_flash_attn:
            output = F.scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.attn_scale,
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)


class CrossAttention(nn.Module):
    """交叉注意力（支持 FlashAttn / GQA），对齐 HRDT 版本。"""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, norm_eps: float, use_flash_attn: bool):
        super().__init__()
        self.n_heads = num_heads
        self.n_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_heads should be divisible by num_kv_heads")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = hidden_size
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads")
        self.head_size = self.hidden_size // self.n_heads

        self.wq = nn.Linear(self.hidden_size, self.n_heads * self.head_size, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_size * 2, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_size, self.hidden_size, bias=False)

        self.norm_q = RMSNorm(self.head_size, eps=norm_eps)
        self.norm_k = RMSNorm(self.head_size, eps=norm_eps)
        self.use_flash_attn = use_flash_attn
        self.attn_scale = 1.0 / math.sqrt(self.head_size)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bs, seq_len, _ = x.shape
        _, c_len, _ = c.shape

        xq = self.wq(x).view(bs, seq_len, self.n_heads, self.head_size)
        ckv = self.wkv(c).view(bs, c_len, self.n_kv_heads, self.head_size, 2)
        ck, cv = ckv.unbind(-1)

        xq, ck = self.norm_q(xq), self.norm_k(ck)
        ck = repeat_kv(ck, self.n_rep)
        cv = repeat_kv(cv, self.n_rep)

        xq = xq.transpose(1, 2)
        ck = ck.transpose(1, 2)
        cv = cv.transpose(1, 2)

        if mask is not None:
            mask = mask.reshape(bs, 1, 1, c_len)
            mask = mask.expand(-1, -1, seq_len, -1)

        if self.use_flash_attn:
            output = F.scaled_dot_product_attention(
                query=xq,
                key=ck,
                value=cv,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.attn_scale,
            )
        else:
            scores = torch.matmul(xq, ck.transpose(2, 3)) * self.attn_scale
            if mask is not None:
                scores = scores.masked_fill_(mask.logical_not(), float('-inf'))
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, cv)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """HRDT 风格的 FFN（SwiGLU 变体 & multiple_of 对齐）。"""

    def __init__(self, dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_size = frequency_embedding_size
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half = self.freq_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(t_freq)


HRDT_DEFAULTS = dict(
    hidden_size=512,
    num_heads=16,
    num_kv_heads=8,
    norm_eps=1e-5,
    multiple_of=256,
    ffn_dim_multiplier=None,
    use_flash_attn=True,
)


class FlowBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = None,
        n_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        norm_eps: Optional[float] = None,
        multiple_of: Optional[int] = None,
        ffn_dim_multiplier: Optional[float] = None,
        use_flash_attn: Optional[bool] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim if hidden_dim is not None else HRDT_DEFAULTS["hidden_size"]
        self.n_heads = n_heads if n_heads is not None else HRDT_DEFAULTS["num_heads"]
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else HRDT_DEFAULTS["num_kv_heads"]
        self.norm_eps = norm_eps if norm_eps is not None else HRDT_DEFAULTS["norm_eps"]
        self.multiple_of = multiple_of if multiple_of is not None else HRDT_DEFAULTS["multiple_of"]
        self.ffn_dim_multiplier = ffn_dim_multiplier if ffn_dim_multiplier is not None else HRDT_DEFAULTS["ffn_dim_multiplier"]
        self.use_flash_attn = HRDT_DEFAULTS["use_flash_attn"]
        # Self-Attention（HRDT 注意力）
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.self_attn = Attention(
            hidden_size=self.hidden_dim,
            num_heads=self.n_heads,
            num_kv_heads=self.num_kv_heads,
            norm_eps=self.norm_eps,
            use_flash_attn=self.use_flash_attn,
        )

        # Cross-Attention（视觉条件）
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.vis_cond_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attn = CrossAttention(
            hidden_size=self.hidden_dim,
            num_heads=self.n_heads,
            num_kv_heads=self.num_kv_heads,
            norm_eps=self.norm_eps,
            use_flash_attn=self.use_flash_attn,
        )

        # FFN（HRDT 风格）
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        self.ffn = FeedForward(
            dim=self.hidden_dim,
            multiple_of=self.multiple_of,
            ffn_dim_multiplier=self.ffn_dim_multiplier,
        )

        # AdaLN 调制，针对 self / cross / ffn 三个子层各自学习 shift / scale / gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 9 * self.hidden_dim),
        )

    def forward(self, x, t_emb, vis_context, vis_mask: Optional[torch.Tensor] = None):
        # 拆分自适应调制参数
        shift_sa, scale_sa, gate_sa, \
        shift_cross, scale_cross, gate_cross, \
        shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(t_emb).chunk(9, dim=1)

        gate_sa = gate_sa.unsqueeze(1)       # [B, 1, D]
        gate_cross = gate_cross.unsqueeze(1) # [B, 1, D]
        gate_ffn = gate_ffn.unsqueeze(1)     # [B, 1, D]

        # 1) Self-Attention with AdaLN + HRDT attn
        sa_in = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa * self.self_attn(sa_in)

        # 2) Cross-Attention with AdaLN（视觉条件也做归一化）
        cross_q = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_kv = self.vis_cond_norm(vis_context)
        x = x + gate_cross * self.cross_attn(cross_q, cross_kv, vis_mask)

        # 3) FFN with AdaLN（HRDT FFN）
        ffn_in = modulate(self.norm3(x), shift_ffn, scale_ffn)
        x = x + gate_ffn * self.ffn(ffn_in)

        return x

class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size: int = 15, hidden_dim: int = HRDT_DEFAULTS["hidden_size"], action_dim: int = 14, proprio_dim: int = 14, n_latents: int = 1, depth: int = 4):
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.n_latents = n_latents

        # 时间与输入投影
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.action_in = nn.Linear(action_dim, hidden_dim)
        # 正余弦 action 位置编码（HRDT 风格）
        self.action_pos_emb = nn.Parameter(torch.zeros(1, window_size, hidden_dim))

        # 保留空间分辨率的视觉/语义/本体感编码
        self.vis_proj = nn.Linear(4096, hidden_dim)
        self.vla_token_proj = nn.Linear(4096, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)

        self._init_action_pos_emb()

        # 多层 HRDT blocks
        self.blocks = nn.ModuleList([
            FlowBlock(
                hidden_dim=hidden_dim,
                n_heads=HRDT_DEFAULTS["num_heads"],
                num_kv_heads=HRDT_DEFAULTS["num_kv_heads"],
                norm_eps=HRDT_DEFAULTS["norm_eps"],
                multiple_of=HRDT_DEFAULTS["multiple_of"],
                ffn_dim_multiplier=HRDT_DEFAULTS["ffn_dim_multiplier"],
                use_flash_attn=HRDT_DEFAULTS["use_flash_attn"],
            ) for _ in range(depth)
        ])

        # 最终 AdaLN + 预测头（对齐 H-RDT：归一化 + AdaLN 调制 + MLP，预测层零初始化稳定训练）
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.final_proj = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.final_proj.weight, 0.0)
        nn.init.constant_(self.final_proj.bias, 0.0)

    def _init_action_pos_emb(self):
        action_pos = get_multimodal_pos_embed(
            embed_dim=self.hidden_dim,
            mm_lens=OrderedDict([("action", self.window_size)])
        )
        with torch.no_grad():
            self.action_pos_emb.data.copy_(torch.from_numpy(action_pos).float().unsqueeze(0))

    def forward(self, latent_action_tokens: torch.Tensor, visual_embed: torch.Tensor, proprio: torch.Tensor, actions: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = latent_action_tokens.shape[0]

        # 视觉上下文保留 patch 维度
        vis_context = self.vis_proj(visual_embed)
        vla_context = self.vla_token_proj(latent_action_tokens)
        proprio_context = self.proprio_proj(proprio.to(vis_context.dtype))

        context_mm_lens = OrderedDict([
            ("image", vis_context.shape[1]),
            ("vla", vla_context.shape[1]),
            ("proprio", proprio_context.shape[1]),
        ])

        full_context = torch.cat([vis_context, vla_context, proprio_context], dim=1)
        # 时间采样 / 嵌入
        if t is None:
            t = torch.rand(B, device=vis_context.device, dtype=vis_context.dtype)
        t_emb = self.t_embedder(t)

        # Flow Matching 噪声与目标

        noise = torch.randn_like(actions)
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * noise + t_expand * actions
        target = actions - noise
 
        x = self.action_in(x_t)
        x = x + self.action_pos_emb[:, : x.shape[1]].to(x.dtype)

        for block in self.blocks:
            x = block(x, t_emb, full_context)

        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        v_pred = self.final_proj(x)


        return F.mse_loss(v_pred, target, reduction='none')



class Wrapped_Model(torch.nn.Module):
    def __init__(self, vla, freeze_vla=False, window_size=12, action_dim=14, proprio_dim=14, n_latents=1):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.action_decoder = ActionDecoder(window_size=window_size, action_dim=action_dim, proprio_dim=proprio_dim, n_latents=n_latents)
        self.dino_transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

        self.dino_encoder = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14_reg'
        )
        self.dino_encoder.to(torch.bfloat16)
        self.dino_encoder.requires_grad_(False)
        self.dino_proj = nn.Linear(768, 4096)
        self.dino_proj.to(torch.bfloat16)

        if freeze_vla:
            self.vla.requires_grad_(False)

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states=True,        # Return intermediate tokens of all layers
            )
        loss, latent_action_tokens = self.action_decoder_forward(batch, vla_output)

        return vla_output, loss, latent_action_tokens

    def action_decoder_forward(self, batch, slow_output):
        # Collect all images [Head, Left, Right]
        pixel_values = batch["initial_pixel_values"]  # [B, 3, H, W]
        pixel_values_left = batch["initial_pixel_values_left"]
        pixel_values_right = batch["initial_pixel_values_right"]

        all_views = torch.cat([pixel_values, pixel_values_left, pixel_values_right], dim=0)
        all_views = self.dino_transform(all_views)

        with torch.no_grad():
            dino_out = self.dino_encoder.forward_features(all_views)
            dino_patches = dino_out["x_norm_patchtokens"]  # [3B, 256, 768]

        visual_embed_all = self.dino_proj(dino_patches).to(torch.float)

        B = pixel_values.shape[0]
        head, left, right = torch.split(visual_embed_all, B, dim=0)
        visual_embed = torch.cat([head, left, right], dim=1)  # [B, 3*256, 4096]

        latent_tokens = slow_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches:]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)
        flow_loss_unreduced = self.action_decoder(latent_action_tokens, visual_embed, batch['proprio'], actions=batch['actions'].to(torch.float))
        loss = flow_loss_unreduced.mean()

        return loss, latent_action_tokens


@dataclass
class FinetuneConfig:
    # Directory Paths
    data_root_dir: Path = Path("RoboTwin/data/place_phone_stand/demo_clean_test/data")     # Path to Open-X dataset directory
    instruction_dir: Path = Path("RoboTwin/data/place_phone_stand/demo_clean_test/instructions")

    vla_path: str = "MFLAM/models/triple_egodexall_20000"            # Path to your local VLA path
    lam_path: str = "MFLAM/logs_egodex/checkpoints/lam2-egodex/epoch=1-step=20000.ckpt"
    dataset_name: str = "robotwin"                                    # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("MFLAM/robotwin_finetune_runs_flow_place_phone_stand_b32_enhance_pos_freeze")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("MFLAM/robotwin_finetune_runs_flow_place_phone_stand_b32_enhance_pos_freeze/adapter")                     # Temporary directory for LoRA weights before fusing
    # Fine-tuning Parameters
    batch_size: int = 8                                             # Fine-tuning batch size
    max_steps: int = 25010                                          # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 3.5e-4                                   # Fine-tuning learning rate
    grad_accumulation_steps: int = 2                                # Gradient accumulation steps
    image_aug: bool = True                                         # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_00                               # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    n_latents: int = 4                                              # Number of latent tokens for action decoder
    #  LAM setting
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_num_latents: int = 32
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    window_size: int = 15

    dim_actions: int = 14
    dim_proprio: int = 14


    freeze_vla: bool = True
    # LoRA Arguments
    use_lora: bool = False                                          # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # hdf5 data config
    camera_names: str = "camera_high"                               #no use

    # Tracking Parameters
    wandb_project: str = "fientune-robotwin-flow"                   # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()

    if distributed_state.is_main_process:
        print("This is the main process (rank 0).")
    else:
        print(f"This is a worker process (rank {distributed_state.process_index}).")

    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    exp_id += f'=w-FlowDecoder-ws-{cfg.window_size}-place_phone_stand_pos_freeze'

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    wrapped_model = Wrapped_Model(vla=vla, freeze_vla=cfg.freeze_vla, window_size=cfg.window_size, 
                                  action_dim=cfg.dim_actions, 
                                  proprio_dim=cfg.dim_proprio, n_latents=cfg.n_latents).to(device_id)

    trainable_total_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in wrapped_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.max_steps * 0.6), gamma=0.1)

    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel_Hybrid

    latent_action_model = ControllableDINOLatentActionModel_Hybrid(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.,
    )

    lam_ckpt = torch.load(cfg.lam_path)['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()

    dataloader, stats = load_data_robotwin(
        cfg.data_root_dir, cfg.instruction_dir, [cfg.camera_names], cfg.batch_size, action_tokenizer, processor, window_size=cfg.window_size, min_window_size=cfg.window_size,
        max_window_size=cfg.window_size, image_transform=processor.image_processor.apply_transform)
    # save stats and key information
    # In multi-process training, only let the main process write the file to
    # avoid races and duplicate writes. Use an atomic write (tmp file + os.replace).
    stats_dir = os.path.join(cfg.data_root_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    if distributed_state.is_main_process:
        print(f'Saving stats into {stats_dir} (main process)...')
        stats_path = os.path.join(stats_dir, 'dataset_stats.pkl')
        tmp_path = stats_path + '.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(stats, f)
        os.replace(tmp_path, stats_path)

    wrapped_model, latent_action_model, optimizer, scheduler, dataloader = accelerator.prepare(
        wrapped_model, latent_action_model, optimizer, scheduler, dataloader
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        wrapped_model.train()
        optimizer.zero_grad()
        current_step = 0
        while current_step < cfg.max_steps:
            for batch_idx, batch in enumerate(dataloader):
                batch["initial_pixel_values"] = batch["initial_pixel_values"].to(device_id)
                batch["initial_pixel_values_left"] = batch["initial_pixel_values_left"].to(device_id)
                batch["initial_pixel_values_right"] = batch["initial_pixel_values_right"].to(device_id)
                batch["target_pixel_values"] = batch["target_pixel_values"].to(device_id)
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
                batch['actions'] = batch['actions'].to(device_id)
                batch['proprio'] = batch['proprio'].to(device_id)


                with torch.no_grad():
                        video = torch.stack([batch["initial_pixel_values"], batch["target_pixel_values"]], dim=1)
                        latent_action_idx_batch = latent_action_model.module.vq_encode(video)['indices'].squeeze()

                input_ids_list = []
                labels_list = []
                for idx, latent_action_idx in enumerate(latent_action_idx_batch):
                    action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]
                    action_tokens = ''.join(action_vocab)

                    prompt_builder = PurePromptBuilder("openvla")
                    conversation = [
                            {"from": "human", "value": f"What action should the robot take to {batch['instructions'][idx].lower()}?"},
                            {"from": "gpt", "value": action_tokens},
                    ]
                    for turn in conversation:
                        prompt_builder.add_turn(turn["from"], turn["value"])

                    input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
                    labels = list(input_ids)

                    input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

                    labels[: -(len(action_vocab) + 1)] = -100

                    input_ids_list.append(input_ids)
                    labels_list.append(labels)

                input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
                labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

                input_ids, labels = input_ids[:, : processor.tokenizer.model_max_length], labels[:, : processor.tokenizer.model_max_length]

                attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)

                batch["input_ids"] = input_ids
                batch["attention_mask"] = attention_mask
                batch["labels"] = labels

                output, act_loss, latent_action_tokens = wrapped_model(batch)

                loss = act_loss if cfg.freeze_vla else act_loss * 10 + output.loss
                normalized_loss = loss / cfg.grad_accumulation_steps

                torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=0.3)
                normalized_loss.backward()

                action_logits = output.logits[:, wrapped_model.module.vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > 32000

                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())

                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                if current_step + gradient_step_idx >= cfg.max_steps:
                    break

                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)

                if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                    wandb.log(
                        {
                            "train_loss": smoothened_loss,
                            "action_accuracy": smoothened_action_accuracy,
                            "flow_loss": act_loss.item(),
                            "lr": optimizer.state_dict()['param_groups'][0]['lr']
                        },
                        step=gradient_step_idx + current_step,
                    )

                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    progress.update()

                if (gradient_step_idx + current_step) > 0 and (gradient_step_idx + current_step) % cfg.save_steps == 0:
                    print(f"This is a process (rank {distributed_state.process_index}).")
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {gradient_step_idx + current_step}")

                        save_dir = adapter_dir if cfg.use_lora else run_dir
                        save_dir = str(save_dir) + "/{}".format(gradient_step_idx + current_step)

                        if not cfg.freeze_vla:
                            processor.save_pretrained(str(run_dir) + "/{}".format(gradient_step_idx + current_step))
                            wrapped_model.module.vla.save_pretrained(save_dir)

                        dir_path = str(run_dir) + "/{}".format(gradient_step_idx + current_step)
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        torch.save(wrapped_model.module.action_decoder.state_dict(), str(run_dir) + "/{}".format(gradient_step_idx + current_step) + f'/action_decoder-{gradient_step_idx + current_step}.pt')
                        torch.save(wrapped_model.module.dino_proj.state_dict(), str(run_dir) + f'/dino_proj-{gradient_step_idx+current_step}.pt')
                    dist.barrier()

                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, str(adapter_dir) + "/{}".format(gradient_step_idx + current_step))
                        merged_vla = merged_vla.merge_and_unload()
                        if distributed_state.is_main_process:
                            if cfg.save_latest_checkpoint_only:
                                merged_vla.save_pretrained(str(run_dir) + "/{}".format(gradient_step_idx + current_step))
                                print(f"Saved Model Checkpoint for Step {gradient_step_idx + current_step} at: {run_dir}/{gradient_step_idx + current_step}")
                            else:
                                checkpoint_dir = Path(str(run_dir) + "/{}".format(gradient_step_idx + current_step) + f"--{gradient_step_idx + current_step}_chkpt")
                                os.makedirs(checkpoint_dir, exist_ok=True)

                                processor.save_pretrained(checkpoint_dir)
                                merged_vla.save_pretrained(checkpoint_dir)

                                print(f"Saved Model Checkpoint for Step {gradient_step_idx + current_step} at: {checkpoint_dir}")

                    dist.barrier()
            current_step = gradient_step_idx + current_step
            description = f"Epoch {current_step//len(dataloader)}, Step {current_step} | action_loss: {act_loss.item():.4f} | acc: {smoothened_action_accuracy:.4f}"
            progress.set_description(description)

            if current_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                wandb.finish()
                break


if __name__ == "__main__":
    finetune()