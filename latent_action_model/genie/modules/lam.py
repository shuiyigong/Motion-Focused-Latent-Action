from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from latent_action_model.genie.modules.blocks import  SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer
                                                    
from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ControllableDINOLatentActionModel_Hybrid(nn.Module):
    """
    Hybrid VQ-VAE for Disentangled Action Learning.
    Input: Frame 1 & Frame 2 (Deep + Shallow DINO features)
    Latent: z_action (Quantized) + z_noise (Gaussian)
    Output: Predicted Frame 2 features (Deep + Shallow)
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            using_diff_features: bool = False,
            using_resnet: bool = False,
            using_action: bool = False
    ) -> None:
        super(ControllableDINOLatentActionModel_Hybrid, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        
        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.dino_encoder.requires_grad_(False)
        dino_dim = 768
        self.num_codes = 4
        
        self.using_diff_features = using_diff_features
        self.using_resnet = using_resnet
        self.using_action = using_action
        
        return_attention = True
        self.return_attention = return_attention


        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
            return_attention=self.return_attention
        )

        self.to_actioncodebook = nn.Linear(model_dim, latent_dim)
        self.vq_action = VectorQuantizer(
            num_latents=16,
            latent_dim=latent_dim,
            code_restart=True,
        )
        self.to_bgcodebook = nn.Linear(model_dim, latent_dim)
        self.vq_bg = VectorQuantizer(
            num_latents=16,
            latent_dim=latent_dim,
            code_restart=True,
        )

        

        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.bg_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,      
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
            return_attention=self.return_attention,
        )

        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim)) #*2
        self.bg_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim)) #*2
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        nn.init.uniform_(self.bg_latent, a=-1, b=1)


    def extract_dino_features(self, videos,B,T):
        """
        Input: (B, T, 3, H, W)
        Output: Deep features, Shallow features. Both (B, T, N_patches, 768)
        """
        intermediates = self.dino_encoder.get_intermediate_layers(videos, n=4, reshape=True,)

        feat_deep = intermediates[-1]   # (B*T, H/14, W/14, 768)
        feat_deep = rearrange(feat_deep, "b d h w -> b (h w) d")

        feat_deep = F.layer_norm(feat_deep, (feat_deep.shape[-1],))
        
        # Flatten patches: (B*T, N, D)
        feat_deep = rearrange(feat_deep, "(b T) l d -> b T l d", b=B, T=T)
        

        
        return feat_deep



    def vq_encode(self, videos: Tensor,batch=None, text_sent_embed: Tensor = None , lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:

        B, T = videos.shape[:2]
        flat_videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(flat_videos)
        feat_deep = self.extract_dino_features(videos, B, T)
        if batch is not None:
            masks = batch["masks"]  # B, T, 2, H, W
            masks = (masks > 0.0).float()
            masks = masks * 5.0
            flat_masks = rearrange(masks, "b T c h w -> (b T) c h w") # (B*T, 2, 224, 224)       
            downsampled_masks = F.interpolate(flat_masks.float(), size=(16, 16), mode='area')
            soft_mask = rearrange(downsampled_masks, "(b T) c h w -> b T (h w) c", b=B)
            background_weight = 0.3
            attention_map = soft_mask + background_weight
            attention_map = attention_map / (5+ background_weight)  # Normalize to [0, 1]
        else:
            attention_map = None
        
        dino_features = feat_deep #feat_deep
        combined_features = dino_features

        action_pad_action = self.action_latent.expand(B, T, -1, -1)
        action_pad_bg = self.bg_latent.expand(B,T,-1,-1)
        padded_patches = torch.cat([action_pad_bg, combined_features], dim=2)#B,2,4+256,1536
        padded_patches = torch.cat([action_pad_action, padded_patches], dim=2)#B,2,4+256,1536

        # Encode
        z = self.encoder(padded_patches)#B,2,264,latent_dim(128)

        z_bg = self.to_bgcodebook(z[:, 1:, self.num_codes : self.num_codes * 2])
        z_bg = z_bg.reshape(B * (T - 1), self.num_codes, self.latent_dim)#B*(T-1),4,128
        z_q_bg, z_bg, emb_bg, indices_bg = self.vq_bg(z_bg)
        z_q_bg = z_q_bg.reshape(B, T - 1, self.num_codes, self.latent_dim)

        z_action = self.to_actioncodebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)
        z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)

        return {
            "patches": combined_features, #Now contains both DINO and ResNet features 64,2,256,1536
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "z_q_bg": z_q_bg,
            "z_bg": z_bg,
            "emb_bg": emb_bg,
            "indices": indices,
            "indices_bg": indices_bg,
            "attention_map": attention_map
        }
        

    def forward(self, batch: Dict) -> Dict:
        B, T = batch["videos"].shape[:2] #64 2
        H, W = batch["videos"].shape[3:5]#224 224
        outputs = self.vq_encode(batch["videos"],batch)
        video_patches = self.patch_up(outputs["patches"][:, :-1]) #64,1,256,768
        lat_action = self.action_up(outputs["z_q"])
        lat_bg = self.bg_up(outputs["z_q_bg"])
        
        

        input_action_only = torch.cat([
            lat_action, 
            torch.zeros_like(lat_bg), 
            video_patches
        ], dim=2) # (B, 1, 4+4+256, model_dim)
        recon_seq_action = self.decoder(input_action_only, return_attention=False)
        recon_img_action = recon_seq_action[:, :, -video_patches.shape[2]:] 

        input_bg_only = torch.cat([
            torch.zeros_like(lat_action), 
            lat_bg, 
            video_patches
        ], dim=2)
        recon_seq_bg = self.decoder(input_bg_only, return_attention=False)
        recon_img_bg = recon_seq_bg[:, :, -video_patches.shape[2]:] 

        input_full = torch.cat([
            lat_action, 
            lat_bg,      
            video_patches
        ], dim=2)
        if self.return_attention:
            video_recon, attn_weights,all_block_weights = self.decoder(input_full, return_attention=True) #64,1,264,768
        else:
            video_recon = self.decoder(input_full)
        recon_img_full = video_recon[:, :, -video_patches.shape[2]:]

        
        dim_half = 768  
        outputs.update({
            "pred_bg_deep": recon_img_bg[..., :dim_half],
            "pred_action_deep": recon_img_action[..., :dim_half],
            "pred_full_deep": recon_img_full[..., :dim_half],
            "target_deep": outputs["patches"][:, [-1], :, :dim_half],
            "target_shallow": outputs["patches"][:, [-1], :, dim_half:],
        })

        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
