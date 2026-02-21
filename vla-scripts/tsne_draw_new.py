
import os
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import draccus

# HF / Prismatic Imports
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransformLatentAction
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

@dataclass
class AnalysisConfig:
    vla_path: str = "/mnt/public/user/MFVLA/models/triplepath_b+f"
    lam_path: str = "/mnt/public/user/MFVLA/logs/checkpoints/epoch=3-step=10000.ckpt"
    
    data_root1: Path = Path("/OXEDataset")
    dataset_mix1: str = "bridge"
    data_root2: Path = Path("/OXEDataset")
    dataset_mix2: str = "furniture"
    
    save_prefix: str = "vla_analysis_results"
    batch_size: int = 32
    max_samples: int = 20000
    device: str = "cuda:0"
    
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    window_size: int = 12
    shuffle_buffer_size: int = 16000


def compute_linear_cka(X, Y):
    def centering(K):
        n = K.shape[0]
        unit = np.ones([n, n]) / n
        return K - unit @ K - K @ unit + unit @ K @ unit

    K = X @ X.T
    L = Y @ Y.T
    Kc = centering(K)
    Lc = centering(L)
    hsic = np.trace(Kc @ Lc)
    norm_x = np.linalg.norm(Kc)
    norm_y = np.linalg.norm(Lc)
    return hsic / (norm_x * norm_y + 1e-8)

def remove_domain_subspace_iterative(X, y, acc_threshold=0.65, max_iters=5, dims_per_iter=10):
    X_scaled = X.astype(np.float32)
    scaler = StandardScaler(with_std=False)
    X_scaled = scaler.fit_transform(X_scaled)

    for it in range(max_iters):
        clf = LogisticRegression(max_iter=200, C=0.1)
        clf.fit(X_scaled, y)
        acc = clf.score(X_scaled, y)
        if acc <= acc_threshold: break
        
        pca = PCA(n_components=dims_per_iter)
        pca.fit(X_scaled)
        V = pca.components_
        projection = (X_scaled @ V.T) @ V
        X_scaled = X_scaled - projection
    return X_scaled


def plot_advanced_cka_analysis(embeds1, tokens1, embeds2, tokens2, save_prefix):
    t1_counts = Counter(tokens1)
    t2_counts = Counter(tokens2)
    common_tokens = sorted([t for t in t1_counts if t1_counts[t] >= 50 and t2_counts.get(t, 0) >= 50 and 
                            min(t1_counts[t], t2_counts.get(t, 0))/max(t1_counts[t], t2_counts.get(t, 0)) >= 0.2])
    
    if not common_tokens:
        print("Error: No common tokens found with enough samples.")
        return
    cents1 = np.array([embeds1[tokens1 == t].mean(axis=0) for t in common_tokens])
    cents2 = np.array([embeds2[tokens2 == t].mean(axis=0) for t in common_tokens])
    plt.figure(figsize=(10, 8))
    cross_sim = cosine_similarity(cents1, cents2)
    sns.heatmap(cross_sim, xticklabels=common_tokens, yticklabels=common_tokens, 
                annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Cosine Similarity'})
    plt.title(f"Cross-Dataset Action Alignment\n(Bridge vs Furniture)")
    plt.xlabel("Furniture Tokens")
    plt.ylabel("Bridge Tokens")
    plt.savefig(f"{save_prefix}_cross_alignment.png")
    plt.close()

    struct1 = cosine_similarity(cents1, cents1)
    struct2 = cosine_similarity(cents2, cents2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(struct1, xticklabels=common_tokens, yticklabels=common_tokens, ax=ax1, cmap="rocket")
    ax1.set_title("Internal Structure: Bridge")
    sns.heatmap(struct2, xticklabels=common_tokens, yticklabels=common_tokens, ax=ax2, cmap="rocket")
    ax2.set_title("Internal Structure: Furniture")

    tri_idx = np.triu_indices(len(common_tokens), k=1)
    struct_corr = np.corrcoef(struct1[tri_idx], struct2[tri_idx])[0, 1]
    plt.suptitle(f"Representational Similarity Analysis (RSA)\nStructural Correlation: {struct_corr:.4f}")
    plt.savefig(f"{save_prefix}_rsa_comparison.png")
    plt.close()

    cka_scores = []
    for _ in range(50):
        c1_boot, c2_boot = [], []
        for t in common_tokens:
            idx1 = np.where(tokens1 == t)[0]
            idx2 = np.where(tokens2 == t)[0]
            s1 = embeds1[np.random.choice(idx1, size=len(idx1)//2)]
            s2 = embeds2[np.random.choice(idx2, size=len(idx2)//2)]
            c1_boot.append(s1.mean(axis=0))
            c2_boot.append(s2.mean(axis=0))
        cka_scores.append(compute_linear_cka(np.array(c1_boot), np.array(c2_boot)))

    plt.figure(figsize=(8, 5))
    sns.histplot(cka_scores, kde=True, color="skyblue")
    plt.axvline(np.mean(cka_scores), color='red', linestyle='--')
    plt.title(f"CKA Stability of Motion-Focused Latent Action\nMean: {np.mean(cka_scores):.4f} ± {np.std(cka_scores):.4f}")
    plt.xlabel("CKA Score")
    plt.savefig(
    f"{save_prefix}_cka_dist.pdf",
    format="pdf",
    bbox_inches="tight"
    )
    plt.close()
    
    print(f"All visualizations saved with prefix: {save_prefix}")


def extract_embeddings(vla, loader, device, max_samples):
    all_embeds, all_tokens = [], []
    vla.eval()
    count = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Extracting"):
            if count >= max_samples: break
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device)
            labels = batch["labels"].to(device)

            outputs = vla(input_ids=input_ids, pixel_values=pixel_values, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            
            vision_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
            action_hidden = last_hidden[:, vision_patches:]
            mask = labels > 32000
            
            for i in range(labels.shape[0]):
                m = mask[i]
                if m.any():
                    all_embeds.append(action_hidden[i][m].cpu().float().numpy())
                    all_tokens.append(labels[i][m].cpu().numpy())
                    count += m.sum().item()
    return np.concatenate(all_embeds), np.concatenate(all_tokens)


@draccus.wrap()
def main(cfg: AnalysisConfig):
    print("Loading Models...")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(cfg.device)
    processor = PrismaticProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel_Hybrid
    lam = ControllableDINOLatentActionModel_Hybrid(
        in_dim=3, model_dim=cfg.lam_model_dim, latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size, patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12
    )
    lam_ckpt = torch.load(cfg.lam_path, map_location='cpu')['state_dict']
    lam.load_state_dict({k.replace("lam.", ""): v for k, v in lam_ckpt.items()})
    lam = lam.to(cfg.device).eval()

    def get_loader(root, mix):
        transform = RLDSBatchTransformLatentAction(
            action_tokenizer=lam, base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            image_transform_lam=transforms.ToTensor(), prompt_builder_fn=PurePromptBuilder
        )
        dataset = RLDSDataset(root, mix, transform, resize_resolution=tuple(vla.config.image_sizes))
        collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id)
        return DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collator)

    loader1 = get_loader(cfg.data_root1, cfg.dataset_mix1)
    loader2 = get_loader(cfg.data_root2, cfg.dataset_mix2)

    embeds1, tokens1 = extract_embeddings(vla, loader1, cfg.device, cfg.max_samples)
    embeds2, tokens2 = extract_embeddings(vla, loader2, cfg.device, cfg.max_samples)
    all_embeds = np.concatenate([embeds1, embeds2])
    labels = np.array([0]*len(embeds1) + [1]*len(embeds2))
    all_embeds_clean = remove_domain_subspace_iterative(all_embeds, labels)
    
    embeds1_c = all_embeds_clean[:len(embeds1)]
    embeds2_c = all_embeds_clean[len(embeds1):]

    plot_advanced_cka_analysis(embeds1_c, tokens1, embeds2_c, tokens2, cfg.save_prefix)

if __name__ == "__main__":
    main()
