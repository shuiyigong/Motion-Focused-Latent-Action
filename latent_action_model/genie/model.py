from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import wandb
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from accelerate import PartialState
import torch.nn.functional as F
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


OptimizerCallable = Callable[[Iterable], Optimizer]

from genie.modules.lam import ControllableDINOLatentActionModel_Hybrid
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def visualize_codebook_distribution(vq_module, title="Codebook Analysis", save_path=None):
                codebook_weights = vq_module.codebook.weight.detach().cpu()

                if hasattr(vq_module, "usage"):
                    usage = vq_module.usage.detach().cpu().numpy()
                    dead_codes_mask = usage < 1
                else:
                    usage = None
                    dead_codes_mask = None
                fig = plt.figure(figsize=(20, 12))
                plt.suptitle(title, fontsize=20, y=0.98)
                ax1 = fig.add_subplot(2, 3, 1)
                norms = torch.norm(codebook_weights, p=2, dim=1).numpy()
                
                sns.histplot(norms, bins=30, kde=True, ax=ax1, color='skyblue')
                ax1.set_title(f"Vector Norm Distribution\nMean: {np.mean(norms):.2f}, Std: {np.std(norms):.2f}")
                ax1.set_xlabel("L2 Norm Magnitude")
                ax1.set_ylabel("Count")
                if dead_codes_mask is not None and dead_codes_mask.any():
                    dead_norms = norms[dead_codes_mask]
                    if len(dead_norms) > 0:
                        sns.histplot(dead_norms, bins=30, ax=ax1, color='red', alpha=0.5, label='Dead Codes')
                        ax1.legend()
                ax2 = fig.add_subplot(2, 3, 2)
                pca = PCA(n_components=2)
                weights_pca = pca.fit_transform(codebook_weights.numpy())
                

                if usage is not None:
                    scatter = ax2.scatter(weights_pca[:, 0], weights_pca[:, 1], 
                                        c=np.log1p(usage), cmap='viridis', alpha=0.7, edgecolors='k', s=50)
                    if dead_codes_mask.any():
                        ax2.scatter(weights_pca[dead_codes_mask, 0], weights_pca[dead_codes_mask, 1], 
                                    c='red', marker='x', s=100, label='Dead Codes')
                        ax2.legend()
                    plt.colorbar(scatter, ax=ax2, label='Log Usage')
                else:
                    ax2.scatter(weights_pca[:, 0], weights_pca[:, 1], alpha=0.6)
                    
                ax2.set_title(f"PCA Projection (Explained Var: {np.sum(pca.explained_variance_ratio_):.2f})")
                ax2.set_xlabel("PC 1")
                ax2.set_ylabel("PC 2")
                ax3 = fig.add_subplot(2, 3, 3)
                normalized_weights = F.normalize(codebook_weights, p=2, dim=1)
                cosine_sim = torch.matmul(normalized_weights, normalized_weights.T).numpy()
                limit = min(64, codebook_weights.shape[0])
                sns.heatmap(cosine_sim[:limit, :limit], cmap='coolwarm', vmin=-1, vmax=1, ax=ax3, square=True)
                ax3.set_title(f"Cosine Similarity (First {limit} codes)")
                ax4 = fig.add_subplot(2, 3, 4)
                with torch.no_grad():
                    dist_matrix = torch.cdist(codebook_weights, codebook_weights, p=2)
                dist_np = dist_matrix.cpu().numpy()
                upper_tri = dist_np[np.triu_indices(dist_np.shape[0], k=1)]

                sns.histplot(
                    upper_tri,
                    bins=50,
                    kde=True,
                    ax=ax4,
                    color="purple"
                )

                ax4.set_title(
                    f"Pairwise Euclidean Distance\n"
                    f"Mean: {upper_tri.mean():.2f}, Std: {upper_tri.std():.2f}"
                )
                ax4.set_xlabel("Euclidean Distance")
                ax4.set_ylabel("Count")
                ax5 = fig.add_subplot(2, 3, 5)
                if usage is not None:
                    sns.histplot(usage, bins=50, ax=ax5, log_scale=(False, True)) 
                    ax5.set_title(f"Code Usage Count (Dead: {np.sum(dead_codes_mask)}/{len(usage)})")
                    ax5.set_xlabel("Usage Count")
                else:
                    ax5.text(0.5, 0.5, "No Usage Stats", ha='center')

                ax6 = fig.add_subplot(2, 3, 6)
                dim_variance = torch.var(codebook_weights, dim=0).numpy()
                sorted_idx = np.argsort(dim_variance)[::-1]
                ax6.plot(dim_variance[sorted_idx])
                ax6.set_title("Variance per Latent Dimension")
                ax6.set_xlabel("Sorted Dimension Index")
                ax6.set_ylabel("Variance")
                ax6.text(0.5, 0.5, f"Dead Dims (<1e-4): {np.sum(dim_variance < 1e-4)}", transform=ax6.transAxes)

                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path)
                    print(f"Codebook visualization saved to {save_path}")
                
                plt.show()
def visualize_cross_codebook_relationship(
                vq_module_a,
                vq_module_b,
                name_a="Codebook A",
                name_b="Codebook B",
                title="Cross Codebook Analysis",
                save_path=None
            ):
                W_a = vq_module_a.codebook.weight.detach().cpu()  # (Na, D)
                W_b = vq_module_b.codebook.weight.detach().cpu()  # (Nb, D)

                Na, D = W_a.shape
                Nb, _ = W_b.shape

                fig = plt.figure(figsize=(20, 6))
                plt.suptitle(title, fontsize=20, y=0.98)
                ax1 = fig.add_subplot(1, 3, 1)

                with torch.no_grad():
                    cross_dist = torch.cdist(W_a, W_b, p=2)  # (Na, Nb)

                cross_np = cross_dist.numpy().reshape(-1)

                sns.histplot(
                    cross_np,
                    bins=80,
                    kde=True,
                    ax=ax1,
                    color="teal"
                )

                ax1.set_title(
                    f"Cross Euclidean Distance\n"
                    f"Mean: {cross_np.mean():.2f}, Std: {cross_np.std():.2f}"
                )
                ax1.set_xlabel(f"||{name_a} - {name_b}||")
                ax1.set_ylabel("Count")
                ax2 = fig.add_subplot(1, 3, 2)

                nn_a2b = cross_dist.min(dim=1).values.numpy()  # A -> B
                nn_b2a = cross_dist.min(dim=0).values.numpy()  # B -> A

                sns.histplot(
                    nn_a2b,
                    bins=50,
                    kde=True,
                    ax=ax2,
                    color="blue",
                    label=f"{name_a} → {name_b}"
                )
                sns.histplot(
                    nn_b2a,
                    bins=50,
                    kde=True,
                    ax=ax2,
                    color="orange",
                    label=f"{name_b} → {name_a}",
                    alpha=0.7
                )

                ax2.set_title("Cross Nearest Neighbor Distance")
                ax2.set_xlabel("Nearest Euclidean Distance")
                ax2.legend()
                ax3 = fig.add_subplot(1, 3, 3)

                W_all = torch.cat([W_a, W_b], dim=0).numpy()
                labels = np.array([0] * Na + [1] * Nb)

                pca = PCA(n_components=2)
                W_pca = pca.fit_transform(W_all)

                ax3.scatter(
                    W_pca[labels == 0, 0],
                    W_pca[labels == 0, 1],
                    label=name_a,
                    alpha=0.7,
                    s=50
                )
                ax3.scatter(
                    W_pca[labels == 1, 0],
                    W_pca[labels == 1, 1],
                    label=name_b,
                    alpha=0.7,
                    s=50
                )

                ax3.set_title(
                    f"Joint PCA (Explained Var: {pca.explained_variance_ratio_.sum():.2f})"
                )
                ax3.set_xlabel("PC 1")
                ax3.set_ylabel("PC 2")
                ax3.legend()

                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path)
                    print(f"Cross codebook visualization saved to {save_path}")

                plt.show()
class MFLAM(LightningModule):
    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 768,
            lam_latent_dim: int = 128,
            lam_num_latents: int = 16,
            lam_patch_size: int = 14,
            lam_enc_blocks: int = 12,
            lam_dec_blocks: int = 12,
            lam_num_heads: int = 12,
            lam_dropout: float = 0.0,
            vq_beta: float = 0.25,
            log_interval: int = 5000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            using_diff_features: bool = False, #
            using_resnet: bool = False,  #
            using_action: bool = False,  #  
            b_size: int = 64, #
    ) -> None:
        super(MFLAM, self).__init__()

        lam = ControllableDINOLatentActionModel_Hybrid

        self.lam = lam(
                    in_dim=image_channels,
                    model_dim=lam_model_dim,
                    latent_dim=lam_latent_dim,
                    num_latents=lam_num_latents,
                    patch_size=lam_patch_size,
                    enc_blocks=lam_enc_blocks,
                    dec_blocks=lam_dec_blocks,
                    num_heads=lam_num_heads,
                    dropout=lam_dropout,
                    using_diff_features=using_diff_features,
                    using_resnet=using_resnet,
                    using_action=using_action,
                )

        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.using_resnet = using_resnet
        self.using_action = using_action    

        self.save_hyperparameters()

        self.task_name = task_name+'+'+str(b_size)
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(name=self.task_name, reinit=True)
       

    def shared_step(self, batch: Dict) -> Tuple: 
        outputs = self.lam(batch)

        pred_bg_deep = outputs.get("pred_bg_deep", None)
        pred_action_deep = outputs.get("pred_action_deep", None)
        pred_full_deep = outputs.get("pred_full_deep", None)
        target_deep = outputs.get("target_deep", None)
        attention_map = outputs["attention_map"][:, [-1]]  

        threshold = 0.5
        fg_mask = (attention_map > threshold).float()  
        count_fg = fg_mask.sum() + 1e-6
        bg_mask = (attention_map <= threshold).float()  
        count_bg = bg_mask.sum() + 1e-6



        def compute_masked_mse(pred, target, mask, count_pixels):
            diff_sq = (pred - target) ** 2
            masked_loss = (diff_sq * mask).sum() 
            return masked_loss / (count_pixels * pred.shape[-1])
        

        def compute_global_mse(pred, target):
            import torch.nn.functional as F
            return F.mse_loss(pred, target)

        loss_act_d = compute_masked_mse(pred_action_deep, target_deep, fg_mask, count_fg)
        loss_bg_d = compute_masked_mse(pred_bg_deep, target_deep, bg_mask, count_bg)
        loss_full_d = compute_global_mse(pred_full_deep, target_deep)


        w_recon_act = 1.0  
        w_recon_full = 1.0   
        w_recon_bg = 1.0   


        loss_recon_action = loss_act_d 
        loss_recon_bg = loss_bg_d 
        loss_recon_full = loss_full_d 

        if "pred_action" in outputs and "target_action" in outputs:
            pred = outputs["pred_action"]
            target = outputs["target_action"]

            pred_trans = pred[..., :3]
            target_trans = target[..., :3]

            pred_rot = pred[..., 3:6]
            target_rot = target[..., 3:6]
            

            pred_grip = pred[..., 6]
            target_grip = target[..., 6]
            
            loss_trans = ((pred_trans - target_trans)**2).mean()
            loss_rot = ((pred_rot - target_rot)**2).mean()
            loss_grip = ((pred_grip - target_grip)**2).mean()
            
            w_trans = 1.0
            w_rot = 1.0
            w_grip = 0.5
            
            action_loss = (w_trans * loss_trans) + (w_rot * loss_rot) + (w_grip * loss_grip)

            mse_loss = mse_loss + action_loss

        q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean() 
        commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

        q_bg_loss = ((outputs["emb_bg"].detach() - outputs["z_bg"]) ** 2).mean() 
        commit_bg_loss = ((outputs["emb_bg"] - outputs["z_bg"].detach()) ** 2).mean()

        loss = w_recon_act * loss_recon_action + w_recon_full * loss_recon_full + w_recon_bg * loss_recon_bg + q_loss + self.vq_beta * commit_loss + q_bg_loss + self.vq_beta * commit_bg_loss



        # Compute code usage
        unique, counts = torch.unique(outputs["indices"], return_counts=True)
        index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
        index_counts[unique] = counts
        code_usage = (index_counts != 0).float().mean()

        unique, counts = torch.unique(outputs["indices_bg"], return_counts=True)
        index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
        index_counts[unique] = counts
        code_usage_bg = (index_counts != 0).float().mean()

        loss_logs = (
            ("loss", loss),
            ("loss_bg_deep", loss_bg_d),
            ("loss_act_deep", loss_act_d),
            ("loss_full_deep", loss_full_d),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("code_usage", code_usage),
            ("q_bg_loss", q_bg_loss),
            ("commit_bg_loss", commit_bg_loss),
            ("code_usage_bg", code_usage_bg),
        )
        return outputs, loss, loss_logs



    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        outputs, loss, aux_losses = self.shared_step(batch)
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        return loss

    def on_after_backward(self):
        if self.global_rank == 0:
            unused_params = []
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is None:
                    unused_params.append(name)
            
            if unused_params:
                print("\n" + "="*30)
                print("warning: the following parameters have requires_grad=True but did not receive gradients:")
                for name in unused_params:
                    print(f" -> {name}")
                print("="*30 + "\n")


    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)
        # Log the test loss
        print(aux_losses)
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch["videos"])
        )
        if self.distributed_state.is_main_process:
            wandb.log({**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}})

        return loss

    def on_train_epoch_end(self):
        self.lam.vq_bg.random_restart()
        self.lam.vq_bg.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")


    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    
    def configure_optimizers(self) -> Optimizer:
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optim = self.optimizer(trainable_params)
        return optim
