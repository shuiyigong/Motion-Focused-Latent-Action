CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline torchrun --standalone --nnodes 1 --nproc-per-node 2 MFVLA/latent_action_model/main.py fit \
    --config MFVLA/latent_action_model/config/lam-stage-2-mix.yaml \
    2>&1 | tee MFVLA/latent_action_model/lam-stage-2-test.log