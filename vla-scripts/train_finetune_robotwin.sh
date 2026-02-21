CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes 1 --nproc-per-node 2 MFVLA/vla-scripts/finetune_robotwin_flow.py
