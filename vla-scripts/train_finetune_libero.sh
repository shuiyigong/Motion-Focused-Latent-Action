CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 /MFVLA/vla-scripts/finetune_libero.py
