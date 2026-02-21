#export LD_LIBRARY_PATH=/home/pai/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
GPUS_PER_NODE=4
NNODES=1
MASTER_PORT=${MASTER_PORT:-28598}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=0


# Run your training script with torchrun
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} /mnt/public/user/MFVLA/vla-scripts/train.py \
                                 --vla.type prism-dinosiglip-224px+mx-bridge \
                                 --run_root_dir "/mnt/public/user/MFVLA/vla_pretraining_log" \

