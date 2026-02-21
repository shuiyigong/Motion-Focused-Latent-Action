# 🚀 Motion-Focused VLA

## 📚 Table of Contents

- [1. Environment Setup & Data Preparation](#1-environment-setup-and-data-preparatin)
- [2. Train Hybrid Disentangled VQ-VAE](#2-train-hybrid-disentangled-vq-vae)
- [3. Pretrain VLM](#3-pretrain-vlm)
- [4. Downstream Fine-tuning](#4-downstream-fine-tuning)

---

## 🛠️ 1. Environment Setup and Data Preparatin


### 1.1 Environment Setup

```bash
conda create -n mfvla python=3.10 -y
conda activate mfvla
```

Install PyTorch (choose the correct command for your CUDA version from the official site; example below):

```bash
pip install torch torchvision
```

Install project dependencies:

```bash
git clone git@github.com:OpenDriveLab/UniVLA.git
cd MFVLA
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```
### 1.2 Data Preparatin

Please refer to [here](https://github.com/moojink/rlds_dataset_mod/blob/ad83e6c0efad5823540c0f6d3a05529596ead0b5/prepare_open_x.sh) to download Bridge datasets from OXE.

Please refer to [here](https://github.com/apple/ml-egodex) to download Egodex Dataset.

The data preprocessing script will be released upon finalization.

---


## 🧠 2. Train Hybrid Disentangled VQ-VAE

### 2.1 Configure

Go to config directory:

```bash
cd latent_action_model/config
```

Select and edit config files according to your experiment setup.

### 2.2 Launch Training

From project root:

```bash
bash latent_action_model/train.sh
```

Checkpoints are saved to:

- `./logs/checkpoints`

---

## 🧪 3. Pretrain VLM

### 3.1 Download Prismatic Backbone

Download the backbone from [here](https://huggingface.co/TRI-ML/prismatic-vlms/tree/main/prism-dinosiglip-224px%2B7b) and place the downloaded files under `MFVLA/models`

### 3.2 Update Training Configurations

Adjust local paths and settings in:

- `MFVLA/vla-scripts/train.py` (`TrainConfig`, e.g., model paths)
- `MFVLA/vla-scripts/train.sh` (`vla.type`)

Available `vla.type` options:

- `prism-dinosiglip-224px+mx-bridge` (train on Bridge dataset)
- `prism-dinosiglip-224px+mx-egodex` (train on EgoDex dataset)

### 3.3 Run pretraining:

```bash
bash vla-scripts/train.sh
```

---

## 🎯 4. Downstream Fine-tuning

Please first download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds/tree/main) for finetuning.

Adjust local paths and settings in:

- `MFVLA/vla-scripts/finetune_libero.py` (`FinetuneConfig`, e.g., model paths, data_dir, task and save_dir)
You can choose ```dataset_name``` from ```libero_spatial_no_noops```, ```libero_object_no_noops```, ```libero_goal_no_noops```, and ```libero_10_no_noops```

```bash
bash vla-scripts/train_finetune_libero.sh
```

Evaluate your model with

```bash
pip install -r experiments/robot/libero/libero_requirements.txt
python experiments/robot/libero/run_libero_eval.py \
    --task_suite_name libero_10 \   # Choose from [libero_spatial, libero_object, libero_goal, libero_10] 
    --action_decoder_path /path/to/your/action_decoder_path.pt \
    --pretrained_checkpoint /path/to/your/libero_10_finetuned_univla \
    --save_video False    # Whether to save rollout videos \
    --num_trials_per_task 20 \
    --seed 7
```

---

## 🙏 Acknowledgment

Special thanks to two outstanding open-source projects: **UniVLA** and **OpenVLA**.  
This project is developed based on these two excellent works.






