#!/bin/bash
#SBATCH --job-name=3000_iters_no_masked_loss
#SBATCH --error=runs/3000_iters_no_masked_loss_errs-%j.err
#SBATCH --output=runs/3000_iters_no_masked_loss_logs-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --constraint="[type_a|type_b|type_c]"

module load Python/Anaconda_v02.2024
module load CUDA/12.2
module load gnu10/10.1

source deactivate
source activate photomaker
export WANDB_API_KEY=034de88c658e0d6f03ee9fd9a618344c075198d9

HYDRA_FULL_ERROR=1 python train.py

TORCH_DISTRIBUTED_DEBUG=INFO NO_ALBUMENTATIONS_UPDATE=1 WANDB_API_KEY=034de88c658e0d6f03ee9fd9a618344c075198d9 HYDRA_FULL_ERROR=1 accelerate launch --config_file=src/configs/ddp/accelerate.yaml train_ip.py model.pretrained_model_name_or_path=hotshotco/SDXL-512 writer.run_name=test trainer.epoch_len=1000 dataloader.train.batch_size=2 dataloader.train.num_workers=2 dataloader.train.pin_memory=True validation_args.num_images_per_prompt=5 writer=empty

NO_ALBUMENTATIONS_UPDATE=1 WANDB_API_KEY=034de88c658e0d6f03ee9fd9a618344c075198d9 HYDRA_FULL_ERROR=1 accelerate launch --config_file=src/configs/ddp/accelerate.yaml train.py model.rank=1 trainer.resume_from=checkpoint-epoch12.pth writer.run_name=enc_head_lora1_512 trainer.epoch_len=1000 dataloader.train.batch_size=12 dataloader.train.num_workers=12 dataloader.train.pin_memory=True validation_args.num_images_per_prompt=5

NO_ALBUMENTATIONS_UPDATE=1 WANDB_API_KEY=034de88c658e0d6f03ee9fd9a618344c075198d9 HYDRA_FULL_ERROR=1 python3 -u inference.py inferencer.save_path="pipi" inferencer.from_pretrained="saved/enc_head_lora1_512/checkpoint-epoch10.pth" model.rank=1 validation_args.num_images_per_prompt=5 writer.run_name=pipi model.pretrained_model_name_or_path=hotshotco/SDXL-512 writer=empty

NO_ALBUMENTATIONS_UPDATE=1 WANDB_API_KEY=034de88c658e0d6f03ee9fd9a618344c075198d9 HYDRA_FULL_ERROR=1 accelerate launch --config_file=src/configs/ddp/accelerate.yaml train.py model.rank=64 trainer.from_pretrained=photomaker-v1.bin writer.run_name=test trainer.epoch_len=1000 dataloader.train.batch_size=2 dataloader.train.num_workers=2 dataloader.train.pin_memory=True validation_args.num_images_per_prompt=5 writer=empty
