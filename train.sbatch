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