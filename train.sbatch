#!/bin/bash
#SBATCH --job-name=train_64_id_datalol
#SBATCH --error=runs/train_64_id_datalol_errs-%j.err
#SBATCH --output=runs/train_64_id_datalol_logs-%j.log
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --constraint="[type_a|type_b|type_c|type_d|type_e]"

module load Python/Anaconda_v02.2024
module load CUDA/12.2
module load gnu10/10.1

source deactivate
source activate photomaker_matos

WANDB_API_KEY=034de88c658e0d6f03ee9fd9a618344c075198d9 HYDRA_FULL_ERROR=1 NO_ALBUMENTATIONS_UPDATE=1 accelerate launch --config_file=/home/dnbobkov/matos/persongen/src/configs/accelerate.yaml train.py