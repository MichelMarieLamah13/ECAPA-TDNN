#!/bin/bash
#SBATCH --job-name=train_ecapa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
# #SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
# #SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --time=10-00:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=8
#SBATCH --output=train_output.log
#SBATCH --error=train_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 8

conda deactivate