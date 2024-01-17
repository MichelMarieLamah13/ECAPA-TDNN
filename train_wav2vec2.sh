#!/bin/bash
#SBATCH --job-name=train_wav2vec2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=10-00:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=10
#SBATCH --output=train_wav2vec2_output.log
#SBATCH --error=train_wav2vec2_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --save_path exps/exp2 --feat_type wav2vec2 --feat_dim 768 --n_cpu 10

conda deactivate