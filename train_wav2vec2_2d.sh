#!/bin/bash
#SBATCH --job-name=bw2d
#SBATCH --partition=gpu
# #SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --time=10-00:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=5
#SBATCH --output=bw2d_output.log
#SBATCH --error=bw2d_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --save_path exps/exp2 --feat_type wav2vec2 --feat_dim 768 --n_cpu 5 --is_2d

conda deactivate