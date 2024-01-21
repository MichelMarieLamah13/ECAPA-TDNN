#!/bin/bash
#SBATCH --job-name=bw
#SBATCH --partition=gpu
# #SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --time=10-00:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=5
#SBATCH --output=bw_output.log
#SBATCH --error=bw_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --save_path exps/exp2 --feat_type wav2vec2 --feat_dim 768 --n_cpu 5 --batch_size 128

conda deactivate