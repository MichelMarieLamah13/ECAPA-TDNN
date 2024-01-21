#!/bin/bash
#SBATCH --job-name=bw2d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
# #SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --time=10-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --output=bw2d_output.log
#SBATCH --error=bw2d_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --save_path exps/exp3 --feat_type wav2vec2 --feat_dim 768 --n_cpu 4 --batch_size 256 --is_2d

conda deactivate