#!/bin/bash
#SBATCH --job-name=bw
#SBATCH --partition=gpu
# #SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
# #SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=10-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --output=bw_output.log
#SBATCH --error=bw_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --save_path exps/exp2 --feat_type wav2vec2 --feat_dim 768 --n_cpu 8 --batch_size 256

conda deactivate