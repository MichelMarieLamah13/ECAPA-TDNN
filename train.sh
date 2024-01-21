#!/bin/bash
#SBATCH --job-name=bf
#SBATCH --partition=gpu
# #SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
# #SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --time=10-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --output=bf_output.log
#SBATCH --error=bf_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 4 --batch_size 256

conda deactivate