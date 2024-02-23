#!/bin/bash
#SBATCH --job-name=b_fbank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPURAM_Min_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModelDDP.py --save_path exps/exp1_ddp --feat_type fbank --feat_dim 80 --n_cpu 16 --batch_size 256 --master_port 54323

conda deactivate