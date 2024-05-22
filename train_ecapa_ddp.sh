#!/bin/bash
#SBATCH --job-name=b_fbank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPURAM_Max_12GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j_output.log
#SBATCH --error=%x_%j_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModelDDP.py --save_path exps/exp1_ddp --feat_type fbank --feat_dim 80 --n_cpu 8 --batch_size 128 --max_epoch 99 --master_port 54323

conda deactivate