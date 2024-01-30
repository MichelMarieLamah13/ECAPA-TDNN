#!/bin/bash
#SBATCH --job-name=bf_ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPURAM_Max_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModelDDP.py --save_path exps/exp1_ddp --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128 --master_port 54323
# python3 trainECAPAModelDDP.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128

conda deactivate