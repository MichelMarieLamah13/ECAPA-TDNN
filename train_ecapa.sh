#!/bin/bash
#SBATCH --job-name=b_fbank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_24GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 -m pdb trainECAPAModel.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128
# python3 trainECAPAModel.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128
python3 trainECAPAModel.py --save_path exps/exp1_ddp --feat_type fbank --feat_dim 80 --n_cpu 16 --batch_size 512

conda deactivate