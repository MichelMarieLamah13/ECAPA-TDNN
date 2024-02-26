#!/bin/bash
#SBATCH --job-name=b_fbank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 -m pdb trainECAPAModel.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128
# python3 trainECAPAModel.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128
python3 trainECAPAModel.py --save_path exps/exp1_ddp --feat_type fbank --feat_dim 80 --n_cpu 8 --batch_size 256 --max_epoch 100

conda deactivate