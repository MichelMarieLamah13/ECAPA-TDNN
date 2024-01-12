#!/bin/bash
#SBATCH --job-name=train_ecapa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=20
#SBATCH --output=train_output.log
#SBATCH --error=train_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --save_path exps/exp3 --feat_type fbank --feat_dim 80
#python3 trainECAPAModel.py --save_path exps/exp1 --feat_type fbank --feat_dim 80
#python3 -m pdb trainECAPAModel.py --save_path exps/exp1

conda deactivate