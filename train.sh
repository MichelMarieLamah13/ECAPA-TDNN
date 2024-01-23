#!/bin/bash
#SBATCH --job-name=bf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=apollon,alpos,helios,eris
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=bf_output.log
#SBATCH --error=bf_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 -m pdb trainECAPAModel.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128
python3 trainECAPAModel.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128

conda deactivate