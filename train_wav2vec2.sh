#!/bin/bash
#SBATCH --job-name=train_wav2vec2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=5
#SBATCH --output=train_wav2vec2_output.log
#SBATCH --error=train_wav2vec2_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --save_path exps/exp2
#python3 -m pdb trainECAPAModel.py --save_path exps/exp2

conda deactivate