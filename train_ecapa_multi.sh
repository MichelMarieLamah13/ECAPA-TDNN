#!/bin/bash
#SBATCH --job-name=fbank_multi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j_output.log
#SBATCH --error=%x_%j_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModelMulti.py
#python3 -m pdb trainECAPAModelMulti.py

conda deactivate