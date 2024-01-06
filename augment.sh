#!/bin/bash
#SBATCH --job-name=augment_ecapa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=5
#SBATCH --output=augment_output.log
#SBATCH --error=augment_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

#python3 dataprep.py --save_path data --download --user USERNAME --password PASSWORD
#python3 dataprep.py --save_path data --extract
#python3 dataprep.py --save_path data --convert
python3 dataprep.py --save_path data --augment

conda deactivate