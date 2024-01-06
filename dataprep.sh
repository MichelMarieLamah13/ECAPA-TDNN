#!/bin/bash
#SBATCH --job-name=dataprep_ecapa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
# #SBATCH --cpus-per-task=20
#SBATCH --output=dataprep_output.log
#SBATCH --error=dataprep_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

#python3 dataprep.py --save_path data --download --user USERNAME --password PASSWORD
#python3 dataprep.py --save_path data --extract
python3 dataprep.py --save_path data --convert
#python3 dataprep.py --save_path data --augment

conda deactivate