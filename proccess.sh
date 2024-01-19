#!/bin/bash
#SBATCH --job-name=bp
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
##SBATCH --mem=16GB
#SBATCH --output=bp_output.log
#SBATCH --error=bp_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# rm -rf data
du -sh /local_disk/helios/mmlamah/

conda deactivate