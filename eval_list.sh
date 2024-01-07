#!/bin/bash
#SBATCH --job-name=eval_list_ecapa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
# #SBATCH --cpus-per-task=5
#SBATCH --output=eval_list_output.log
#SBATCH --error=eval_list_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt -P data/voxceleb1/
wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt -P data/voxceleb1/
wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt -P data/voxceleb1/


conda deactivate