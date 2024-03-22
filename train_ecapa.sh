#!/bin/bash
# Job names: ecapa(voxceleb) ecapa_cc(cn-celeb) ecapa-vc(vietnam-celeb)
#SBATCH --job-name=ecapa-vc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Max_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 trainECAPAModel.py --config config_ecapa.yml
# python3 trainECAPAModel.py --config config_ecapa_cn_celeb.yml
python3 trainECAPAModel.py --config config_ecapa_vietnam_celeb.yml

conda deactivate