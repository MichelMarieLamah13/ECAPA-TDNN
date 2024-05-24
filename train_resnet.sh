#!/bin/bash
# Job names: resnet(voxceleb) resnet_vc(vietnam-celeb), resnet_cc(cn-celeb)
# Job names: resnet_f_vox (finetuner voxceleb), resnet_f_vc (finetuner vietnam-celeb), resnet_f_cc (finetuner cn-celeb)
#SBATCH --job-name=resnet_cc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
#SBATCH --exclude=eris,apollon,helios
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j_output.log
#SBATCH --error=%x_%j_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 trainRESNETModel.py --config config_resnet.yml
python3 trainRESNETModel.py --config config_resnet_cn_celeb.yml
# python3 trainRESNETModel.py --config config_resnet_vietnam_celeb.yml
# python3 trainRESNETModel.py --config config_resnet_finetuner_vietnam_celeb.yml
# python3 trainRESNETModel.py --config config_resnet_finetuner_cn_celeb.yml
# python3 trainRESNETModel.py --config config_resnet_finetuner.yml
# python3 -m pdb trainRESNETModel.py

conda deactivate