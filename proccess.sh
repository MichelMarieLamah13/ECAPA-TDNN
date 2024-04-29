#!/bin/bash
#SBATCH --job-name=bp
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --output=bp_output.log
#SBATCH --error=bp_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# rm -rf data
# du -sh /local_disk/helios/mmlamah/
# du -sh /local_disk/helios/mmlamah/projects/kiwano-project/recipes/resnet/db
# du -sh /local_disk/helios/mmlamah/projects/kiwano-project/recipes/resnet/db
# du -sh /local_disk/helios/mmlamah/projects/dataset/db

#cp -r ./../dataset/db/musan_split ./../kiwano-project/recipes/resnet/db/
#rm -rf ./../dataset/db
#du -sh /local_disk/helios/mmlamah/

#rm -rf /local_disk/helios/mmlamah/projects/kiwano-project/recipes/resnet/db/nist
#du -sh /local_disk/helios/mmlamah/

conda deactivate