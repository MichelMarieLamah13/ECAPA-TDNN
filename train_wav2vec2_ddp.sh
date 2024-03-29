#!/bin/bash
#SBATCH --job-name=bw_1_ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPURAM_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModelDDP.py --save_path exps/exp2_1_ddp --feat_type wav2vec2 --n_cpu 10 --batch_size 128 --model_name facebook/wav2vec2-large-960h --master_port 54324
# python3 trainECAPAModelDDP.py --save_path exps/exp2_1 --feat_type wav2vec2 --n_cpu 10 --batch_size 128 --model_name facebook/wav2vec2-large-960h

conda deactivate