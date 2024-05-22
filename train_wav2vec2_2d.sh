#!/bin/bash
# job names: bw2d (base), bw2d_1 (large), bw2d_2 (robust), bw2d_3 (self)
#SBATCH --job-name=bw2d_1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Max_40GB
#SBATCH --exclude=alpos
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_%j_output.log
#SBATCH --error=%x_%j_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 trainECAPAModel.py --save_path exps/exp3_3 --feat_type wav2vec2 --n_cpu 10 --batch_size 128 --is_2d --model_name facebook/wav2vec2-large-960h

# python3 trainECAPAModel.py --save_path exps/exp3_2 --feat_type wav2vec2 --n_cpu 10 --batch_size 128 --is_2d --model_name facebook/wav2vec2-large-960h

python3 trainECAPAModel.py --save_path exps/exp3_1 --feat_type wav2vec2 --n_cpu 10 --batch_size 128 --is_2d --model_name facebook/wav2vec2-large-960h

# python3 trainECAPAModel.py --save_path exps/exp3 --feat_type wav2vec2 --n_cpu 10 --batch_size 128 --is_2d

conda deactivate