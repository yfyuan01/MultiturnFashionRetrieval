#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000M
#SBATCH --partition=infofil02
python get_score.py --backbone resnet101 --batch_size 8 --gpu_id 0 --target dress --text_method encode
