#!/bin/bash

#SBATCH --job-name=run_resnet110
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --mem=8GB
#SBATCH --output=resnet110_%j.out
#SBATCH --error=resnet110_%j.err


eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap

## Source bashrc to get conda
#source ~/.bashrc
#conda activate newcap
#cd proect2/network/

echo "Running RESNET110..."
python -m cifar10_evaluating_resnet

conda deactivate

# To run it:
# open cluster shell, navigate to the correct folder and run the following:

# sbatch --job-name run_resnet56 teo_script.sh


