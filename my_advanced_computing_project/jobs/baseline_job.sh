#!/bin/bash
#SBATCH --job-name=baseline_imagenet
#SBATCH --account=f202500010hpcvlabuminhog
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

module load Python/3.12.3-GCCcore-13.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source /projects/F202500010HPCVLABUMINHO/uminhocp150/DATALOADER_PROJECT/venv_adaptive/bin/activate

cd /projects/F202500010HPCVLABUMINHO/uminhocp150/DATALOADER_PROJECT/my_advanced_computing_project

python scripts/train_baseline.py
