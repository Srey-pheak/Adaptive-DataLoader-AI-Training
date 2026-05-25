#!/bin/bash
#SBATCH --job-name=adaptive_alexnet
#SBATCH --account=f202500010hpcvlabuminhog
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/adaptive_%j.out
#SBATCH --error=logs/adaptive_%j.err

echo "Job started  : $(date)"
echo "Node         : $SLURMD_NODENAME"

echo "[1] Loading Python module..."
module load Python/3.12.3-GCCcore-13.3.0
echo "[2] Python loaded"

echo "[3] Loading PyTorch module..."
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
echo "[4] PyTorch loaded"

echo "[5] Activating venv..."
source /projects/F202500010HPCVLABUMINHO/uminhocp150/DATALOADER_PROJECT/venv_adaptive/bin/activate
echo "[6] venv activated"

echo "[7] Changing directory..."
cd /projects/F202500010HPCVLABUMINHO/uminhocp150/DATALOADER_PROJECT/my_advanced_computing_project
echo "[8] Starting Python..."

python -u scripts/adaptive_train.py

echo "Job finished : $(date)"