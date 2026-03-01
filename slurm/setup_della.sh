#!/bin/bash
# Run once to set up environment on Della
module purge
module load anaconda3/2024.6
conda create -n c13k-stacking python=3.11 -y
conda activate c13k-stacking
pip install -e ".[dev]"
pytest
echo "Setup complete. Submit with: sbatch slurm/run_stacking.slurm"
