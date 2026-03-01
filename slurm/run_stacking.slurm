#!/bin/bash
#SBATCH --job-name=c13k-stack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL

module purge
module load anaconda3/2024.6
conda activate c13k-stacking

cd $SLURM_SUBMIT_DIR
python scripts/run_stacking_mvp.py
