#!/bin/bash
#SBATCH --job-name=c13k-stack
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

# --- Della setup ---
module purge
module load anaconda3/2024.6
conda activate stacking

# --- Run ---
cd $SLURM_SUBMIT_DIR
mkdir -p results

echo "Starting stacking run at $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"

python scripts/run_stacking_mvp.py

echo "Finished at $(date)"
