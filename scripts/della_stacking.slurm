#!/bin/bash
#SBATCH --job-name=c13k-stacking
#SBATCH --output=results/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL

# --- Load modules (adjust for Della's module system) ---
module purge
module load anaconda3/2024.6

# --- Setup environment ---
# First time: conda create -n c13k python=3.11 && pip install -e ".[dev]"
conda activate c13k

# --- Download data if not present ---
if [ ! -f data/c13k_selections.csv ]; then
    echo "Downloading choices13k data..."
    mkdir -p data
    curl -sL https://raw.githubusercontent.com/jcpeterson/choices13k/main/c13k_selections.csv -o data/c13k_selections.csv
    curl -sL https://raw.githubusercontent.com/jcpeterson/choices13k/main/c13k_problems.json -o data/c13k_problems.json
fi

# --- Run tests first (Poldrack: tests before results) ---
echo "Running tests..."
pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo "TESTS FAILED — aborting."
    exit 1
fi

# --- Run stacking ---
echo "Running stacking pipeline..."
python scripts/run_stacking_mvp.py

echo "Done. Results in results/"
