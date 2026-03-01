#!/bin/bash
#SBATCH --job-name=c13k-bayes
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@princeton.edu

# =============================================================================
# Bayesian Stacking on Della
# =============================================================================
# Resources:
#   - 8 CPUs: for PyMC parallel chains (4 chains x 2 threads)
#   - 32GB RAM: PyMC can be memory-hungry with large datasets
#   - 8 hours: full MCMC takes ~2-4 hours, buffer for safety
#
# Usage:
#   sbatch slurm/run_bayesian.slurm           # production (4000 samples)
#   sbatch slurm/run_bayesian.slurm --quick   # quick test (500 samples)
# =============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# --- Load modules ---
module purge
module load anaconda3/2024.6

# --- Activate environment ---
conda activate c13k

# --- Set PyMC threading ---
export PYTENSOR_FLAGS="device=cpu,floatX=float64"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Change to repo directory ---
# SLURM_SUBMIT_DIR is where sbatch was called from
# If submitted from repo root: stay there
# If submitted from slurm/: go up one level
if [ -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -f "$SLURM_SUBMIT_DIR/../pyproject.toml" ]; then
    cd "$SLURM_SUBMIT_DIR/.."
else
    echo "ERROR: Cannot find repo root (pyproject.toml)"
    echo "Submit from repo directory: cd /path/to/choices13k-stacking && sbatch slurm/run_bayesian.slurm"
    exit 1
fi
echo "Working directory: $(pwd)"

# --- Create results directory ---
mkdir -p results

# --- Run tests first ---
echo ""
echo "Running tests..."
pytest tests/ -v --tb=short -x
if [ $? -ne 0 ]; then
    echo "TESTS FAILED — aborting."
    exit 1
fi

# --- Run Bayesian analysis ---
echo ""
echo "Running Bayesian analysis..."
python scripts/run_bayesian_full.py "$@"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
