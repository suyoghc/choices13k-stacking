#!/bin/bash
# =============================================================================
# One-time setup on Della
# =============================================================================
# Run this once after cloning the repo:
#   cd /scratch/gpfs/YOUR_NETID/choices13k-stacking
#   bash slurm/setup_della.sh
# =============================================================================

set -e  # Exit on error

echo "Setting up choices13k-stacking environment on Della..."

# Load modules
module purge
module load anaconda3/2024.6

# Create conda environment
echo "Creating conda environment..."
conda create -n c13k python=3.11 -y
conda activate c13k

# Install package with all dependencies
echo "Installing package..."
pip install -e ".[bayesian,dev]"

# Verify installation
echo "Verifying installation..."
python -c "import pymc; print(f'PyMC version: {pymc.__version__}')"
python -c "import stacking; print('stacking package OK')"

# Run tests
echo "Running tests..."
pytest tests/ -v

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To run Bayesian analysis:"
echo "  sbatch slurm/run_bayesian.slurm"
echo ""
echo "To run quick test:"
echo "  sbatch slurm/run_bayesian.slurm --quick"
echo ""
