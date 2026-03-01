"""Analysis configuration for choices13k stacking project.

Uses dataclasses to bundle parameters (Poldrack §3).
Named constants replace magic numbers (Poldrack §2).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --- Named constants (Poldrack §2: no magic numbers) ---

# Default number of CV folds for stacking
# 10-fold standard per Breiman (1996) and Vehtari & Ojanen (2012)
DEFAULT_N_FOLDS = 10

DEFAULT_RANDOM_SEED = 42

# Softmax temperature bounds for grid search
# Peterson et al. (2021) used learned temperature per model
TEMPERATURE_BOUNDS = (0.01, 100.0)

# Optimizer defaults for MLE fitting
MAX_OPTIMIZER_ITERATIONS = 1000
OPTIMIZER_TOLERANCE = 1e-8


@dataclass
class DataConfig:
    """Where to find data and which subset to use."""

    data_dir: Path = Path("data")
    selections_file: str = "c13k_selections.csv"
    problems_file: str = "c13k_problems.json"
    # None = use all rows; "feedback" or "no_feedback" to filter
    feedback_filter: Optional[str] = None


@dataclass
class StackingConfig:
    """Parameters for K-fold stacking."""

    n_folds: int = DEFAULT_N_FOLDS
    random_seed: int = DEFAULT_RANDOM_SEED
    loss: str = "mse"  # "mse" or "cross_entropy"


@dataclass
class ModelFitConfig:
    """Parameters for fitting individual decision models."""

    max_iterations: int = MAX_OPTIMIZER_ITERATIONS
    tolerance: float = OPTIMIZER_TOLERANCE
    random_seed: int = DEFAULT_RANDOM_SEED
    n_restarts: int = 5  # multi-start optimization to avoid local minima


@dataclass
class AnalysisConfig:
    """Top-level config bundling all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    model_fit: ModelFitConfig = field(default_factory=ModelFitConfig)
