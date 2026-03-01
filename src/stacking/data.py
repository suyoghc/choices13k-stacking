"""Load and validate choices13k data.

Defensive coding (Poldrack §5): validate data on load.
Announce errors loudly — raise exceptions, don't return None.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig


def load_selections(config: DataConfig) -> pd.DataFrame:
    """Load the human choice data (bRate per problem).

    Returns DataFrame with one row per problem-condition,
    validated for expected structure and value ranges.

    Adds _json_idx column: the original CSV row index,
    which is the key into the problems JSON file.
    """
    filepath = Path(config.data_dir) / config.selections_file
    if not filepath.exists():
        raise FileNotFoundError(f"Selections file not found: {filepath}")

    df = pd.read_csv(filepath)
    _validate_selections(df)

    # Track original row index — this is the key into problems JSON
    # (JSON is keyed by CSV row number, NOT by Problem ID)
    df["_json_idx"] = np.arange(len(df))

    if config.feedback_filter == "feedback":
        df = df[df["Feedback"] == True].copy()
    elif config.feedback_filter == "no_feedback":
        df = df[df["Feedback"] == False].copy()
    elif config.feedback_filter is not None:
        raise ValueError(
            f"feedback_filter must be None, 'feedback', or 'no_feedback', "
            f"got '{config.feedback_filter}'"
        )

    return df.reset_index(drop=True)


def load_problems(config: DataConfig) -> dict:
    """Load raw gamble specifications (outcome-probability pairs).

    Returns dict mapping problem_id (str) -> {"A": [...], "B": [...]}.
    """
    filepath = Path(config.data_dir) / config.problems_file
    if not filepath.exists():
        raise FileNotFoundError(f"Problems file not found: {filepath}")

    with open(filepath) as f:
        problems = json.load(f)

    _validate_problems(problems)
    return problems


def extract_problem_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract covariates for hierarchical stacking from selections data.

    These are the problem features that stacking weights can vary over:
    stake magnitude, probability structure, feedback, ambiguity, etc.
    """
    features = pd.DataFrame(index=df.index)

    # Stake features
    features["max_outcome"] = df[["Ha", "Hb"]].max(axis=1)
    features["min_outcome"] = df[["La", "Lb"]].min(axis=1)
    features["outcome_range"] = features["max_outcome"] - features["min_outcome"]

    # Domain classification
    features["is_gains_only"] = (df["La"] >= 0) & (df["Lb"] >= 0)
    features["is_losses_only"] = (df["Ha"] <= 0) & (df["Hb"] <= 0)
    features["is_mixed"] = ~features["is_gains_only"] & ~features["is_losses_only"]

    # Probability structure
    features["pHa"] = df["pHa"]
    features["pHb"] = df["pHb"]
    features["prob_asymmetry"] = np.abs(df["pHa"] - df["pHb"])

    # Design variables
    features["feedback"] = df["Feedback"].astype(float)
    features["ambiguity"] = df["Amb"].astype(float)
    features["correlation"] = df["Corr"].astype(float)
    features["lottery_outcomes"] = df["LotNumB"].astype(float)

    return features


# --- Validation helpers (Poldrack §5: the conspiracy theory paper lesson) ---


def _validate_selections(df: pd.DataFrame) -> None:
    """Check selections data meets expected structure and value ranges."""
    required_columns = {
        "Problem", "Feedback", "n", "Ha", "pHa", "La",
        "Hb", "pHb", "Lb", "LotShapeB", "LotNumB",
        "Amb", "Corr", "bRate",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in selections data: {missing}")

    # bRate must be in [0, 1] — this is a proportion
    assert df["bRate"].between(0, 1).all(), (
        f"bRate values out of [0,1] range. "
        f"Min={df['bRate'].min()}, Max={df['bRate'].max()}"
    )

    # Probabilities must be in [0, 1]
    for col in ["pHa", "pHb"]:
        assert df[col].between(0, 1).all(), (
            f"{col} values out of [0,1] range. "
            f"Min={df[col].min()}, Max={df[col].max()}"
        )

    # Sample sizes must be positive
    assert (df["n"] > 0).all(), f"Non-positive sample sizes found"

    # No NaN in critical columns
    critical = ["bRate", "Ha", "pHa", "La", "Hb", "pHb", "Lb"]
    for col in critical:
        assert df[col].notna().all(), f"NaN values in column {col}"


def _validate_problems(problems: dict) -> None:
    """Check raw problem data structure."""
    if not isinstance(problems, dict):
        raise TypeError(f"Problems must be dict, got {type(problems)}")

    if len(problems) == 0:
        raise ValueError("Problems dict is empty")

    # Spot-check first problem
    first_key = next(iter(problems))
    first = problems[first_key]
    if not {"A", "B"} <= set(first.keys()):
        raise ValueError(
            f"Problem must have keys 'A' and 'B', got {set(first.keys())}"
        )

    # Each option should be list of [probability, outcome] pairs
    for option_name in ["A", "B"]:
        option = first[option_name]
        if not isinstance(option, list) or len(option) == 0:
            raise ValueError(
                f"Option {option_name} must be non-empty list, "
                f"got {type(option)}"
            )
        for pair in option:
            if len(pair) != 2:
                raise ValueError(
                    f"Each outcome must be [prob, payoff], got length {len(pair)}"
                )
