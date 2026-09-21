"""Loading the Criteo Uplift dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HF_PATH = "hf://datasets/criteo/criteo-uplift/criteo-research-uplift-v2.1.csv.gz"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SAMPLE_PATH = DATA_DIR / "sample" / "sample_1m.parquet"


def load_full() -> pd.DataFrame:
    """Load the full ~14M-row dataset directly from Hugging Face."""
    return pd.read_csv(HF_PATH)


FULL_PATH = DATA_DIR / "raw" / "criteo_full.parquet"
_FULL_DTYPES = {**{f"f{i}": "float32" for i in range(12)},
                "treatment": "int8", "conversion": "int8", "visit": "int8", "exposure": "int8"}


def load_full_cached() -> pd.DataFrame:
    """Load the full dataset from a local parquet cache, downloading it once first.

    Features are stored as float32 and flags as int8 to keep the ~14M rows well under 1 GB.
    """
    if not FULL_PATH.exists():
        FULL_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(HF_PATH, dtype=_FULL_DTYPES).to_parquet(FULL_PATH, index=False)
    return pd.read_parquet(FULL_PATH)


def build_sample(n: int = 1_000_000, seed: int = 0) -> pd.DataFrame:
    """Sample n rows from the full dataset and cache to disk as parquet."""
    df = load_full().sample(n=n, random_state=seed).reset_index(drop=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SAMPLE_PATH, index=False)
    return df


def load_sample() -> pd.DataFrame:
    """Load the cached 1M-row sample, building it first if it doesn't exist yet."""
    if not SAMPLE_PATH.exists():
        return build_sample()
    return pd.read_parquet(SAMPLE_PATH)


def split_train_eval(df: pd.DataFrame, eval_size: float = 0.3, seed: int = 0):
    """Split into train/eval, stratified by treatment so both keep the ~85/15 ratio."""
    return train_test_split(df, test_size=eval_size, stratify=df["treatment"], random_state=seed)
