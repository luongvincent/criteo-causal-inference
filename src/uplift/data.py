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
