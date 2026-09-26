"""Loading the processed NetFlow v2 datasets and their train/val/test splits."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from . import config


def split_dataframe(df: pd.DataFrame, seed: int = config.SPLIT_SEED):
    """Stratified 72/18/10 split, identical to the one used to train the base models."""
    label = config.SPLIT_LABEL_COL
    train_val, test = train_test_split(df, stratify=df[label], test_size=config.TEST_SIZE, random_state=seed)
    train, val = train_test_split(
        train_val, stratify=train_val[label], test_size=config.VAL_SIZE, random_state=seed
    )
    return train, val, test


def load_split(dataset: str, split: str):
    """Return (X, y) for one split. X holds the 39 flow features, y the binary label."""
    df = pd.read_csv(config.split_path(dataset, split))
    df = df.dropna(axis=0)
    y = df.pop(config.SPLIT_LABEL_COL).to_numpy()
    X = df.to_numpy()
    assert X.shape[1] == config.N_FEATURES, f"{dataset}/{split}: expected {config.N_FEATURES} features"
    return X, y


def make_scaler(seed: int | None = None) -> QuantileTransformer:
    """The feature scaler used for every model in the study."""
    return QuantileTransformer(output_distribution="normal", random_state=seed)


def fit_scaler(X: np.ndarray, seed: int | None = None) -> QuantileTransformer:
    return make_scaler(seed).fit(X)
