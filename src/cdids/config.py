"""Shared constants: dataset names, file locations and split settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "split-data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Short name used throughout the paper -> file stem of the processed CSV.
DATASETS = {
    "NFv2-UNSW-NB15": "NF-UNSW-NB15-v2_processed",
    "NFv2-BoT-IoT": "NF-BoT-IoT-v2_processed",
    "NFv2-ToN-IoT": "NF-ToN-IoT-v2_processed",
    "NFv2-CIC-2018": "NF-CSE-CIC-IDS2018-v2_processed",
}

ARCHITECTURES = ["MLP", "DNN", "GRU", "LSTM"]

N_FEATURES = 39
# Columns in data/processed. The split files keep only the features and a
# lower-case "label" column (1 = attack, 0 = benign).
LABEL_COL = "Label"
ATTACK_COL = "Attack"
SPLIT_LABEL_COL = "label"

# Two-stage stratified split: 10% test, then 20% of the remainder for validation
# (overall 72% train / 18% val / 10% test).
TEST_SIZE = 0.1
VAL_SIZE = 0.2
SPLIT_SEED = 42


def model_path(arch: str, dataset: str) -> Path:
    return MODEL_DIR / f"{arch}_{dataset}_model.keras"


def split_path(dataset: str, split: str) -> Path:
    return SPLIT_DIR / f"{DATASETS[dataset]}_{split}.csv"
