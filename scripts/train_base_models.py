"""Tune and train the base models (one per architecture x dataset).

    python scripts/train_base_models.py                          # all 16 models
    python scripts/train_base_models.py --arch GRU --dataset NFv2-CIC-2018

Procedure (as in notebooks/base_training/):
  1. Fit a QuantileTransformer (normal output) on the train split; apply it to val/test.
  2. Keras Tuner Hyperband over learning rate {1e-4, 1e-3, 1e-2} x batch size {32, 64, 128, 256},
     objective val_accuracy, max_epochs=30, factor=3. With only 12 configurations the tuner
     exhausts the space in its first round (2 epochs per trial).
  3. Train the best configuration for 30 epochs, keeping the checkpoint with the lowest val_loss.
  4. Report precision, recall, F1 and FPR on the test split.

Outputs go to runs/ by default so the published models in models/ are never overwritten:
  runs/models/<ARCH>_<Dataset>_model.keras
  runs/training_history/<ARCH>_<Dataset>.csv
  runs/tuning/<arch>_<ds>/                    Keras Tuner project
  runs/test_metrics.csv, runs/best_hyperparameters.csv
"""
import argparse
from pathlib import Path

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from cdids import config
from cdids.data import fit_scaler, load_split
from cdids.metrics import binary_metrics
from cdids.models import BATCH_SIZES, LEARNING_RATES, build_model

SHORT = {"NFv2-UNSW-NB15": "unsw", "NFv2-BoT-IoT": "bot", "NFv2-ToN-IoT": "ton", "NFv2-CIC-2018": "cic"}


class BaseHyperModel(kt.HyperModel):
    def __init__(self, arch: str):
        super().__init__()
        self.arch = arch

    def build(self, hp):
        return build_model(self.arch, hp.Choice("learning_rate", values=LEARNING_RATES))

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size=hp.Choice("batch_size", values=BATCH_SIZES), **kwargs)


def train_one(arch, dataset, out, epochs, seed):
    tf.keras.utils.set_random_seed(seed)
    X_train, y_train = load_split(dataset, "train")
    X_val, y_val = load_split(dataset, "val")
    X_test, y_test = load_split(dataset, "test")
    scaler = fit_scaler(X_train, seed)
    X_train, X_val, X_test = (scaler.transform(X) for X in (X_train, X_val, X_test))

    tuner = kt.Hyperband(
        BaseHyperModel(arch),
        objective="val_accuracy",
        max_epochs=epochs,
        factor=3,
        seed=seed,
        directory=str(out / "tuning"),
        project_name=f"{arch.lower()}_{SHORT[dataset]}",
    )
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    lr, batch_size = best_hps.get("learning_rate"), best_hps.get("batch_size")
    print(f"{arch} / {dataset}: best learning_rate={lr} batch_size={batch_size}")

    model_path = out / "models" / f"{arch}_{dataset}_model.keras"
    model = build_model(arch, lr)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ModelCheckpoint(filepath=str(model_path), monitor="val_loss", save_best_only=True, verbose=1)],
    )
    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(out / "training_history" / f"{arch}_{dataset}.csv", index=False)

    best = tf.keras.models.load_model(model_path)
    result = binary_metrics(y_test, best.predict(X_test, verbose=0, batch_size=4096))
    tf.keras.backend.clear_session()
    metrics = {"Architecture": arch, "Dataset": dataset, **{k: round(result[k], 2) for k in
                                                            ["Accuracy", "Precision", "Recall", "F1", "FPR"]}}
    hps = {"Architecture": arch, "Dataset": dataset, "learning_rate": lr, "batch_size": batch_size}
    return metrics, hps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", nargs="+", default=config.ARCHITECTURES, choices=config.ARCHITECTURES)
    parser.add_argument("--dataset", nargs="+", default=list(config.DATASETS), choices=list(config.DATASETS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(config.ROOT / "runs"))
    args = parser.parse_args()

    out = Path(args.out)
    for sub in ("models", "training_history", "tuning"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    metrics, hps = [], []
    for arch in args.arch:
        for dataset in args.dataset:
            m, h = train_one(arch, dataset, out, args.epochs, args.seed)
            metrics.append(m)
            hps.append(h)
            print(m)
    pd.DataFrame(metrics).to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(hps).to_csv(out / "best_hyperparameters.csv", index=False)


if __name__ == "__main__":
    main()
