"""Plot hyperparameter-tuning heatmaps and training curves for the base models.

    python scripts/plot_base_results.py

Everything is drawn from recorded values only:
  results/base_models/tuning/<arch>_<ds>/trial_*/trial.json   Keras Tuner trial records
  results/base_models/training_history/<ARCH>_<Dataset>.csv   per-epoch Keras metrics

Writes to results/base_models/figures/.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cdids import config

BASE = config.RESULTS_DIR / "base_models"
FIG = BASE / "figures"
SHORT = {"NFv2-UNSW-NB15": "unsw", "NFv2-BoT-IoT": "bot", "NFv2-ToN-IoT": "ton", "NFv2-CIC-2018": "cic"}


def load_trials(arch: str, dataset: str) -> pd.DataFrame:
    rows = []
    for path in sorted((BASE / "tuning" / f"{arch.lower()}_{SHORT[dataset]}").glob("trial_*/trial.json")):
        trial = json.loads(path.read_text())
        hp = trial["hyperparameters"]["values"]
        rows.append({"trial": trial["trial_id"], "learning_rate": hp["learning_rate"], "batch_size": hp["batch_size"],
                     "epochs": hp.get("tuner/epochs"), "val_accuracy": trial["score"], "status": trial["status"]})
    return pd.DataFrame(rows)


def plot_tuning(arch: str, dataset: str):
    trials = load_trials(arch, dataset)
    trials.to_csv(FIG / "tuning" / f"{arch}_{dataset}_trials.csv", index=False)
    done = trials[trials["status"] == "COMPLETED"]
    # Hyperband can revisit a configuration with a larger epoch budget; show the best score per cell.
    grid = done.pivot_table(index="batch_size", columns="learning_rate", values="val_accuracy", aggfunc="max")
    plt.figure(figsize=(8, 5))
    sns.heatmap(grid, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5,
                cbar_kws={"label": "Best validation accuracy"})
    plt.title(f"{arch} on {dataset}: Hyperband trials ({len(done)} completed)")
    plt.xlabel("Learning rate")
    plt.ylabel("Batch size")
    plt.tight_layout()
    plt.savefig(FIG / "tuning" / f"{arch}_{dataset}_Hyperparameter_Tuning.png", dpi=300)
    plt.close()


def plot_history_grid(metric: str, ylabel: str, log: bool):
    """One panel per architecture, one pair of lines (train/val) per dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = sns.color_palette("tab10", len(config.DATASETS))
    for ax, arch in zip(axes.flat, config.ARCHITECTURES):
        for color, dataset in zip(colors, config.DATASETS):
            h = pd.read_csv(BASE / "training_history" / f"{arch}_{dataset}.csv")
            ax.plot(h["epoch"], h[metric], color=color, label=f"{dataset} train")
            ax.plot(h["epoch"], h[f"val_{metric}"], color=color, linestyle="--", label=f"{dataset} val")
        ax.set_title(arch)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "training" / f"Training_{ylabel.replace(' ', '_')}_All.png", dpi=300)
    plt.close(fig)


def plot_history_single(arch: str, dataset: str):
    h = pd.read_csv(BASE / "training_history" / f"{arch}_{dataset}.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(h["epoch"], h["accuracy"], label="Training")
    ax1.plot(h["epoch"], h["val_accuracy"], label="Validation")
    ax1.set_ylabel("Accuracy")
    ax2.plot(h["epoch"], h["loss"], label="Training")
    ax2.plot(h["epoch"], h["val_loss"], label="Validation")
    ax2.set_ylabel("Binary cross-entropy (log scale)")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{arch} on {dataset}")
    fig.tight_layout()
    fig.savefig(FIG / "training" / f"{arch}_{dataset}_Training_Curves.png", dpi=300)
    plt.close(fig)


def main():
    (FIG / "tuning").mkdir(parents=True, exist_ok=True)
    (FIG / "training").mkdir(parents=True, exist_ok=True)
    for arch in config.ARCHITECTURES:
        for dataset in config.DATASETS:
            plot_tuning(arch, dataset)
            plot_history_single(arch, dataset)
    plot_history_grid("accuracy", "Accuracy", log=False)
    plot_history_grid("loss", "Loss", log=True)
    print(f"Figures written to {FIG.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
