"""Evaluate every base model on the test split of every dataset (4 x 4 per architecture).

    python scripts/cross_domain_eval.py                        # all architectures, target scaling
    python scripts/cross_domain_eval.py --arch MLP --scaling source

Feature scaling (QuantileTransformer, normal output) has two modes:

  target  The scaler is refit on each target test set before prediction. This is
          the protocol behind the per-architecture matrices reported in the paper
          (notebooks/cross_domain/Cross-{ARCH}.ipynb).
  source  The scaler is fit once on the source dataset (the one the model was
          trained on) and applied unchanged to every target. No target
          statistics are used. notebooks/cross_domain/Cross-Best.ipynb used this
          mode with --source-split val.

Outputs, in results/cross_domain/<scaling>/<ARCH>/:
  <ARCH>_<Metric>_Matrix.csv/.png           source x target, in percent
  <ARCH>_CrossDomain_Evaluation_Summary.csv in-domain vs. mean cross-domain
  <ARCH>_Confusion_Counts.csv               TN/FP/FN/TP for every pair
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

from cdids import config
from cdids.data import fit_scaler, load_split
from cdids.metrics import METRICS, binary_metrics, cross_domain_summary

HEATMAP_LABELS = {"Accuracy": "Accuracy", "Precision": "Precision", "Recall": "Recall", "F1": "F1-Score", "FPR": "FPR"}


def plot_matrix(matrix: pd.DataFrame, metric: str, arch: str, path):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=".2f",
        # Lower is better for FPR, so reverse the colour map.
        cmap="YlGnBu_r" if metric == "FPR" else "YlGnBu",
        cbar_kws={"label": f"{HEATMAP_LABELS[metric]} (%) - {arch}"},
    )
    ax.set_xlabel("Target Dataset", fontsize=12)
    ax.set_ylabel("Source Dataset", fontsize=12)
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(rotation=90, fontsize=9)
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


def evaluate_architecture(arch, datasets, test_sets, scaling, source_split, seed, out_dir):
    matrices = {m: pd.DataFrame(index=datasets, columns=datasets, dtype=float) for m in METRICS}
    counts = []
    for source in datasets:
        model = tf.keras.models.load_model(config.model_path(arch, source))
        source_scaler = fit_scaler(load_split(source, source_split)[0], seed) if scaling == "source" else None
        for target in datasets:
            X_test, y_test = test_sets[target]
            scaler = source_scaler if scaling == "source" else fit_scaler(X_test, seed)
            y_prob = model.predict(scaler.transform(X_test), verbose=0, batch_size=4096)
            result = binary_metrics(y_test, y_prob)
            for m in METRICS:
                matrices[m].loc[source, target] = round(result[m], 2)
            counts.append({"Source": source, "Target": target, **{k: result[k] for k in ["TN", "FP", "FN", "TP"]}})
            print(f"  {arch} {source:>15} -> {target:<15} "
                  + "  ".join(f"{m} {result[m]:6.2f}" for m in METRICS))
        tf.keras.backend.clear_session()

    out_dir.mkdir(parents=True, exist_ok=True)
    for m in METRICS:
        matrices[m].to_csv(out_dir / f"{arch}_{m}_Matrix.csv")
        plot_matrix(matrices[m], m, arch, out_dir / f"{arch}_{m}_Matrix.png")
    pd.DataFrame(counts).to_csv(out_dir / f"{arch}_Confusion_Counts.csv", index=False)
    summary = cross_domain_summary(matrices)
    summary.to_csv(out_dir / f"{arch}_CrossDomain_Evaluation_Summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", nargs="+", default=config.ARCHITECTURES, choices=config.ARCHITECTURES)
    parser.add_argument("--scaling", choices=["target", "source"], default="target")
    parser.add_argument("--source-split", choices=["train", "val"], default="train",
                        help="split used to fit the scaler in source mode (default: train, as in training)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random_state for the QuantileTransformer's subsampling (default: unseeded, as originally run)")
    parser.add_argument("--out", default=None, help="output directory (default: results/cross_domain/<scaling>)")
    args = parser.parse_args()

    datasets = list(config.DATASETS)
    test_sets = {name: load_split(name, "test") for name in datasets}
    out_root = config.RESULTS_DIR / "cross_domain" / args.scaling if args.out is None else Path(args.out)
    for arch in args.arch:
        print(f"\n=== {arch} ({args.scaling} scaling) ===")
        summary = evaluate_architecture(arch, datasets, test_sets, args.scaling, args.source_split, args.seed,
                                        out_root / arch)
        print(summary[["Dataset", "In-Domain F1", "Avg Cross F1", "F1 Decay"]].to_string(index=False))


if __name__ == "__main__":
    main()
