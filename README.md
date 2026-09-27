# Cross-Domain Evaluation of Deep-Learning NIDS on NetFlow v2

Code and results for the paper *Unveiling the Generalizability Gap: A Cross-Domain Evaluation of Machine Learning Algorithms for Network Intrusion Detection*, presented at IEEE ICETAS 2024 ([doi:10.1109/ICETAS62372.2024.11119949](https://doi.org/10.1109/ICETAS62372.2024.11119949)).

How well does a network intrusion detection model trained on one network generalise to another? This repository trains four deep-learning architectures (MLP, DNN, GRU, LSTM) on each of four NetFlow v2 datasets. It then tests every model on the held-out test split of every dataset, giving a 4 × 4 source → target matrix per architecture.

The headline result: every model reaches ≥ 99.4% F1 in-domain, but the mean cross-domain F1 falls to 0.5–36%.

## Repository layout

```
├── src/cdids/                 library code: config, data loading/splitting, model definitions, metrics
├── scripts/
│   ├── preprocess.py          raw NetFlow v2 CSVs -> balanced Benign-vs-DoS datasets
│   ├── make_splits.py         processed CSVs -> stratified 72/18/10 train/val/test splits
│   ├── train_base_models.py   Hyperband tuning + training of the 16 base models
│   ├── cross_domain_eval.py   4 x 4 source -> target evaluation per architecture
│   └── plot_base_results.py   tuning heatmaps and training curves from the recorded runs
├── models/                    the 16 trained base models (Keras 3, .keras)
├── notebooks/                 original execution records (code + outputs) of the reported runs
│   ├── base_training/<ARCH>/  one notebook per architecture x dataset
│   └── cross_domain/          Cross-<ARCH>.ipynb (target scaling), Cross-Best.ipynb (source scaling)
├── results/
│   ├── base_models/           test metrics, chosen hyperparameters, per-epoch history, tuning trials, figures
│   └── cross_domain/
│       ├── target/<ARCH>/     4 x 4 matrices (csv + png) and in- vs cross-domain summary per architecture
│       └── source_best/       best model per dataset, scaler fit on the source dataset
└── data/                      not tracked; see data/README.md
```

## Setup

Python 3.10 with TensorFlow 2.17 / Keras 3.5 (the versions the models were saved with):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Then put the data in place as described in [`data/README.md`](data/README.md).

## Reproducing the results

| Step | Command | Output |
|---|---|---|
| 0. Build the balanced datasets from the raw CSVs | `python scripts/preprocess.py` | `data/processed/` (byte-identical to the files used in the paper) |
| 1. Split the processed data | `python scripts/make_splits.py` | `data/split-data/` (byte-identical to the splits used in the paper) |
| 2. *(optional)* Retrain the base models | `python scripts/train_base_models.py` | `runs/` (the models in `models/` are not overwritten) |
| 3. Cross-domain matrices, target scaling | `python scripts/cross_domain_eval.py --scaling target` | `results/cross_domain/target/<ARCH>/` |
| 4. Cross-domain, source scaling | `python scripts/cross_domain_eval.py --scaling source` | `results/cross_domain/source/<ARCH>/` |
| 5. Tuning and training figures | `python scripts/plot_base_results.py` | `results/base_models/figures/` |

Steps 3–5 need only the provided models and run in a few minutes on a CPU.

## Method

**Data.** Four NetFlow v2 datasets, reduced to a balanced binary task (Benign vs. DoS) over 39 flow features. Source IP/port and destination IP/port are removed. Balancing undersamples the larger class with `random_state=42` within 4,000,000-row chunks of each raw file. See [`data/README.md`](data/README.md) for the full procedure.

| Dataset | Flows | Train / Val / Test |
|---|---:|---|
| NF-UNSW-NB15-v2 | 11,588 | 8,343 / 2,086 / 1,159 |
| NF-BoT-IoT-v2 | 270,074 | 194,452 / 48,614 / 27,008 |
| NF-ToN-IoT-v2 | 1,425,216 | 1,026,155 / 256,539 / 142,522 |
| NF-CSE-CIC-IDS2018-v2 | 967,992 | 696,953 / 174,239 / 96,800 |

**Scaling.** `QuantileTransformer(output_distribution="normal")`, fit on the training split for training.

**Models** (`src/cdids/models.py`); all use a sigmoid output, binary cross-entropy and Adam:

| Architecture | Hidden layers |
|---|---|
| MLP | Dense 24 → 16 |
| DNN | Dense 32 → 16 → 8 → 4 → 2 |
| GRU | Reshape (1, 39) → GRU 24 → GRU 16 → Dense 8 |
| LSTM | Reshape (1, 39) → LSTM 24 → LSTM 16 → Dense 8 |

**Tuning and training.** Keras Tuner Hyperband over learning rate {1e-4, 1e-3, 1e-2} × batch size {32, 64, 128, 256}, with `val_accuracy` as the objective. The 12-configuration space is exhausted in Hyperband's first round, so in practice this is a grid search with 2 epochs per trial (`results/base_models/tuning/`). The best configuration is then trained for 30 epochs, and the checkpoint with the lowest `val_loss` is kept. The chosen values are in `results/base_models/best_hyperparameters.csv`.

**Cross-domain evaluation.** Each model is evaluated on every dataset's test split with a 0.5 threshold, reporting accuracy, precision, recall, F1 and FPR. Two scaling protocols are provided:

- **target**: the scaler is refit on each target test set. This produces the per-architecture matrices in `results/cross_domain/target/`.
- **source**: the source dataset's scaler is reused unchanged on every target, so no target statistics are used. `results/cross_domain/source_best/` holds this evaluation for the best model per dataset, with the scaler fit on the source validation split.

## Results

In-domain F1 vs. mean F1 on the three other datasets (target scaling, %). From `results/cross_domain/target/<ARCH>/<ARCH>_CrossDomain_Evaluation_Summary.csv`:

| Source dataset | MLP in / cross | DNN in / cross | GRU in / cross | LSTM in / cross |
|---|---|---|---|---|
| NFv2-UNSW-NB15 | 99.74 / 36.13 | 99.74 / 35.46 | 99.83 / 21.04 | 99.74 / 9.96 |
| NFv2-BoT-IoT | 100.00 / 4.49 | 99.98 / 2.85 | 99.99 / 11.92 | 99.99 / 4.35 |
| NFv2-ToN-IoT | 99.49 / 0.98 | 99.65 / 0.46 | 99.60 / 1.08 | 99.71 / 0.89 |
| NFv2-CIC-2018 | 99.99 / 9.50 | 99.99 / 5.59 | 100.00 / 7.01 | 100.00 / 4.33 |

Per-pair values for all five metrics are in the `*_Matrix.csv` files next to each summary.

## Notes on reproducibility

- **Scaler randomness.** `QuantileTransformer` fits its quantiles on a random subsample of 10,000 rows, and the reported runs did not seed it. Re-running therefore reproduces most matrix cells to within a few tenths of a point, but a few cells are sensitive to the subsample. For example, GRU UNSW-NB15 → BoT-IoT ranges from 43.7 to 85.8% F1 over ten scaler seeds (43.8% reported). Pass `--seed` to `cross_domain_eval.py` for deterministic output.
- **Keras version.** Use Keras 3.5.0, the version the models were saved with. Under Keras 3.9 the LSTM models give slightly different predictions for inputs far outside their training range; on in-range inputs the two versions agreed in our checks. MLP, DNN and GRU predictions were identical across both versions.
- **Duplicate flows.** Many rows in the processed datasets have identical feature values: 89% for BoT-IoT, 64% for ToN-IoT, 46% for UNSW-NB15 and 34% for CIC-2018. This is a property of the NetFlow v2 data once flow identifiers are removed; the preprocessing does not create the duplicates.
  - **Why they occur.** Every raw flow is unique, but mostly because of its source port, which the operating system assigns more or less at random. The identifiers (`IPV4_SRC_ADDR`, `L4_SRC_PORT`, `IPV4_DST_ADDR`, `L4_DST_PORT`) are removed so that models learn traffic behaviour rather than the addresses of specific testbed hosts. The remaining 39 features (packet and byte counts, TCP flags, TTLs, packet-size histograms, durations in milliseconds) are coarse, and network traffic is highly repetitive: DoS tools send the same request over and over, and benign traffic contains many identical short exchanges such as DNS lookups, keep-alives and bare TCP handshakes. Once the identifiers are removed, these flows become indistinguishable. For example, 93% of BoT-IoT DoS rows are duplicates, and a single ToN-IoT DoS feature vector occurs 36,234 times. No feature vector occurs under both labels in any dataset, so the duplicates do not introduce label noise.
  - **Effect.** The reported results use the data as described above, split at random, so identical feature vectors can appear in both the training and test splits. Within a dataset, 38% (CIC-2018) to 91% (BoT-IoT) of the test rows have a feature vector that also occurs in the training split, which makes the in-domain scores optimistic. Cross-domain scores are much less affected: across datasets the overlap is 0% for most pairs and at most 10% (ToN-IoT training data vs. BoT-IoT test data). To avoid the overlap, deduplicate on the 39 features before splitting, or use a group-aware split (e.g. `GroupShuffleSplit`) that keeps identical vectors on the same side.
- **Provenance.** `results/` holds the outputs of the original runs. The 4 × 4 matrix CSVs were extracted from the logged outputs of `notebooks/cross_domain/`, and they reproduce the saved summary tables exactly. Training histories were extracted from the Keras logs in `notebooks/base_training/`.
- **Base-model test metrics vs. saved models.** The training notebooks computed `results/base_models/test_metrics.csv` and the confusion matrices with the in-memory model after the last epoch. The files in `models/` are the lowest-`val_loss` checkpoints, which can differ by a few test predictions. For example, DNN on UNSW-NB15 has 7 false positives after the last epoch and 3 with the saved checkpoint. All cross-domain results use the saved checkpoints.

## Citation

If you use this code or these results, please cite:

> M. I. Amin, M. Shen, S. Ul Arfeen Laghari, M. Parthipan and S. Karuppayah, "Unveiling the Generalizability Gap: A Cross-Domain Evaluation of Machine Learning Algorithms for Network Intrusion Detection," in *2024 IEEE 9th International Conference on Engineering Technologies and Applied Sciences (ICETAS)*, 2024, pp. 1–7, doi: 10.1109/ICETAS62372.2024.11119949.

```bibtex
@inproceedings{11119949,
  title     = {Unveiling the Generalizability Gap: A Cross-Domain Evaluation of Machine Learning Algorithms for Network Intrusion Detection},
  booktitle = {2024 {{IEEE}} 9th International Conference on Engineering Technologies and Applied Sciences ({{ICETAS}})},
  author    = {Amin, Muhammad Iqrar and Shen, Menqing and Ul Arfeen Laghari, Shams and Parthipan, Mithiiran and Karuppayah, Shankar},
  year      = 2024,
  pages     = {1--7},
  doi       = {10.1109/ICETAS62372.2024.11119949}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The NetFlow v2 datasets are distributed by their original authors under their own terms.
