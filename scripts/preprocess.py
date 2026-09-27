"""Build the balanced Benign-vs-DoS datasets from the raw NetFlow v2 CSVs.

    python scripts/preprocess.py                      # all four datasets
    python scripts/preprocess.py --datasets NF-UNSW-NB15-v2

Reads  data/raw/<name>.csv            (from https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
Writes data/processed/<name>_processed.csv

This reproduces the processed files used in the paper byte for byte. Per dataset:

  1. Read the raw CSV in chunks of 4,000,000 rows.
  2. Drop the flow identifiers IPV4_SRC_ADDR, L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT.
  3. Keep Benign and DoS flows only. DoS is 'DoS' (UNSW-NB15, BoT-IoT), 'dos' (ToN-IoT)
     or the four 'DoS attacks-*' classes (CSE-CIC-IDS2018); all are relabelled 'DoS'.
  4. Within each chunk, undersample the larger class to the size of the smaller one with
     DataFrame.sample(random_state=42), and write the DoS rows followed by the Benign rows.
  5. Drop rows with a value outside the float32 range. A handful of flows carry
     SRC_TO_DST_SECOND_BYTES values of 1e39 and above.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cdids import config

CHUNKSIZE = 4_000_000
SEED = 42
IDENTIFIERS = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT"]
DOS_CLASSES = {
    "NF-UNSW-NB15-v2": lambda attack: attack == "DoS",
    "NF-BoT-IoT-v2": lambda attack: attack == "DoS",
    "NF-ToN-IoT-v2": lambda attack: attack == "dos",
    "NF-CSE-CIC-IDS2018-v2": lambda attack: attack.str.startswith("DoS attacks-"),
}
FLOAT32_MAX = np.finfo(np.float32).max


def balance_chunk(chunk: pd.DataFrame, is_dos) -> pd.DataFrame:
    chunk = chunk.drop(columns=IDENTIFIERS)
    dos = chunk[is_dos(chunk[config.ATTACK_COL])].assign(**{config.ATTACK_COL: "DoS"})
    benign = chunk[chunk[config.ATTACK_COL] == "Benign"]
    n = min(len(dos), len(benign))
    if len(dos) > n:
        dos = dos.sample(n=n, random_state=SEED)
    if len(benign) > n:
        benign = benign.sample(n=n, random_state=SEED)
    out = pd.concat([dos, benign])
    numeric = out.select_dtypes("number")
    return out[(numeric.abs() <= FLOAT32_MAX).all(axis=1)]


def preprocess(name: str, raw_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_processed.csv"
    counts = {"DoS": 0, "Benign": 0}
    with open(out_path, "w") as f:
        for i, chunk in enumerate(pd.read_csv(raw_dir / f"{name}.csv", chunksize=CHUNKSIZE)):
            part = balance_chunk(chunk, DOS_CLASSES[name])
            part.to_csv(f, index=False, header=(i == 0))
            for k, v in part[config.ATTACK_COL].value_counts().items():
                counts[k] += int(v)
    print(f"{name}: {counts['DoS']:,} DoS + {counts['Benign']:,} Benign -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DOS_CLASSES), choices=list(DOS_CLASSES))
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    args = parser.parse_args()
    for name in args.datasets:
        preprocess(name, args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
