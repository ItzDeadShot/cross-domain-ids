"""Create the stratified train/val/test splits from the processed datasets.

    python scripts/make_splits.py

Reads  data/processed/<name>_processed.csv
Writes data/split-data/<name>_processed_{train,val,test}.csv (39 features + "label")
"""
import argparse

import pandas as pd

from cdids import config
from cdids.data import split_dataframe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(config.DATASETS), choices=list(config.DATASETS))
    args = parser.parse_args()

    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        stem = config.DATASETS[name]
        df = pd.read_csv(config.PROCESSED_DIR / f"{stem}.csv")
        df = df.drop(columns=[config.ATTACK_COL]).rename(columns={config.LABEL_COL: config.SPLIT_LABEL_COL})
        train, val, test = split_dataframe(df)
        for split, part in [("train", train), ("val", val), ("test", test)]:
            part.to_csv(config.split_path(name, split), index=False)
        print(f"{name}: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
