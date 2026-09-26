# Data

The data files are not tracked in git. The scripts expect this layout:

```
data/
├── processed/                               # input to scripts/make_splits.py
│   ├── NF-UNSW-NB15-v2_processed.csv
│   ├── NF-BoT-IoT-v2_processed.csv
│   ├── NF-ToN-IoT-v2_processed.csv
│   └── NF-CSE-CIC-IDS2018-v2_processed.csv
└── split-data/                              # written by scripts/make_splits.py
    └── <name>_processed_{train,val,test}.csv
```

## Source datasets

The NetFlow v2 datasets are published by the University of Queensland:

> M. Sarhan, S. Layeghy, M. Portmann, "Towards a Standard Feature Set for Network Intrusion Detection System Datasets", *Mobile Networks and Applications*, 2022.

Download them from https://staff.itee.uq.edu.au/marius/NIDS_datasets/. Check the dataset terms before redistributing.

## Processed files

Each processed CSV has the 39 NetFlow v2 features plus `Label` (0 = benign, 1 = attack) and `Attack` (class name). Compared with the published datasets:

- `IPV4_SRC_ADDR`, `L4_SRC_PORT`, `IPV4_DST_ADDR` and `L4_DST_PORT` are removed.
- Only `Benign` and `DoS` flows are kept.
- The classes are balanced 50/50:

| File | Rows | Benign | DoS |
|---|---:|---:|---:|
| NF-UNSW-NB15-v2_processed.csv | 11,588 | 5,794 | 5,794 |
| NF-BoT-IoT-v2_processed.csv | 270,074 | 135,037 | 135,037 |
| NF-ToN-IoT-v2_processed.csv | 1,425,216 | 712,607 | 712,609 |
| NF-CSE-CIC-IDS2018-v2_processed.csv | 967,992 | 483,993 | 483,999 |

The script that derives these files from the raw datasets is not yet part of this repository. Until it is, obtain the processed files from the authors.

## Splits

`scripts/make_splits.py` does a two-stage stratified split with `random_state=42`: 10% test, then 20% of the rest for validation, giving 72 / 18 / 10 overall. It drops `Attack` and renames `Label` to `label`. The output is byte-identical to the splits used for the reported results.
