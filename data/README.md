# Data

The data files are not tracked in git. The scripts expect this layout:

```
data/
├── raw/                                     # the published NetFlow v2 CSVs (download, see below)
│   ├── NF-UNSW-NB15-v2.csv
│   ├── NF-BoT-IoT-v2.csv
│   ├── NF-ToN-IoT-v2.csv
│   └── NF-CSE-CIC-IDS2018-v2.csv
├── processed/                               # written by scripts/preprocess.py
│   └── <name>_processed.csv
└── split-data/                              # written by scripts/make_splits.py
    └── <name>_processed_{train,val,test}.csv
```

```bash
python scripts/preprocess.py    # raw -> processed   (~3 minutes, reads the raw files in 4M-row chunks)
python scripts/make_splits.py   # processed -> split-data
```

Both steps are deterministic. Their outputs are byte-identical to the files behind the reported results.

## Source datasets

The NetFlow v2 datasets are published by the University of Queensland:

> M. Sarhan, S. Layeghy, M. Portmann, "Towards a Standard Feature Set for Network Intrusion Detection System Datasets", *Mobile Networks and Applications*, 2022.

Download them from https://staff.itee.uq.edu.au/marius/NIDS_datasets/. Check the dataset terms before redistributing. Each download is a zip archive; place the CSV from its `data/` folder in `data/raw/`. These are the SHA-1 checksums of the files used here:

| File | Rows | SHA-1 |
|---|---:|---|
| NF-UNSW-NB15-v2.csv | 2,390,275 | `dde9a9e38009ec60121cbb4fd884bf2c7cf5c0b0` |
| NF-BoT-IoT-v2.csv | 37,763,497 | `cddcc616a032f55a6a5ab3e382093636449653d6` |
| NF-ToN-IoT-v2.csv | 16,940,496 | `c0e014512c9a825146c52982d3cf12d3fce1ccc9` |
| NF-CSE-CIC-IDS2018-v2.csv | 18,893,708 | `80ff7fd07e2bc31164197f07fe9d9606ee565f9a` |

The UNSW-NB15, ToN-IoT and CSE-CIC-IDS2018 checksums match the manifests in the official archives.

## Processed files (`scripts/preprocess.py`)

Each processed CSV has the 39 NetFlow v2 features plus `Label` (0 = benign, 1 = attack) and `Attack` (class name). Per dataset:

1. Read the raw CSV in chunks of 4,000,000 rows.
2. Drop the flow identifiers `IPV4_SRC_ADDR`, `L4_SRC_PORT`, `IPV4_DST_ADDR` and `L4_DST_PORT`.
3. Keep only Benign and DoS flows, relabelling DoS as `DoS`:
   - UNSW-NB15 and BoT-IoT: `DoS`
   - ToN-IoT: `dos`
   - CSE-CIC-IDS2018: `DoS attacks-Hulk`, `-GoldenEye`, `-SlowHTTPTest` and `-Slowloris` (DDoS classes are excluded)
4. Within each chunk, undersample the larger class to the size of the smaller one with `DataFrame.sample(random_state=42)`, then write the DoS rows followed by the Benign rows.
5. Drop rows with any value outside the float32 range. This removes 8 benign flows (2 in ToN-IoT, 6 in CSE-CIC-IDS2018) whose `SRC_TO_DST_SECOND_BYTES` is between 1e39 and 1e213.

Because balancing happens per chunk, the benign (or, for BoT-IoT, DoS) sample follows the distribution of DoS traffic across each file. It is not a uniform sample over the whole dataset.

| File | Rows | Benign | DoS |
|---|---:|---:|---:|
| NF-UNSW-NB15-v2_processed.csv | 11,588 | 5,794 | 5,794 |
| NF-BoT-IoT-v2_processed.csv | 270,074 | 135,037 | 135,037 |
| NF-ToN-IoT-v2_processed.csv | 1,425,216 | 712,607 | 712,609 |
| NF-CSE-CIC-IDS2018-v2_processed.csv | 967,992 | 483,993 | 483,999 |

## Splits (`scripts/make_splits.py`)

A two-stage stratified split with `random_state=42`: 10% test, then 20% of the rest for validation, giving 72 / 18 / 10 overall. It drops `Attack` and renames `Label` to `label`.
