"""Binary detection metrics and the in-domain vs. cross-domain summary."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

METRICS = ["Accuracy", "Precision", "Recall", "F1", "FPR"]


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """All metrics in percent. FPR = FP / (FP + TN)."""
    y_pred = (np.ravel(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "F1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "FPR": (fp / (fp + tn) if (fp + tn) > 0 else 0.0) * 100,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def cross_domain_summary(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per source dataset: in-domain score, mean over the other targets, and the drop.

    ``matrices`` maps a metric name to a source x target DataFrame. For FPR the
    reported change is an increase (cross minus in-domain); for every other metric
    it is a decay (in-domain minus cross).
    """
    records = []
    for source in matrices["F1"].index:
        row = {"Dataset": source}
        for m in ["F1", "Accuracy", "Precision", "Recall", "FPR"]:
            row[f"In-Domain {m}"] = matrices[m].loc[source, source]
        for m in ["F1", "Accuracy", "Precision", "Recall", "FPR"]:
            row[f"Avg Cross {m}"] = matrices[m].loc[source].drop(source).mean()
        for m in ["F1", "Accuracy", "Precision", "Recall"]:
            row[f"{m} Decay"] = row[f"In-Domain {m}"] - row[f"Avg Cross {m}"]
        row["FPR Increase"] = row["Avg Cross FPR"] - row["In-Domain FPR"]
        records.append(row)
    return pd.DataFrame(records).round(2)
