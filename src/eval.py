from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Metrics:
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
    )


def roc_points(y_true: np.ndarray, scores: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, scores)
    return fpr, tpr, thr
