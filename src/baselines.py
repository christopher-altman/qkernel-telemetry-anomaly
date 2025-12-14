from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM, SVC


@dataclass
class BaselineConfig:
    rbf_c: float = 3.0
    rbf_gamma: str = "scale"
    oneclass_nu: float = 0.1
    oneclass_gamma: str = "scale"
    iforest_contamination: float = 0.15
    random_state: int = 0


def rbf_svm_scores(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    clf = SVC(C=cfg.rbf_c, kernel="rbf", gamma=cfg.rbf_gamma, probability=True, random_state=cfg.random_state)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def oneclass_svm_scores(X_train_normals: np.ndarray, X_test: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    clf = OneClassSVM(kernel="rbf", nu=cfg.oneclass_nu, gamma=cfg.oneclass_gamma)
    clf.fit(X_train_normals)
    # decision_function: larger = more normal → flip sign to get anomaly score
    return -clf.decision_function(X_test)


def isolation_forest_scores(X_train: np.ndarray, X_test: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    clf = IsolationForest(
        n_estimators=300,
        contamination=cfg.iforest_contamination,
        random_state=cfg.random_state,
    )
    clf.fit(X_train)
    # score_samples: larger = more normal → flip sign
    return -clf.score_samples(X_test)


def precomputed_kernel_svm_scores(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_test: np.ndarray,
    C: float = 3.0,
    random_state: int = 0,
) -> np.ndarray:
    clf = SVC(C=C, kernel="precomputed", probability=True, random_state=random_state)
    clf.fit(K_train, y_train)
    return clf.predict_proba(K_test)[:, 1]
