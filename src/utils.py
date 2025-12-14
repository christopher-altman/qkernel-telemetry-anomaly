"""Utility functions for quantum kernel telemetry anomaly detection."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np
from scipy.stats import entropy


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    """Save object to JSON file."""
    path = Path(path)
    if is_dataclass(obj):
        obj = asdict(obj)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def set_global_seed(seed: int) -> None:
    """Set global random seed."""
    np.random.seed(seed)


def window_features_mean_std(X: np.ndarray) -> np.ndarray:
    """Compute simple window features: per-channel mean and std.

    Args:
        X: (n_samples, timesteps, n_channels)
    Returns:
        F: (n_samples, 2*n_channels)
    """
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    return np.concatenate([mu, sd], axis=1)


def minmax_to_0_2pi(X: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> np.ndarray:
    """Scale features to [0, 2π] using train-set min/max."""
    denom = np.where((xmax - xmin) < 1e-12, 1.0, (xmax - xmin))
    z = (X - xmin) / denom
    z = np.clip(z, 0.0, 1.0)
    return z * (2.0 * np.pi)


def match_feature_dim(
    X: np.ndarray,
    target_dim: int,
    mode: Literal["pad", "tile", "truncate"] = "tile",
) -> np.ndarray:
    """Match feature dimension to target dimension.
    
    Fixes the n_qubits/feature-dimension incompatibility by allowing
    arbitrary n_qubits settings regardless of extracted feature count.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        target_dim: Desired feature dimension (typically n_qubits)
        mode: How to match dimensions:
            - 'pad': Zero-pad if too few features, truncate if too many
            - 'tile': Repeat features cyclically to reach target_dim
            - 'truncate': Only truncate if too many, error if too few
    
    Returns:
        X_matched: (n_samples, target_dim)
    """
    n_samples, n_features = X.shape
    
    if n_features == target_dim:
        return X
    
    if mode == "pad":
        if n_features < target_dim:
            # Zero-pad
            padding = np.zeros((n_samples, target_dim - n_features))
            return np.concatenate([X, padding], axis=1)
        else:
            # Truncate
            return X[:, :target_dim]
    
    elif mode == "tile":
        if n_features < target_dim:
            # Tile features cyclically
            n_repeats = (target_dim // n_features) + 1
            X_tiled = np.tile(X, (1, n_repeats))
            return X_tiled[:, :target_dim]
        else:
            # Truncate
            return X[:, :target_dim]
    
    elif mode == "truncate":
        if n_features < target_dim:
            raise ValueError(
                f"Cannot truncate {n_features} features to {target_dim}. "
                f"Use mode='pad' or mode='tile' instead."
            )
        return X[:, :target_dim]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def nystrom_factor(K_mm: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Compute Nyström factorization matrix C for kernel approximation.
    
    Given a kernel matrix K_mm computed on m landmark points,
    compute the matrix C such that K ≈ K_nm @ C @ K_nm^T
    
    Args:
    K_mm: Kernel matrix on landmarks (m, m)
    jitter: Regularization for numerical stability
    
    Returns:
    C: Factorization matrix (m, m)
    """
    m = K_mm.shape[0]
    K_mm_reg = K_mm + jitter * np.eye(m)
    
    # Eigendecomposition for stable pseudo-inverse
    eigvals, eigvecs = np.linalg.eigh(K_mm_reg)
    
    # Keep only positive eigenvalues
    pos_mask = eigvals > 1e-10
    eigvals_pos = eigvals[pos_mask]
    eigvecs_pos = eigvecs[:, pos_mask]
    
    C = eigvecs_pos @ np.diag(1.0 / eigvals_pos) @ eigvecs_pos.T
    return C


def nystrom_approx(K_nm: np.ndarray, K_mm: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Approximate kernel matrix: K_approx = K_nm @ K_mm^{-1} @ K_nm^T."""
    C = nystrom_factor(K_mm, jitter=jitter)
    K_approx = K_nm @ C @ K_nm.T
    return (K_approx + K_approx.T) / 2.0


def kernel_mean_embedding_scores(
    K_train: np.ndarray,
    K_test: np.ndarray,
    K_diag_test: np.ndarray,
) -> np.ndarray:
    """Compute one-class anomaly scores via kernel mean embedding.
    
    This is an unsupervised method that works even with label-scarce scenarios.
    
    Score formula:
        score(x) = k(x,x) - 2*E_n[k(x,n)] + E_{n,n'}[k(n,n')]
    
    where n are normal training samples.
    
    High scores indicate anomalies (far from normal distribution center in RKHS).
    
    Args:
        K_train: Kernel matrix on training data (n_train, n_train)
        K_test: Kernel matrix between test and train (n_test, n_train)
        K_diag_test: Diagonal kernel values k(x,x) for test samples (n_test,)
        
    Returns:
        scores: Anomaly scores (n_test,) - higher = more anomalous
    """
    n_train = K_train.shape[0]
    
    # E[k(n,n')] - mean of all pairwise similarities
    mean_train_train = K_train.sum() / (n_train * n_train)
    
    # E_n[k(x,n)] - mean similarity to training data
    mean_test_train = K_test.mean(axis=1)
    
    # score(x) = k(x,x) - 2*E_n[k(x,n)] + E[k(n,n')]
    scores = K_diag_test - 2.0 * mean_test_train + mean_train_train
    
    return scores


def mmd_drift_score(K_train: np.ndarray, K_test: np.ndarray, K_cross: np.ndarray) -> float:
    """Maximum Mean Discrepancy for distribution shift detection.
    
    Detects slow drift in telemetry distributions using kernel two-sample test.
    
    MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    
    where x ~ P_train, y ~ P_test
    
    Args:
        K_train: Kernel matrix on training data (n, n)
        K_test: Kernel matrix on test data (m, m)  
        K_cross: Kernel matrix between train and test (n, m)
        
    Returns:
        mmd: MMD distance (≥0), higher indicates distribution shift
    """
    n = K_train.shape[0]
    m = K_test.shape[0]
    
    term1 = (K_train.sum() - np.trace(K_train)) / (n * (n - 1))
    term2 = (K_test.sum() - np.trace(K_test)) / (m * (m - 1))
    term3 = K_cross.sum() / (n * m)
    
    # Unbiased MMD estimator
    mmd_squared = term1 + term2 - 2.0 * term3
    
    # MMD is non-negative, numerical errors may give small negative values
    return float(np.sqrt(max(0.0, mmd_squared)))


def get_spectral_signature(K: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Extract top-k normalized eigenvalues (spectral signature)."""
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.sort(eigvals)[::-1]  # Descending
    eigvals = np.maximum(eigvals, 0)
    
    # Normalize by trace (sum of eigenvalues) to treat as probability distribution
    total_var = eigvals.sum()
    if total_var < 1e-12:
        return np.zeros(min(len(eigvals), top_k))
        
    normalized_eigs = eigvals / total_var
    return normalized_eigs[:top_k]


def spectral_divergence(sig_p: np.ndarray, sig_q: np.ndarray) -> float:
    """Compute Jensen-Shannon distance between two spectral signatures.
    
    Used to quantify how distinct an anomaly class is from normal behavior
    in the kernel feature space.
    """
    # Pad to same length if necessary
    length = max(len(sig_p), len(sig_q))
    p = np.pad(sig_p, (0, length - len(sig_p)))
    q = np.pad(sig_q, (0, length - len(sig_q)))
    
    # Ensure they sum to 1 (distributions)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    
    # JSD
    m = 0.5 * (p + q)
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    return float(np.sqrt(np.abs(jsd)))

def nystrom_landmarks(X, m, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=m, replace=False)
    return idx

def nystrom_approx(K_mm, K_nm, jitter=1e-6):
    """
    K ≈ K_nm K_mm^{-1} K_mn
    """
    U, S, _ = np.linalg.svd(K_mm + jitter * np.eye(len(K_mm)))
    S_inv = np.diag(1.0 / S)
    return K_nm @ U @ S_inv @ U.T