"""Tests for new utility functions (dimension matching, Nyström, mean embedding, MMD)."""
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.utils import (
    match_feature_dim,
    nystrom_factor,
    nystrom_approx,
    kernel_mean_embedding_scores,
    mmd_drift_score,
)
from src.quantum_kernel import QuantumKernel, QuantumKernelConfig


def test_match_feature_dim_pad():
    """Test padding mode for dimension matching."""
    X = np.random.randn(10, 5)
    
    # Pad to larger dimension
    X_padded = match_feature_dim(X, target_dim=8, mode="pad")
    assert X_padded.shape == (10, 8)
    assert np.allclose(X_padded[:, :5], X)
    assert np.allclose(X_padded[:, 5:], 0)


def test_match_feature_dim_tile():
    """Test tiling mode for dimension matching."""
    X = np.random.randn(10, 3)
    
    # Tile to larger dimension
    X_tiled = match_feature_dim(X, target_dim=8, mode="tile")
    assert X_tiled.shape == (10, 8)
    # First 3 should match original
    assert np.allclose(X_tiled[:, :3], X)
    # Next 3 should be a copy
    assert np.allclose(X_tiled[:, 3:6], X)


def test_match_feature_dim_truncate():
    """Test truncate mode for dimension matching."""
    X = np.random.randn(10, 10)
    
    # Truncate to smaller dimension
    X_trunc = match_feature_dim(X, target_dim=5, mode="truncate")
    assert X_trunc.shape == (10, 5)
    assert np.allclose(X_trunc, X[:, :5])
    
    # Should error if trying to truncate smaller dimension
    X_small = np.random.randn(10, 3)
    with pytest.raises(ValueError, match="Cannot truncate"):
        match_feature_dim(X_small, target_dim=5, mode="truncate")


def test_nystrom_approximation():
    """Test Nyström kernel approximation."""
    # Create a simple kernel matrix
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(50, 5))
    
    # RBF kernel
    from sklearn.metrics.pairwise import rbf_kernel
    K_full = rbf_kernel(X, gamma=0.5)
    
    # Nyström approximation
    m = 20
    landmarks = X[:m]
    K_mm = rbf_kernel(landmarks, gamma=0.5)
    K_nm = rbf_kernel(X, landmarks, gamma=0.5)
    
    K_approx = nystrom_approx(K_nm, K_mm, jitter=1e-6)
    
    assert K_approx.shape == K_full.shape
    assert np.allclose(K_approx, K_approx.T, atol=1e-6)  # Symmetric
    
    # Approximation should be reasonably close
    error = np.linalg.norm(K_full - K_approx, 'fro') / np.linalg.norm(K_full, 'fro')
    assert error < 0.5  # Relative error less than 50%


def test_kernel_mean_embedding_scores():
    """Test kernel mean embedding anomaly scoring."""
    rng = np.random.default_rng(123)
    
    # Create simple kernel matrices
    n_train = 20
    n_test = 10
    
    K_train = np.eye(n_train) + 0.5 * np.ones((n_train, n_train))
    K_test = 0.6 * np.ones((n_test, n_train))
    K_diag_test = np.ones(n_test)
    
    scores = kernel_mean_embedding_scores(K_train, K_test, K_diag_test)
    
    assert scores.shape == (n_test,)
    assert np.all(np.isfinite(scores))


def test_mmd_drift_score():
    """Test MMD drift score computation."""
    rng = np.random.default_rng(456)
    
    # Same distribution → low MMD
    X1 = rng.normal(0, 1, size=(30, 5))
    X2 = rng.normal(0, 1, size=(30, 5))
    
    from sklearn.metrics.pairwise import rbf_kernel
    K_11 = rbf_kernel(X1, gamma=0.5)
    K_22 = rbf_kernel(X2, gamma=0.5)
    K_12 = rbf_kernel(X1, X2, gamma=0.5)
    
    mmd_same = mmd_drift_score(K_11, K_22, K_12)
    
    # Different distribution → higher MMD
    X3 = rng.normal(2, 1, size=(30, 5))
    K_33 = rbf_kernel(X3, gamma=0.5)
    K_13 = rbf_kernel(X1, X3, gamma=0.5)
    
    mmd_diff = mmd_drift_score(K_11, K_33, K_13)
    
    assert mmd_same >= 0
    assert mmd_diff >= 0
    assert mmd_diff > mmd_same  # Different distributions should have higher MMD


def test_quantum_kernel_diag():
    """Test diagonal kernel computation."""
    cfg = QuantumKernelConfig(n_qubits=3, n_layers=1)
    qk = QuantumKernel(cfg)
    
    rng = np.random.default_rng(789)
    X = rng.uniform(0, 2*np.pi, size=(5, 3))
    
    # Compute diagonal
    diag = qk.kernel_diag(X, show_progress=False)
    
    assert diag.shape == (5,)
    assert np.all(diag > 0.9)  # Fidelity kernel diagonal should be near 1
    assert np.all(diag <= 1.000001)


def test_spectral_entropy():
    """Test spectral entropy computation."""
    cfg = QuantumKernelConfig(n_qubits=3, n_layers=1)
    qk = QuantumKernel(cfg)
    
    # Create a simple kernel matrix
    K = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
    ])
    
    entropy = qk.spectral_entropy(K)
    
    assert entropy > 0
    assert np.isfinite(entropy)


def test_effective_rank():
    """Test effective rank computation."""
    cfg = QuantumKernelConfig(n_qubits=3, n_layers=1)
    qk = QuantumKernel(cfg)
    
    # Identity matrix should have high effective rank
    K_full = np.eye(5)
    eff_rank_full = qk.effective_rank(K_full)
    assert eff_rank_full > 4  # Close to 5
    
    # Rank-1 matrix should have low effective rank
    v = np.ones((5, 1))
    K_rank1 = v @ v.T
    eff_rank_low = qk.effective_rank(K_rank1)
    assert eff_rank_low < 2  # Close to 1


def test_integration_with_nystrom():
    """Test end-to-end with Nyström approximation."""
    from src.telemetry_sim import make_dataset, TelemetryConfig
    from src.utils import window_features_mean_std, minmax_to_0_2pi
    
    # Small dataset
    cfg = TelemetryConfig(timesteps=16)
    X, y, _ = make_dataset(30, 0.2, cfg, seed=999)
    
    # Extract and scale features
    F = window_features_mean_std(X)
    xmin, xmax = F.min(axis=0), F.max(axis=0)
    F_scaled = minmax_to_0_2pi(F, xmin, xmax)
    
    # Match to n_qubits
    F_matched = match_feature_dim(F_scaled, target_dim=4, mode="tile")
    
    # Quantum kernel with Nyström
    qcfg = QuantumKernelConfig(n_qubits=4, n_layers=1)
    qk = QuantumKernel(qcfg)
    
    m = 10
    landmarks = F_matched[:m]
    
    K_mm = qk.kernel_matrix(landmarks, show_progress=False)
    K_nm = qk.kernel_matrix(F_matched, landmarks, show_progress=False)
    
    K_approx = nystrom_approx(K_nm, K_mm)
    
    assert K_approx.shape == (30, 30)
    assert np.allclose(K_approx, K_approx.T, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
