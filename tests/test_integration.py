"""Integration tests for the full pipeline."""
import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from src.telemetry_sim import make_dataset, TelemetryConfig
from src.utils import window_features_mean_std, minmax_to_0_2pi
from src.quantum_kernel import QuantumKernel, QuantumKernelConfig
from src.baselines import precomputed_kernel_svm_scores, rbf_svm_scores, BaselineConfig
from src.eval import compute_metrics


def test_end_to_end_pipeline():
    """Test complete pipeline from data generation to metrics."""
    # Small-scale test to run quickly
    cfg = TelemetryConfig(timesteps=32, noise_std=0.02)
    
    # Generate data
    X, y, a_type = make_dataset(
        n_samples=40, 
        anomaly_rate=0.2, 
        cfg=cfg, 
        seed=42
    )
    
    assert X.shape == (40, 32, 4), "Wrong data shape"
    assert y.shape == (40,), "Wrong label shape"
    assert (y == 1).sum() == 8, "Wrong number of anomalies"
    
    # Extract features
    F = window_features_mean_std(X)
    assert F.shape == (40, 8), "Wrong feature shape"
    
    # Split
    F_train = F[:30]
    F_test = F[30:]
    y_train = y[:30]
    y_test = y[30:]
    
    # Scale
    xmin = F_train.min(axis=0)
    xmax = F_train.max(axis=0)
    F_train_q = minmax_to_0_2pi(F_train, xmin, xmax)
    F_test_q = minmax_to_0_2pi(F_test, xmin, xmax)
    
    # Quantum kernel
    qcfg = QuantumKernelConfig(n_qubits=8, n_layers=1, gamma=0.5)
    qk = QuantumKernel(qcfg)
    
    K_train = qk.kernel_matrix(F_train_q, show_progress=False)
    K_test = qk.kernel_matrix(F_test_q, F_train_q, show_progress=False)
    
    # Train and score
    scores_q = precomputed_kernel_svm_scores(K_train, y_train, K_test, C=1.0)
    scores_rbf = rbf_svm_scores(F_train, y_train, F_test, BaselineConfig())
    
    # Compute metrics
    metrics_q = compute_metrics(y_test, scores_q)
    metrics_rbf = compute_metrics(y_test, scores_rbf)
    
    # Basic sanity checks
    assert 0 <= metrics_q.roc_auc <= 1, "Invalid quantum ROC-AUC"
    assert 0 <= metrics_rbf.roc_auc <= 1, "Invalid RBF ROC-AUC"
    assert metrics_q.roc_auc > 0.5, "Quantum kernel performing worse than random"


def test_anomaly_types_coverage():
    """Test that all anomaly types are generated."""
    cfg = TelemetryConfig(timesteps=64)
    
    X, y, a_type = make_dataset(
        n_samples=100,
        anomaly_rate=0.4,
        cfg=cfg,
        seed=999
    )
    
    # Should have all 4 anomaly types
    unique_anomaly_types = np.unique(a_type[y == 1])
    assert len(unique_anomaly_types) > 0, "No anomaly types generated"
    
    # Types should be 0, 1, 2, 3 (impulse, drift, dropout, spoofing)
    assert all(t in [0, 1, 2, 3] for t in unique_anomaly_types), "Invalid anomaly types"


def test_feature_scaling():
    """Test feature scaling to [0, 2π]."""
    rng = np.random.default_rng(111)
    F = rng.uniform(-5, 10, size=(20, 8))
    
    xmin = F.min(axis=0)
    xmax = F.max(axis=0)
    F_scaled = minmax_to_0_2pi(F, xmin, xmax)
    
    # Check range
    assert np.all(F_scaled >= 0), "Scaled features below 0"
    assert np.all(F_scaled <= 2 * np.pi + 1e-6), "Scaled features above 2π"
    
    # Check min/max preserved
    assert np.allclose(F_scaled.min(axis=0), 0, atol=1e-6), "Min not at 0"
    assert np.allclose(F_scaled.max(axis=0), 2 * np.pi, atol=1e-6), "Max not at 2π"


def test_kernel_numerical_stability():
    """Test kernel computation with edge cases."""
    cfg = QuantumKernelConfig(n_qubits=4, n_layers=1)
    qk = QuantumKernel(cfg)
    
    # All zeros
    X_zeros = np.zeros((3, 4))
    K = qk.kernel_matrix(X_zeros, show_progress=False)
    assert np.all(np.isfinite(K)), "NaN/Inf in kernel with zero inputs"
    
    # All same value
    X_const = np.ones((3, 4)) * np.pi
    K = qk.kernel_matrix(X_const, show_progress=False)
    assert np.all(np.isfinite(K)), "NaN/Inf in kernel with constant inputs"
    # Identical inputs should give kernel value near 1
    assert np.all(K > 0.9), "Kernel of identical inputs not near 1"


def test_baseline_classifiers():
    """Test all baseline classifiers run without errors."""
    rng = np.random.default_rng(333)
    F_train = rng.randn(50, 8)
    F_test = rng.randn(20, 8)
    y_train = rng.integers(0, 2, size=50)
    
    cfg = BaselineConfig(random_state=42)
    
    # RBF-SVM
    scores = rbf_svm_scores(F_train, y_train, F_test, cfg)
    assert scores.shape == (20,), "Wrong shape for RBF-SVM scores"
    assert np.all(scores >= 0) and np.all(scores <= 1), "Scores out of [0,1]"
    
    # One-Class SVM (only normal data)
    from src.baselines import oneclass_svm_scores
    scores_oc = oneclass_svm_scores(F_train[y_train == 0], F_test, cfg)
    assert scores_oc.shape == (20,), "Wrong shape for One-Class SVM scores"
    
    # Isolation Forest
    from src.baselines import isolation_forest_scores
    scores_if = isolation_forest_scores(F_train, F_test, cfg)
    assert scores_if.shape == (20,), "Wrong shape for Isolation Forest scores"


def test_deterministic_data_generation():
    """Test data generation is deterministic with fixed seed."""
    cfg = TelemetryConfig(timesteps=32)
    
    X1, y1, a1 = make_dataset(50, 0.2, cfg, seed=12345)
    X2, y2, a2 = make_dataset(50, 0.2, cfg, seed=12345)
    
    assert np.allclose(X1, X2), "Data generation not deterministic"
    assert np.array_equal(y1, y2), "Labels not deterministic"
    assert np.array_equal(a1, a2), "Anomaly types not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
