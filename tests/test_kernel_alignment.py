"""Tests for kernel alignment optimization module."""
import numpy as np
import pytest


def test_center_kernel():
    """Test kernel centering."""
    from src.kernel_alignment import center_kernel
    
    # Simple kernel
    K = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
    ])
    
    K_c = center_kernel(K)
    
    # Centered kernel should have zero row/column means
    assert np.allclose(K_c.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(K_c.mean(axis=1), 0.0, atol=1e-10)


def test_kta_score():
    """Test KTA computation."""
    from src.kernel_alignment import kta_score
    
    # Perfect alignment: K matches label structure
    y = np.array([0, 0, 1, 1])
    
    # Kernel that separates classes perfectly
    K = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.9],
        [0.1, 0.1, 0.9, 1.0]
    ])
    
    kta = kta_score(K, y)
    
    # Should have high alignment
    assert kta > 0.5
    assert -1.0 <= kta <= 1.0


def test_finite_diff_grad():
    """Test finite difference gradient computation."""
    from src.kernel_alignment import finite_diff_grad
    from sklearn.metrics.pairwise import rbf_kernel
    
    # Simple kernel builder
    def kernel_builder(params, X):
        gamma = params[0]
        return rbf_kernel(X, gamma=gamma)
    
    # Data
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(20, 5))
    y = np.array([0]*10 + [1]*10)
    
    # Initial params
    params = np.array([0.5])
    
    # Compute gradient
    grad, score = finite_diff_grad(kernel_builder, params, X, y)
    
    assert grad.shape == params.shape
    assert np.isfinite(score)
    assert -1.0 <= score <= 1.0


def test_align_optimize():
    """Test alignment optimization."""
    from src.kernel_alignment import align_optimize
    from sklearn.metrics.pairwise import rbf_kernel
    
    # Simple kernel builder
    def kernel_builder(params, X):
        gamma = params[0]
        return rbf_kernel(X, gamma=gamma)
    
    # Separable data
    rng = np.random.default_rng(42)
    X1 = rng.normal(-1, 0.5, size=(15, 5))
    X2 = rng.normal(1, 0.5, size=(15, 5))
    X = np.vstack([X1, X2])
    y = np.array([0]*15 + [1]*15)
    
    # Initial params
    init_params = np.array([0.1])
    
    # Optimize
    opt_params, history = align_optimize(
        kernel_builder,
        init_params,
        X, y,
        iters=10,
        lr=0.1,
        batch=30,
        verbose=False,
        param_bounds=(1e-6, np.inf)  # Ensure gamma stays positive
    )
    
    # Should improve
    assert len(history) == 10
    assert history[-1] > history[0]  # KTA should increase
    assert opt_params[0] != init_params[0]  # Params should change


def test_unsupervised_alignment():
    """Test unsupervised drift alignment."""
    from src.kernel_alignment import unsupervised_alignment_drift
    from sklearn.metrics.pairwise import rbf_kernel
    
    # Kernel builder
    def kernel_builder(params, X):
        gamma = params[0]
        return rbf_kernel(X, gamma=gamma)
    
    # Early vs late windows (different distributions)
    rng = np.random.default_rng(42)
    X_early = rng.normal(0, 1, size=(20, 5))
    X_late = rng.normal(2, 1, size=(20, 5))  # Shifted distribution
    
    # Initial params
    init_params = np.array([0.2])
    
    # Optimize
    opt_params, history = unsupervised_alignment_drift(
        kernel_builder,
        init_params,
        X_early,
        X_late,
        iters=10,
        lr=0.1,
        verbose=False,
        param_bounds=(1e-6, np.inf)  # Ensure gamma stays positive
    )
    
    # Should run optimization
    assert len(history) == 10
    # Score should be reasonable (not guaranteed to improve with fixed lr)
    assert all(-1.0 <= s <= 1.0 for s in history)
    # Parameters should change
    assert not np.allclose(opt_params, init_params)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
