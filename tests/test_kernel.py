"""Comprehensive tests for quantum kernel module."""
import numpy as np
import pytest

from src.quantum_kernel import (
    QuantumKernel,
    QuantumKernelConfig,
    center_kernel_matrix,
)


def test_kernel_symmetry_and_diagonal():
    """Test kernel matrix is symmetric with valid diagonal."""
    cfg = QuantumKernelConfig(n_qubits=4, n_layers=1, gamma=1.0)
    qk = QuantumKernel(cfg)

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2*np.pi, size=(6, cfg.n_qubits))

    K = qk.kernel_matrix(X, show_progress=False)

    assert K.shape == (6, 6), "Kernel matrix has wrong shape"
    assert np.allclose(K, K.T, atol=1e-6), "Kernel matrix not symmetric"

    # Fidelity kernel should have diagonal near 1
    assert np.all(K.diagonal() > 0.90), "Diagonal elements too small"
    assert np.all(K.diagonal() <= 1.000001), "Diagonal elements exceed 1"


def test_kernel_positive_semidefinite():
    """Test kernel matrix is positive semi-definite."""
    cfg = QuantumKernelConfig(n_qubits=3, n_layers=2, gamma=0.5)
    qk = QuantumKernel(cfg)

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 2*np.pi, size=(8, cfg.n_qubits))

    K = qk.kernel_matrix(X, show_progress=False)
    
    # Check all eigenvalues are non-negative
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-6), f"Kernel has negative eigenvalues: {eigvals.min()}"


def test_rectangular_kernel_matrix():
    """Test computing K(X1, X2) for different X1, X2."""
    cfg = QuantumKernelConfig(n_qubits=4, n_layers=1)
    qk = QuantumKernel(cfg)

    rng = np.random.default_rng(123)
    X1 = rng.uniform(0, 2*np.pi, size=(5, cfg.n_qubits))
    X2 = rng.uniform(0, 2*np.pi, size=(7, cfg.n_qubits))

    K = qk.kernel_matrix(X1, X2, show_progress=False)
    
    assert K.shape == (5, 7), "Rectangular kernel has wrong shape"
    assert np.all(K >= 0) and np.all(K <= 1), "Kernel values out of [0,1] range"


def test_different_feature_maps():
    """Test all supported feature maps execute without errors."""
    feature_maps = ['zz', 'pauli', 'iqp']
    
    rng = np.random.default_rng(99)
    X = rng.uniform(0, 2*np.pi, size=(4, 4))
    
    for fm in feature_maps:
        cfg = QuantumKernelConfig(n_qubits=4, n_layers=1, feature_map=fm)
        qk = QuantumKernel(cfg)
        
        K = qk.kernel_matrix(X, show_progress=False)
        
        assert K.shape == (4, 4), f"Feature map '{fm}' produced wrong shape"
        assert np.allclose(K, K.T, atol=1e-6), f"Feature map '{fm}' not symmetric"


def test_noise_models():
    """Test noise models execute without errors."""
    noise_models = ['none', 'depolarizing', 'amplitude_damping']
    
    rng = np.random.default_rng(777)
    X = rng.uniform(0, 2*np.pi, size=(3, 4))
    
    for noise in noise_models:
        cfg = QuantumKernelConfig(
            n_qubits=4, 
            n_layers=1, 
            noise_model=noise,
            noise_strength=0.05
        )
        qk = QuantumKernel(cfg)
        
        K = qk.kernel_matrix(X, show_progress=False)
        
        assert K.shape == (3, 3), f"Noise model '{noise}' produced wrong shape"


def test_kernel_centering():
    """Test kernel centering produces zero-mean kernel."""
    K = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
    ])
    
    K_centered = center_kernel_matrix(K)
    
    # Row and column sums should be approximately zero
    row_sums = K_centered.sum(axis=1)
    col_sums = K_centered.sum(axis=0)
    
    assert np.allclose(row_sums, 0, atol=1e-10), "Centered kernel rows don't sum to zero"
    assert np.allclose(col_sums, 0, atol=1e-10), "Centered kernel cols don't sum to zero"


def test_kernel_alignment():
    """Test kernel alignment metric."""
    cfg = QuantumKernelConfig(n_qubits=3, n_layers=1)
    qk = QuantumKernel(cfg)
    
    rng = np.random.default_rng(555)
    X = rng.uniform(0, 2*np.pi, size=(5, cfg.n_qubits))
    
    K1 = qk.kernel_matrix(X, show_progress=False)
    K2 = qk.kernel_matrix(X, show_progress=False)  # Same kernel
    
    # Self-alignment should be 1
    alignment = qk.kernel_alignment(K1, K2)
    assert np.isclose(alignment, 1.0, atol=1e-6), f"Self-alignment not 1: {alignment}"
    
    # Alignment with random matrix should be lower
    K_random = rng.uniform(0, 1, size=K1.shape)
    K_random = (K_random + K_random.T) / 2  # Make symmetric
    
    alignment_random = qk.kernel_alignment(K1, K_random)
    assert 0 <= alignment_random <= 1, f"Alignment out of range: {alignment_random}"


def test_effective_dimension():
    """Test effective dimension computation."""
    cfg = QuantumKernelConfig(n_qubits=4, n_layers=2)
    qk = QuantumKernel(cfg)
    
    rng = np.random.default_rng(888)
    X = rng.uniform(0, 2*np.pi, size=(10, cfg.n_qubits))
    
    K = qk.kernel_matrix(X, show_progress=False)
    eff_dim = qk.effective_dimension(K, threshold=0.99)
    
    assert 1 <= eff_dim <= 10, f"Effective dimension out of range: {eff_dim}"


def test_config_validation():
    """Test configuration validation."""
    # Invalid n_qubits
    with pytest.raises(ValueError, match="n_qubits must be positive"):
        QuantumKernel(QuantumKernelConfig(n_qubits=0))
    
    # Invalid n_layers
    with pytest.raises(ValueError, match="n_layers must be positive"):
        QuantumKernel(QuantumKernelConfig(n_layers=0))
    
    # Invalid noise_strength
    with pytest.raises(ValueError, match="noise_strength must be in"):
        QuantumKernel(QuantumKernelConfig(noise_strength=1.5))


def test_dimension_mismatch():
    """Test error handling for dimension mismatches."""
    cfg = QuantumKernelConfig(n_qubits=4)
    qk = QuantumKernel(cfg)
    
    # Wrong feature dimension
    X_wrong = np.random.randn(5, 6)  # 6 features, need 4
    
    with pytest.raises(ValueError, match="doesn't match n_qubits"):
        qk.kernel_matrix(X_wrong, show_progress=False)


def test_reproducibility():
    """Test kernel computation is deterministic."""
    cfg = QuantumKernelConfig(n_qubits=3, n_layers=1, gamma=0.8)
    
    rng = np.random.default_rng(12345)
    X = rng.uniform(0, 2*np.pi, size=(4, cfg.n_qubits))
    
    qk1 = QuantumKernel(cfg)
    K1 = qk1.kernel_matrix(X, show_progress=False)
    
    qk2 = QuantumKernel(cfg)
    K2 = qk2.kernel_matrix(X, show_progress=False)
    
    assert np.allclose(K1, K2, atol=1e-10), "Kernel computation not reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
