"""Test new v0.3.1 modules: kernel_spectrum and mmd."""
import numpy as np
import pytest


def test_kernel_spectrum_module():
    """Test kernel spectrum analysis functions."""
    from src.kernel_spectrum import (
        kernel_eigenspectrum,
        spectral_entropy,
        effective_rank,
        spectral_stats
    )
    
    # Create a simple positive definite kernel
    K = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
    ])
    
    # Test eigenspectrum
    eigvals = kernel_eigenspectrum(K)
    assert len(eigvals) == 3
    assert np.all(eigvals >= 0)
    assert eigvals[0] >= eigvals[1] >= eigvals[2]  # Descending order
    
    # Test spectral entropy
    entropy = spectral_entropy(K)
    assert entropy > 0
    assert np.isfinite(entropy)
    
    # Test effective rank
    eff_rank = effective_rank(K)
    assert eff_rank > 0
    assert eff_rank <= 3
    
    # Test spectral_stats wrapper
    stats = spectral_stats(K)
    assert 'spectral_entropy' in stats
    assert 'effective_rank' in stats
    assert 'eigenvalues' in stats
    assert len(stats['eigenvalues']) == 3


def test_mmd_module():
    """Test MMD drift detection functions."""
    from src.mmd import mmd2_unbiased, mmd
    
    # Same distribution → low MMD
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(20, 5))
    Y = rng.normal(0, 1, size=(20, 5))
    
    from sklearn.metrics.pairwise import rbf_kernel
    Kxx = rbf_kernel(X)
    Kyy = rbf_kernel(Y)
    Kxy = rbf_kernel(X, Y)
    
    mmd2_same = mmd2_unbiased(Kxx, Kyy, Kxy)
    mmd_same = mmd(Kxx, Kyy, Kxy)
    
    # Different distribution → higher MMD
    Z = rng.normal(3, 1, size=(20, 5))
    Kzz = rbf_kernel(Z)
    Kxz = rbf_kernel(X, Z)
    
    mmd2_diff = mmd2_unbiased(Kxx, Kzz, Kxz)
    mmd_diff = mmd(Kxx, Kzz, Kxz)
    
    # MMD should be non-negative
    assert mmd_same >= 0
    assert mmd_diff >= 0
    
    # Different distributions should have higher MMD
    assert mmd_diff > mmd_same


def test_quantum_kernel_spectrum_stats():
    """Test QuantumKernel.kernel_spectrum_stats() integration."""
    from src.quantum_kernel import QuantumKernel, QuantumKernelConfig
    
    cfg = QuantumKernelConfig(n_qubits=3, n_layers=1)
    qk = QuantumKernel(cfg)
    
    # Create a simple kernel matrix
    K = np.eye(5) + 0.5 * np.ones((5, 5))
    
    stats = qk.kernel_spectrum_stats(K)
    
    assert 'spectral_entropy' in stats
    assert 'effective_rank' in stats
    assert 'eigenvalues' in stats
    assert stats['spectral_entropy'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
