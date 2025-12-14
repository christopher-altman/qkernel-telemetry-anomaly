"""Kernel spectral analysis for mechanistic understanding of quantum advantage.

This module provides spectral entropy, effective rank, and eigenvalue diagnostics
to explain *why* quantum kernels work rather than just showing *that* they work.
"""
import numpy as np


def kernel_eigenspectrum(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Returns sorted eigenvalues (descending), clipped for stability.
    
    Args:
        K: Kernel matrix (n, n)
        eps: Minimum eigenvalue threshold for numerical stability
        
    Returns:
        Eigenvalues sorted in descending order
    """
    w = np.linalg.eigvalsh(K)
    w = np.clip(w, eps, None)
    return np.sort(w)[::-1]


def spectral_entropy(K: np.ndarray) -> float:
    """
    Shannon entropy of normalized kernel eigenvalues.
    
    Measures the "richness" of the kernel's feature space.
    Higher entropy = more uniform eigenvalue distribution = richer geometry.
    
    Args:
        K: Kernel matrix (n, n)
        
    Returns:
        Spectral entropy (≥0)
    """
    w = kernel_eigenspectrum(K)
    p = w / w.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def effective_rank(K: np.ndarray) -> float:
    """
    exp(spectral entropy); interpretable dimensionality of RKHS.
    
    Provides an interpretable measure of kernel expressivity:
    - Effective rank ≈ n: Kernel uses full feature space
    - Effective rank << n: Kernel concentrates on low-dimensional subspace
    
    Args:
        K: Kernel matrix (n, n)
        
    Returns:
        Effective rank
    """
    return float(np.exp(spectral_entropy(K)))


def spectral_stats(K: np.ndarray) -> dict:
    """
    Convenience wrapper for all spectral diagnostics.
    
    Args:
        K: Kernel matrix (n, n)
        
    Returns:
        Dictionary with:
            - spectral_entropy: Shannon entropy of eigenvalues
            - effective_rank: exp(spectral_entropy)
            - eigenvalues: Full eigenspectrum (descending)
    """
    w = kernel_eigenspectrum(K)
    return {
        "spectral_entropy": spectral_entropy(K),
        "effective_rank": effective_rank(K),
        "eigenvalues": w,
    }
