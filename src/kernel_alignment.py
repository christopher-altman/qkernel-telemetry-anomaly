"""Kernel alignment optimization for quantum feature map learning.

This module implements Kernel Target Alignment (KTA) optimization to learn
quantum feature map parameters that maximize alignment with the target task.

KTA is a clean separation of:
1. Feature map learning (via alignment)
2. Classifier training (SVM/mean-embedding)

This transforms "we picked a feature map" into "we fit the feature map to the task."
"""
from typing import Optional
import numpy as np


def center_kernel(K: np.ndarray) -> np.ndarray:
    """Center kernel matrix: K_c = (I - 1/n*11^T) K (I - 1/n*11^T).
    
    Centering is critical for proper KTA computation.
    
    Args:
        K: Kernel matrix (n, n)
        
    Returns:
        K_centered: Centered kernel matrix
    """
    n = K.shape[0]
    one = np.ones((n, n)) / n
    return K - one @ K - K @ one + one @ K @ one


def kta_score(K: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Kernel Target Alignment with centering.
    
    KTA(K, Y) = <K_c, Y_c>_F / (||K_c||_F * ||Y_c||_F)
    
    where:
    - K is the kernel matrix
    - Y = yy^T is the "ideal" label kernel
    - subscript c denotes centered
    
    High KTA indicates the kernel geometry aligns well with the task structure.
    
    Args:
        K: Kernel matrix (n, n)
        y: Binary labels {0,1} or {-1,+1} (n,)
        eps: Numerical stability constant
        
    Returns:
        KTA score in [-1, 1], higher is better
    """
    y = np.asarray(y).astype(float)
    
    # Convert {0,1} to {-1,+1} if needed
    if set(np.unique(y)).issubset({0.0, 1.0}):
        y = 2.0 * y - 1.0
    
    # Ideal label kernel
    Y = np.outer(y, y)
    
    # Center both kernels
    Kc = center_kernel(K)
    Yc = center_kernel(Y)
    
    # Frobenius inner product and norms
    num = np.sum(Kc * Yc)
    den = np.linalg.norm(Kc, "fro") * np.linalg.norm(Yc, "fro") + eps
    
    return float(num / den)


def finite_diff_grad(
    kernel_builder,
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 1e-3,
    max_params: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Finite-difference gradient of KTA w.r.t. parameters.
    
    Works with any quantum backend, no autograd required.
    For large parameter spaces, randomly samples max_params for efficiency.
    
    Args:
        kernel_builder: Function params, X -> K(X, X)
        params: Current parameters (any shape)
        X: Training data (n, d)
        y: Training labels (n,)
        delta: Finite difference step size
        max_params: Maximum parameters to perturb (for efficiency)
        seed: Random seed for parameter sampling
        
    Returns:
        gradient: Same shape as params
        base_score: KTA at current params
    """
    rng = np.random.default_rng(seed)
    flat = params.reshape(-1)
    d = min(len(flat), max_params)
    
    # Sample subset of parameters for large spaces
    idx = rng.choice(len(flat), size=d, replace=False) if d < len(flat) else np.arange(len(flat))
    
    # Base KTA
    baseK = kernel_builder(params, X)
    base = kta_score(baseK, y)
    
    # Compute gradient via finite differences
    g = np.zeros_like(flat)
    for j in idx:
        old = flat[j]
        flat[j] = old + delta
        
        # Perturbed kernel
        Kp = kernel_builder(params, X)
        sp = kta_score(Kp, y)
        
        # Gradient estimate
        g[j] = (sp - base) / delta
        flat[j] = old
    
    return g.reshape(params.shape), base


def align_optimize(
    kernel_builder,
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    iters: int = 25,
    lr: float = 0.15,
    batch: int = 64,
    seed: int = 0,
    verbose: bool = True,
    param_bounds: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, list[float]]:
    """Optimize kernel parameters via gradient ascent on KTA.
    
    Simple but effective optimization that works with any backend.
    Uses mini-batching for efficiency on large datasets.
    
    Args:
        kernel_builder: Function params, X -> K(X, X)
        params: Initial parameters
        X: Training data (n, d)
        y: Training labels (n,)
        iters: Number of optimization steps
        lr: Learning rate (step size)
        batch: Mini-batch size (0 = full batch)
        seed: Random seed
        verbose: Print progress
        param_bounds: Optional (min, max) bounds for parameters (e.g. (0.0, inf) for non-negative)
        
    Returns:
        optimized_params: Final parameters
        history: KTA score at each iteration
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    
    history = []
    
    if verbose:
        print(f"  Starting KTA optimization: {iters} iters, lr={lr}, batch={batch if batch else 'full'}")
    
    for t in range(iters):
        # Mini-batch sampling
        if batch and batch < n:
            idx = rng.choice(n, size=batch, replace=False)
            Xb = X[idx]
            yb = y[idx]
        else:
            Xb = X
            yb = y
        
        # Compute gradient and score
        g, score = finite_diff_grad(kernel_builder, params, Xb, yb, seed=seed + t)
        
        # Gradient ascent step
        params = params + lr * g
        
        # Apply parameter bounds if specified
        if param_bounds is not None:
            params = np.clip(params, param_bounds[0], param_bounds[1])
        
        history.append(score)
        
        if verbose and (t == 0 or (t + 1) % 5 == 0 or t == iters - 1):
            print(f"    Iter {t+1:3d}: KTA = {score:.4f}")
    
    return params, history


def unsupervised_alignment_drift(
    kernel_builder,
    params: np.ndarray,
    X_early: np.ndarray,
    X_late: np.ndarray,
    iters: int = 25,
    lr: float = 0.15,
    seed: int = 0,
    verbose: bool = True,
    param_bounds: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, list[float]]:
    """Unsupervised kernel alignment to maximize drift detection.
    
    Optimizes kernel to maximize separation between early and late windows.
    Useful for label-free feature map learning on temporal data.
    
    Args:
        kernel_builder: Function params, X -> K(X, X)
        params: Initial parameters
        X_early: Early window data (n, d)
        X_late: Late window data (m, d)
        iters: Number of optimization steps
        lr: Learning rate
        seed: Random seed
        verbose: Print progress
        param_bounds: Optional (min, max) bounds for parameters (e.g. (0.0, inf) for non-negative)
        
    Returns:
        optimized_params: Final parameters
        history: Separation scores at each iteration
    """
    rng = np.random.default_rng(seed)
    
    # Create pseudo-labels: early=0, late=1
    X = np.vstack([X_early, X_late])
    y = np.concatenate([np.zeros(len(X_early)), np.ones(len(X_late))])
    
    history = []
    
    if verbose:
        print(f"  Unsupervised drift alignment: {iters} iters, lr={lr}")
    
    for t in range(iters):
        # Compute gradient treating early/late as pseudo-labels
        g, score = finite_diff_grad(kernel_builder, params, X, y, seed=seed + t)
        
        # Gradient ascent
        params = params + lr * g
        
        # Apply parameter bounds if specified
        if param_bounds is not None:
            params = np.clip(params, param_bounds[0], param_bounds[1])
        
        history.append(score)
        
        if verbose and (t == 0 or (t + 1) % 5 == 0 or t == iters - 1):
            print(f"    Iter {t+1:3d}: Drift Separation = {score:.4f}")
    
    return params, history
