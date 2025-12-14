"""Maximum Mean Discrepancy (MMD) for distribution shift detection.

MMD measures the distance between probability distributions in a reproducing kernel
Hilbert space (RKHS). Used for detecting slow-onset drift in telemetry distributions.
"""
import numpy as np


def mmd2_unbiased(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    """
    Unbiased MMD² estimator for two-sample test.
    
    Tests whether samples X and Y come from the same distribution using
    kernel embeddings in RKHS.
    
    Formula:
        MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    
    where the expectations are estimated using unbiased U-statistics.
    
    Args:
        Kxx: Kernel matrix on first sample (n, n)
        Kyy: Kernel matrix on second sample (m, m)
        Kxy: Cross-kernel matrix between samples (n, m)
        
    Returns:
        Unbiased MMD² estimate (can be negative due to sampling variance)
    """
    n = Kxx.shape[0]
    m = Kyy.shape[0]

    # Unbiased estimators (exclude diagonal to avoid bias)
    term_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) if n > 1 else 0.0
    term_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) if m > 1 else 0.0
    term_xy = 2.0 * Kxy.mean()

    return float(term_xx + term_yy - term_xy)


def mmd(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    """
    MMD (square root of unbiased MMD²).
    
    Always non-negative. If MMD² < 0 due to sampling variance, returns 0.
    
    Args:
        Kxx: Kernel matrix on first sample (n, n)
        Kyy: Kernel matrix on second sample (m, m)
        Kxy: Cross-kernel matrix between samples (n, m)
        
    Returns:
        MMD distance (≥0)
    """
    mmd2 = mmd2_unbiased(Kxx, Kyy, Kxy)
    return float(np.sqrt(max(0.0, mmd2)))
