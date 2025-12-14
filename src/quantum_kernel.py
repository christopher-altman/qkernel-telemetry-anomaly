"""Enhanced quantum kernel with multiple feature maps, noise models, and optimization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pennylane as qml
from tqdm import tqdm


@dataclass
class QuantumKernelConfig:
    """Configuration for quantum kernel computation.
    
    Args:
        n_qubits: Number of qubits (must match feature dimension)
        n_layers: Number of data-reuploading layers
        gamma: ZZ interaction coupling strength
        feature_map: Type of encoding ('zz', 'pauli', 'iqp')
        noise_model: Hardware noise simulation ('none', 'depolarizing', 'amplitude_damping')
        noise_strength: Noise parameter (p for depolarizing, γ for amplitude damping)
        centered: Whether to center the kernel matrix
    """
    n_qubits: int = 8
    n_layers: int = 2
    gamma: float = 1.0
    feature_map: Literal['zz', 'pauli', 'iqp'] = 'zz'
    noise_model: Literal['none', 'depolarizing', 'amplitude_damping'] = 'none'
    noise_strength: float = 0.01
    centered: bool = False


def _apply_noise_channel(noise_model: str, noise_strength: float, wires: int) -> None:
    """Apply noise channel after gates to simulate hardware imperfections."""
    if noise_model == 'depolarizing':
        qml.DepolarizingChannel(noise_strength, wires=wires)
    elif noise_model == 'amplitude_damping':
        qml.AmplitudeDamping(noise_strength, wires=wires)


def _zz_feature_map(x: np.ndarray, cfg: QuantumKernelConfig) -> None:
    """ZZ-entangling feature map (similar to Qiskit's ZZFeatureMap).
    
    Uses data reuploading with ZZ interactions encoding feature correlations.
    """
    n = cfg.n_qubits
    x = np.asarray(x, dtype=float)
    if x.shape[0] != n:
        raise ValueError(f"Feature dimension {x.shape[0]} doesn't match n_qubits {n}")

    for layer in range(cfg.n_layers):
        # Hadamard layer for superposition
        if layer == 0:
            for i in range(n):
                qml.Hadamard(wires=i)
        
        # Data rotations (angle encoding with reuploading)
        for i in range(n):
            qml.RZ(2.0 * x[i], wires=i)
            if cfg.noise_model != 'none':
                _apply_noise_channel(cfg.noise_model, cfg.noise_strength, i)
        
        # Entangling ZZ interactions (encode pairwise feature products)
        for i in range(n):
            for j in range(i + 1, n):
                angle = cfg.gamma * (np.pi - x[i]) * (np.pi - x[j])
                qml.IsingZZ(angle, wires=[i, j])
                if cfg.noise_model != 'none':
                    _apply_noise_channel(cfg.noise_model, cfg.noise_strength, i)
                    _apply_noise_channel(cfg.noise_model, cfg.noise_strength, j)


def _pauli_feature_map(x: np.ndarray, cfg: QuantumKernelConfig) -> None:
    """Pauli rotation feature map with full connectivity.
    
    Applies RX, RY, RZ rotations followed by CNOT entanglement.
    """
    n = cfg.n_qubits
    x = np.asarray(x, dtype=float)
    if x.shape[0] != n:
        raise ValueError(f"Feature dimension {x.shape[0]} doesn't match n_qubits {n}")

    for layer in range(cfg.n_layers):
        # Triple Pauli rotations
        for i in range(n):
            qml.RX(x[i], wires=i)
            qml.RY(x[i], wires=i)
            qml.RZ(x[i], wires=i)
            if cfg.noise_model != 'none':
                _apply_noise_channel(cfg.noise_model, cfg.noise_strength, i)
        
        # Entangling layer (ring topology)
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
            if cfg.noise_model != 'none':
                _apply_noise_channel(cfg.noise_model, cfg.noise_strength, i)
                _apply_noise_channel(cfg.noise_model, cfg.noise_strength, i + 1)
        qml.CNOT(wires=[n - 1, 0])


def _iqp_feature_map(x: np.ndarray, cfg: QuantumKernelConfig) -> None:
    """IQP-inspired feature map (Instantaneous Quantum Polynomial).
    
    Known to be hard to simulate classically for certain problem structures.
    """
    n = cfg.n_qubits
    x = np.asarray(x, dtype=float)
    if x.shape[0] != n:
        raise ValueError(f"Feature dimension {x.shape[0]} doesn't match n_qubits {n}")

    for layer in range(cfg.n_layers):
        # Hadamard layer
        for i in range(n):
            qml.Hadamard(wires=i)
        
        # Diagonal encoding
        for i in range(n):
            qml.RZ(x[i], wires=i)
            if cfg.noise_model != 'none':
                _apply_noise_channel(cfg.noise_model, cfg.noise_strength, i)
        
        # Entangling diagonal gates (ZZ interactions)
        for i in range(n):
            for j in range(i + 1, n):
                qml.IsingZZ(x[i] * x[j], wires=[i, j])
                if cfg.noise_model != 'none':
                    _apply_noise_channel(cfg.noise_model, cfg.noise_strength, i)
                    _apply_noise_channel(cfg.noise_model, cfg.noise_strength, j)


def _get_feature_map(cfg: QuantumKernelConfig):
    """Return the appropriate feature map function."""
    if cfg.feature_map == 'zz':
        return _zz_feature_map
    elif cfg.feature_map == 'pauli':
        return _pauli_feature_map
    elif cfg.feature_map == 'iqp':
        return _iqp_feature_map
    else:
        raise ValueError(f"Unknown feature map: {cfg.feature_map}")


def make_fidelity_kernel_element(cfg: QuantumKernelConfig):
    """Construct quantum kernel element k(x1, x2) = |<φ(x1)|φ(x2)>|²."""
    dev = qml.device("default.qubit", wires=cfg.n_qubits)
    feature_map = _get_feature_map(cfg)

    @qml.qnode(dev)
    def kernel_circuit(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute fidelity between feature states |φ(x1)> and |φ(x2)>."""
        feature_map(x1, cfg)
        qml.adjoint(feature_map)(x2, cfg)
        # Measuring |0...0> probability gives the squared overlap
        return qml.probs(wires=range(cfg.n_qubits))

    def k(x1: np.ndarray, x2: np.ndarray) -> float:
        """Kernel function: probability of measuring all-zero state."""
        try:
            probs = kernel_circuit(x1, x2)
            return float(np.clip(probs[0], 0.0, 1.0))
        except Exception as e:
            # Fallback for numerical issues
            print(f"Warning: Kernel evaluation failed, returning 0.0. Error: {e}")
            return 0.0

    return k


def center_kernel_matrix(K: np.ndarray) -> np.ndarray:
    """Center kernel matrix: K_centered = (I - 1/n*11^T) K (I - 1/n*11^T).
    
    This is standard preprocessing for kernel PCA and improves numerical stability.
    """
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_centered


class QuantumKernel:
    """Compute fidelity-based quantum kernel matrices with multiple feature maps.
    
    Supports:
    - Multiple feature map types (ZZ, Pauli, IQP)
    - Hardware noise simulation
    - Kernel centering
    - Progress tracking for large matrices
    - Symmetric computation optimization
    """

    def __init__(self, cfg: Optional[QuantumKernelConfig] = None):
        """Initialize quantum kernel with configuration."""
        self.cfg = cfg or QuantumKernelConfig()
        self._k = make_fidelity_kernel_element(self.cfg)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.cfg.n_qubits < 1:
            raise ValueError(f"n_qubits must be positive, got {self.cfg.n_qubits}")
        if self.cfg.n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {self.cfg.n_layers}")
        if not 0.0 <= self.cfg.noise_strength <= 1.0:
            raise ValueError(f"noise_strength must be in [0,1], got {self.cfg.noise_strength}")

    def kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute kernel matrix K[i,j] = k(X1[i], X2[j]).
        
        Args:
            X1: Data matrix (n1, n_features)
            X2: Second data matrix (n2, n_features). If None, compute K(X1, X1)
            show_progress: Show tqdm progress bar
            
        Returns:
            K: Kernel matrix (n1, n2)
        """
        X1 = np.asarray(X1, dtype=float)
        if X1.ndim != 2:
            raise ValueError(f"X1 must be 2D, got shape {X1.shape}")
        if X1.shape[1] != self.cfg.n_qubits:
            raise ValueError(f"Feature dim {X1.shape[1]} doesn't match n_qubits {self.cfg.n_qubits}")

        X2 = X1 if X2 is None else np.asarray(X2, dtype=float)
        if X2.ndim != 2 or X2.shape[1] != self.cfg.n_qubits:
            raise ValueError(f"X2 must be 2D with {self.cfg.n_qubits} features")

        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2), dtype=float)

        # Check if symmetric (same data)
        symmetric = X2 is X1 or (X2.shape == X1.shape and np.allclose(X2, X1))

        if symmetric:
            # Optimize symmetric case: compute upper triangle only
            total_evals = (n1 * (n1 + 1)) // 2
            pbar = tqdm(total=total_evals, desc="Quantum kernel (symmetric)", disable=not show_progress)
            
            for i in range(n1):
                K[i, i] = self._k(X1[i], X1[i])
                pbar.update(1)
                for j in range(i + 1, n1):
                    v = self._k(X1[i], X1[j])
                    K[i, j] = v
                    K[j, i] = v
                    pbar.update(1)
            pbar.close()
        else:
            # General case: compute full matrix
            total_evals = n1 * n2
            pbar = tqdm(total=total_evals, desc="Quantum kernel", disable=not show_progress)
            
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self._k(X1[i], X2[j])
                    pbar.update(1)
            pbar.close()

        # Apply centering if requested
        if self.cfg.centered and symmetric:
            K = center_kernel_matrix(K)

        return K

    def kernel_diag(self, X: np.ndarray, show_progress: bool = False) -> np.ndarray:
        """Compute diagonal kernel values k(x_i, x_i) for each sample.
        
        More efficient than computing full kernel matrix when only diagonal is needed.
        Used for kernel mean embedding anomaly scoring.
        
        Args:
            X: Data matrix (n_samples, n_features)
            show_progress: Show tqdm progress bar
            
        Returns:
            diag: Diagonal kernel values (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.cfg.n_qubits:
            raise ValueError(f"Feature dim {X.shape[1]} doesn't match n_qubits {self.cfg.n_qubits}")
        
        n = X.shape[0]
        diag = np.zeros(n, dtype=float)
        
        pbar = tqdm(total=n, desc="Kernel diagonal", disable=not show_progress)
        for i in range(n):
            diag[i] = self._k(X[i], X[i])
            pbar.update(1)
        pbar.close()
        
        return diag

    def kernel_alignment(self, K1: np.ndarray, K2: np.ndarray) -> float:
        """Compute kernel alignment between two kernel matrices.
        
        Alignment = <K1, K2>_F / (||K1||_F * ||K2||_F)
        
        Measures similarity between kernel geometries. Values near 1 indicate
        similar decision boundaries.
        """
        K1_flat = K1.flatten()
        K2_flat = K2.flatten()
        
        numerator = np.dot(K1_flat, K2_flat)
        denominator = np.linalg.norm(K1_flat) * np.linalg.norm(K2_flat)
        
        if denominator < 1e-12:
            return 0.0
        
        return float(numerator / denominator)

    def effective_dimension(self, K: np.ndarray, threshold: float = 0.99) -> int:
        """Compute effective dimension of kernel matrix.
        
        Number of eigenvalues needed to explain `threshold` fraction of variance.
        Low effective dimension suggests the kernel doesn't utilize the full
        feature space efficiently.
        """
        eigvals = np.linalg.eigvalsh(K)
        eigvals = np.sort(eigvals)[::-1]
        eigvals = np.maximum(eigvals, 0)  # Handle numerical noise
        
        cumsum = np.cumsum(eigvals)
        total = cumsum[-1]
        
        if total < 1e-12:
            return 0
        
        normalized_cumsum = cumsum / total
        n_components = np.searchsorted(normalized_cumsum, threshold) + 1
        
        return min(n_components, len(eigvals))

    def spectral_entropy(self, K: np.ndarray) -> float:
        """Compute Shannon entropy of normalized eigenvalue distribution.
        
        Measures the "richness" or "diversity" of the kernel's feature space.
        Higher entropy indicates more uniform eigenvalue distribution.
        Lower entropy indicates concentration on a few dominant eigenvectors.
        
        This provides a mechanistic explanation for quantum advantage:
        quantum kernels may show different spectral entropy than classical kernels,
        reflecting richer feature space geometry.
        
        Args:
            K: Kernel matrix (n, n)
            
        Returns:
            entropy: Spectral entropy (≥0)
        """
        eigvals = np.linalg.eigvalsh(K)
        eigvals = eigvals[eigvals > 1e-10]  # Filter near-zero eigenvalues
        
        if len(eigvals) == 0:
            return 0.0
        
        # Normalize to probability distribution
        p = eigvals / eigvals.sum()
        
        # Shannon entropy: H = -Σ p_i log(p_i)
        entropy = -np.sum(p * np.log(p + 1e-12))
        
        return float(entropy)

    def effective_rank(self, K: np.ndarray) -> float:
        """Compute effective rank of kernel matrix.
        
        Effective rank = exp(spectral_entropy)
        
        Provides an interpretable measure of kernel expressivity:
        - Effective rank ≈ n: Kernel uses full feature space
        - Effective rank << n: Kernel concentrates on low-dimensional subspace
        
        Args:
            K: Kernel matrix (n, n)
            
        Returns:
            eff_rank: Effective rank
        """
        entropy = self.spectral_entropy(K)
        return float(np.exp(entropy))
    
    def kernel_spectrum_stats(self, K: np.ndarray) -> dict:
        """Compute comprehensive spectral statistics for the kernel matrix."""
        from .kernel_spectrum import spectral_stats
        return spectral_stats(K)
    
    # Kernel alignment optimization methods
    def get_params(self) -> np.ndarray:
        """Get current learnable parameters as numpy array.
        
        Returns array of [gamma, noise_strength] (if noise enabled) or [gamma].
        """
        params = [self.cfg.gamma]
        if self.cfg.noise_model != 'none':
            params.append(self.cfg.noise_strength)
        return np.array(params)
    
    def set_params(self, params: np.ndarray) -> None:
        """Set learnable parameters from numpy array.
        
        Updates self.cfg and rebuilds kernel function.
        """
        params = np.asarray(params)
        self.cfg.gamma = float(params[0])
        if self.cfg.noise_model != 'none' and len(params) > 1:
            self.cfg.noise_strength = float(np.clip(params[1], 0.0, 1.0))
        # Rebuild kernel with new parameters
        self._k = make_fidelity_kernel_element(self.cfg)
    
    def build_kernel_for_alignment(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Build kernel matrix with given parameters (for alignment optimization).
        
        Temporarily sets params, computes kernel, restores old params.
        """
        old_params = self.get_params()
        self.set_params(params)
        K = self.kernel_matrix(X, show_progress=False)
        self.set_params(old_params)
        return K
    