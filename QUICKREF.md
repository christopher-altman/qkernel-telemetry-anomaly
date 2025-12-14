# Quick Reference Card

## Installation
```bash
cd qkernel-telemetry-anomaly
pip install -e .
```

## Basic Usage
```bash
# Default run
python -m src.main

# Custom configuration
python -m src.main --n-train 600 --n-test 300 --anomaly-rate 0.2
```

## Feature Maps
```bash
--feature-map zz      # ZZ-entangling (default, pairwise correlations)
--feature-map pauli   # RX-RY-RZ rotations (maximal coverage)
--feature-map iqp     # IQP encoding (classically hard)
```

## Noise Models
```bash
--noise-model none                 # No noise (default)
--noise-model depolarizing         # Symmetric gate errors
--noise-model amplitude_damping    # T1 relaxation
--noise-strength 0.02              # Noise parameter (0-1)
```

## Advanced Options
```bash
--n-qubits 8         # Number of qubits (default: 8)
--n-layers 3         # Feature map depth (default: 2)
--gamma 1.5          # ZZ coupling strength (default: 1.0)
--qsvm-c 5.0         # SVM regularization (default: 3.0)
--centered           # Apply kernel centering
--no-plots           # Disable plotting
```

## Testing
```bash
# Run all tests
pytest tests/ -v

# Specific test file
pytest tests/test_kernel.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Validation
```bash
# Comprehensive validation
python validate.py

# Should output:
# ✅ ALL TESTS PASSED - Code is ready to deploy.
```

## Output Files
```
figures/
├── roc.png                  # ROC curve comparison
├── kernel_pca.png           # Kernel geometry visualization
└── kernel_properties.png    # Eigenspectrum + heatmap

results/run1/
├── metrics.json             # Performance metrics
└── config.json              # Experimental config

data/
└── telemetry_dataset.npz    # Generated data
```

## Example Workflows

### 1. Quick Test (Small Scale)
```bash
python -m src.main --n-train 100 --n-test 50 --n-qubits 4 --n-layers 1
```

### 2. Publication-Quality Run
```bash
python -m src.main \
  --n-train 600 \
  --n-test 300 \
  --anomaly-rate 0.15 \
  --feature-map iqp \
  --n-layers 3 \
  --outdir results/publication_run
```

### 3. Noise Robustness Study
```bash
for noise in 0.0 0.01 0.02 0.05 0.1; do
  python -m src.main \
    --noise-model depolarizing \
    --noise-strength $noise \
    --outdir results/noise_$noise
done
```

### 4. Feature Map Comparison
```bash
for fm in zz pauli iqp; do
  python -m src.main \
    --feature-map $fm \
    --outdir results/fm_$fm
done
```

## Programmatic Usage

```python
from src.quantum_kernel import QuantumKernel, QuantumKernelConfig
from src.telemetry_sim import make_dataset, TelemetryConfig
from src.utils import window_features_mean_std, minmax_to_0_2pi
import numpy as np

# Generate data
cfg = TelemetryConfig(timesteps=128)
X, y, _ = make_dataset(100, 0.15, cfg, seed=42)

# Extract features
F = window_features_mean_std(X)

# Scale to [0, 2π]
xmin, xmax = F.min(axis=0), F.max(axis=0)
F_scaled = minmax_to_0_2pi(F, xmin, xmax)

# Quantum kernel
qcfg = QuantumKernelConfig(
    n_qubits=8,
    n_layers=2,
    feature_map='iqp',
    noise_model='depolarizing',
    noise_strength=0.02
)
qk = QuantumKernel(qcfg)

# Compute kernel
K = qk.kernel_matrix(F_scaled, show_progress=True)

# Analysis
eff_dim = qk.effective_dimension(K, threshold=0.99)
print(f"Effective dimension: {eff_dim}")
```

## Key Classes

### QuantumKernelConfig
```python
QuantumKernelConfig(
    n_qubits=8,           # Qubit count
    n_layers=2,           # Circuit depth
    gamma=1.0,            # ZZ coupling
    feature_map='zz',     # {zz, pauli, iqp}
    noise_model='none',   # {none, depolarizing, amplitude_damping}
    noise_strength=0.01,  # Noise parameter
    centered=False        # Kernel centering
)
```

### TelemetryConfig
```python
TelemetryConfig(
    timesteps=128,        # Window length
    dt=1.0,              # Time step
    n_channels=4,        # Number of channels
    noise_std=0.03       # Observation noise
)
```

## Performance Tips

1. **Small-scale testing**: Use `--n-qubits 4 --n-layers 1` for rapid iteration
2. **Symmetric speedup**: Training kernels are 2x faster (automatic)
3. **Progress bars**: Disable with `show_progress=False` in Python API
4. **Memory**: Kernel matrix is O(n²), limit n_train to <2000 for 16GB RAM

## Common Issues

### Import Error
```bash
# Solution: Install in editable mode
pip install -e .
```

### Slow Kernel Computation
```bash
# Solution: Reduce n_layers or use fewer qubits
--n-layers 1 --n-qubits 6
```

### Memory Error
```bash
# Solution: Reduce training set size
--n-train 400
```

## Citation
```bibtex
@software{altman2025qkernel,
  title={Quantum Kernel Telemetry Anomaly Detection},
  author={Altman, Christopher},
  year={2025},
  version={0.2.0},
  url={https://github.com/christopher-altman/qkernel-telemetry-anomaly}
}
```

## Help
```bash
python -m src.main --help
```

## Version
**Current**: 0.3.1 (14 December 2025)
