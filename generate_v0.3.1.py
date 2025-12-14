import os
import zipfile
from pathlib import Path

# --- File Contents Definition ---

FILES = {}

# 1. pyproject.toml
FILES["pyproject.toml"] = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "qkernel-telemetry-anomaly"
version = "0.3.1"
description = "Quantum-kernel novelty detection on telemetry-like manifold data"
authors = [
    {name = "Christopher Altman", email = "x@christopheraltman.com"}
]
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "scikit-learn>=1.3.0",
    "pennylane>=0.35.0",
    "scipy>=1.11.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[tool.setuptools]
packages = ["src"]
"""

# 2. requirements.txt
FILES["requirements.txt"] = """numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pennylane>=0.35.0
scipy>=1.11.0
tqdm>=4.66.0
"""

# 3. CHANGELOG.md
FILES["CHANGELOG.md"] = """# Changelog

## [0.3.1] - 2025-12-14

### 🚀 Features & Enhancements
- **Per-Anomaly Spectral Signatures**: Visualizes kernel eigenvalue decay separated by anomaly type (Impulse, Drift, Dropout, Spoofing).
- **Spectral Divergence Metric**: Quantifies deviation of anomaly geometry from normal using Jensen-Shannon Divergence.
- **Kernel-Target Alignment (KTA)**: Added metric to measure alignment between kernel geometry and label structure.
- **Classical Smoke Test**: Added `--skip-quantum` flag for rapid pipeline verification (100x speedup for debugging).
- **Granular Data Splitting**: Fixed train/test split to preserve anomaly type labels for detailed analysis.

### 🔧 Optimizations
- **Unified Nyström**: Merged experimental Nyström utils into production pipeline.
- **Robust Math**: Added safe log/division handling in spectral entropy calculations.

## [0.3.0] - 2025-12-14
- Initial release of spectral analysis tools (Entropy, Effective Rank).
- Nyström approximation support.
"""

# 4. README.md
FILES["README.md"] = """# Quantum Kernel Telemetry Anomaly Detection (v0.3.1)

An advanced framework for detecting anomalies in satellite telemetry using quantum kernel methods. This system investigates whether quantum feature maps can better capture the geometry of low-dimensional manifolds (normal operations) embedded in high-dimensional observation spaces compared to classical kernels.

## New in v0.3.1
*   **Per-Anomaly Spectral Signatures**: Visualize how different failure modes (Impulse, Drift, Spoofing) perturb the quantum kernel spectrum.
*   **Spectral Divergence**: Quantify the geometric distance between normal and anomalous subspaces.
*   **Classical Smoke Test**: Run `--skip-quantum` to validate the pipeline in seconds.


## Installation

```bash
pip install -e .

"""
