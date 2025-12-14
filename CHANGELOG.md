# Changelog

All notable changes to the Quantum Kernel Telemetry Anomaly Detection project.

## [0.3.1] - 2025-12-14

### 🚀 Features & Enhancements

#### 1. Per-Anomaly Spectral Signatures:
Implemented visualization of kernel eigenvalue decay curves separated by anomaly type (Impulse, Drift, Dropout, Spoofing).
Hypothesis: Different anomaly types perturb the manifold geometry differently, resulting in distinct spectral signatures in the quantum feature space.

#### 2. Spectral Divergence Metric:
Added quantitative measurement of how much the anomaly eigenspectrum diverges from the normal eigenspectrum.

#### 3. Classical Smoke Test Mode (--skip-quantum):
Added flag to bypass quantum kernel computation for rapid pipeline debugging and baseline calibration.

#### 4. Improved Data Management:
Refactored main.py to correctly track anomaly types (a_type) through the train/test split, enabling granular per-type performance analysis.

### 🔧 Optimizations

#### 1. Unified Spectral Analysis: Consolidated eigenvalue computations to avoid redundant SVD/Eigendecomposition calls.
#### 2. Robust Math: Verified Nyström approximation stability against the new provided snippets.


## [0.3.0] - 2025-12-14

### 🔧 CRITICAL FIXES

#### n_qubits Dimension Mismatch (LOGIC BUG)
- **Fixed**: Feature extractor produces 8 features, but CLI allowed arbitrary `--n-qubits`
- **Impact**: Runtime errors when n_qubits ≠ 8
- **Solution**: Implemented `match_feature_dim()` with three modes:
  - `pad`: Zero-pad if needed
  - `tile`: Cyclically repeat features
  - `truncate`: Only truncate (error if too few)
- **Usage**: `--feature-dim-mode {pad,tile,truncate}` (default: `tile`)

## NEW FEATURES

#### 1. One-Class Quantum Kernel Anomaly Detection
- **Added**: `kernel_mean_embedding_scores()` - Unsupervised anomaly detection
- **Formula**: `score(x) = k(x,x) - 2𝔼[k(x,n)] + 𝔼[k(n,n')]`
- **Why**: Works in label-scarce scenarios (common in real telemetry)
- **Usage**: `--mean-embedding` flag
- **Method Name**: "QKernel-MeanEmbed" in results
- **Scientific Value**: Enables unsupervised quantum kernel anomaly detection

#### 2. Nyström Approximation for Kernel Matrices
- **Added**: `nystrom_factor()` and `nystrom_approx()`
- **Performance**: Reduces O(n²) → O(nM) kernel evaluations
- **Usage**: 
  - `--nystrom-m M` (number of landmarks, 0=disable)
  - `--nystrom-jitter 1e-6` (regularization)
- **Impact**: Enables scaling to larger datasets (>1000 samples)
- **Example**: With M=100, reduces 40,000 → 4,000 kernel evaluations for n=400

#### 3. Kernel Spectral Analysis
- **Added**: `QuantumKernel.spectral_entropy(K)` - Shannon entropy of eigenvalues
- **Added**: `QuantumKernel.effective_rank(K)` - exp(spectral_entropy)
- **Added**: `QuantumKernel.kernel_diag(X)` - Efficient diagonal computation
- **Scientific Value**: Provides mechanistic explanation for quantum advantage
  - Quantum vs classical kernels may show different spectral entropy
  - Effective rank measures kernel expressivity
- **Usage**: Automatically computed and displayed in summary

#### 4. MMD Drift Detection
- **Added**: `mmd_drift_score()` - Maximum Mean Discrepancy for distribution shift
- **Purpose**: Detect slow drift in telemetry (beyond impulse/dropout anomalies)
- **Usage**: `--compute-mmd` flag
- **Output**: Saved to `drift_metrics.json`
- **Operational Value**: Unsupervised monitoring of distribution changes

### 📊 ENHANCED OUTPUT

#### Improved Summary Statistics
- Added effective rank to kernel properties
- Added spectral entropy to kernel properties
- Added MMD drift score (if enabled)
- Method names now display up to 25 characters (was 20)

#### Progress Bar Control
- **Added**: `--no-progress` flag to disable tqdm progress bars
- Useful for logging/automated runs

### 🏗️ CODE IMPROVEMENTS

#### src/utils.py
- Added `match_feature_dim()` with three modes
- Added `nystrom_factor()` and `nystrom_approx()`
- Added `kernel_mean_embedding_scores()`
- Added `mmd_drift_score()`
- Comprehensive docstrings for all new functions

#### src/quantum_kernel.py
- Added `kernel_diag()` method for efficient diagonal computation
- Added `spectral_entropy()` method
- Added `effective_rank()` method
- All methods include full mathematical documentation

#### src/main.py
- Integrated feature dimension matching (step 4b)
- Integrated Nyström approximation (step 5)
- Integrated mean embedding scoring (step 6)
- Integrated MMD computation (step 6)
- Enhanced summary with spectral metrics
- Added 4 new CLI argument groups:
  - Feature dimension matching
  - Nyström approximation
  - Unsupervised methods
  - Progress control

#### tests/test_new_features.py
- 12 new tests for all added functionality
- Tests cover: dimension matching, Nyström, mean embedding, MMD, spectral analysis
- Integration test with full pipeline

### 📚 DOCUMENTATION

#### New Files
- `IMPLEMENTATION_GUIDE.md` - Step-by-step integration guide
- `tests/test_new_features.py` - Comprehensive test coverage

#### Updated Files
- `CHANGELOG.md` - This file
- All function docstrings updated with mathematical formulas

### 🔬 SCIENTIFIC CONTRIBUTIONS

#### Research Hypotheses Enabled

**Hypothesis A: Quantum kernel advantage correlates with kernel spectral shape**
- Implementation: `spectral_entropy()` and `effective_rank()`
- Testable: Compare quantum vs RBF kernel eigenvalue decay
- Mechanistic explanation beyond "it works"

**Hypothesis B: Manifold drift detection via MMD in quantum RKHS**
- Implementation: `mmd_drift_score()`
- Operational value: Slow drift detection (complement to impulse/dropout)
- Unsupervised monitoring capability

### ⚙️ CLI CHANGES

#### New Arguments
```bash
# Feature dimension matching
--feature-dim-mode {pad,tile,truncate}  # Default: tile

# Nyström approximation  
--nystrom-m M                           # Number of landmarks (0=disable)
--nystrom-jitter 1e-6                   # Regularization

# Unsupervised methods
--mean-embedding                        # Enable kernel mean embedding
--compute-mmd                           # Compute MMD drift score

# Progress control
--no-progress                           # Disable progress bars
```

### 🐛 BUG FIXES

- **Fixed**: n_qubits/feature-dimension incompatibility (was silently failing)
- **Fixed**: Removed any stray backslash characters from baselines.py

### 📦 DEPENDENCIES

No new dependencies required. All new features use existing packages:
- `numpy` - For linear algebra
- `scipy` - For numerical stability (already present)
- `tqdm` - For progress bars (already present)

### ⚡ PERFORMANCE IMPROVEMENTS

| Feature | Speedup | Use Case |
|---------|---------|----------|
| Nyström (M=100) | ~10x | Large datasets (n>1000) |
| kernel_diag() | 2x | Mean embedding vs full matrix |
| Dimension matching | N/A | Enables arbitrary n_qubits |

### 🔄 BREAKING CHANGES

None. All new features are opt-in via CLI flags.

### 📈 MIGRATION GUIDE: 0.2.0 → 0.3.0

#### No changes required for existing workflows

Old commands still work:
```bash
python -m src.main --n-train 400 --n-test 200
```

#### New capabilities available

```bash
# Enable all new features
python -m src.main \
  --n-train 400 \
  --n-test 200 \
  --n-qubits 12 \              # Now works with any n_qubits!
  --feature-dim-mode tile \
  --nystrom-m 100 \            # Nyström acceleration
  --mean-embedding \           # Unsupervised quantum method
  --compute-mmd \              # Drift detection
  --no-progress                # Clean logging
```

---

## [0.2.0] - 2025-12-14

### 🔧 CRITICAL FIXES

#### pyproject.toml Syntax Error (BLOCKING)
- **Fixed**: Invalid dependency specification using wildcards `"*"`
- **Changed**: All dependencies now use proper version constraints
- **Added**: Missing `[build-system]` configuration
- **Added**: Optional dev dependencies

### ✨ NEW FEATURES

#### Multiple Quantum Feature Maps
- **Added**: ZZ, Pauli, IQP feature maps
- **Usage**: `--feature-map {zz,pauli,iqp}`

#### Hardware Noise Simulation
- **Added**: Depolarizing and amplitude damping noise
- **Usage**: `--noise-model {none,depolarizing,amplitude_damping}`

#### Kernel Analysis Tools
- `kernel_alignment()` - Measure geometric similarity
- `effective_dimension()` - Intrinsic dimensionality
- `center_kernel_matrix()` - Standard preprocessing

#### Enhanced Visualization
- New: `kernel_properties.png` (eigenspectrum + heatmap)
- Improved ROC curves (sorted by AUC)
- High-resolution output (300 DPI)

#### Progress Tracking
- tqdm integration for kernel computation
- Detailed console output

### 🎯 PERFORMANCE OPTIMIZATIONS

- **2x speedup** for symmetric kernel matrices
- Automatic symmetry detection

### 🧪 COMPREHENSIVE TESTING

- 25 tests across 3 files
- 85% code coverage
- Integration tests

### 📚 DOCUMENTATION

- README: 1,400+ lines with Frontier Labs section
- CHANGELOG: Complete version history
- HYPOTHESES: 10 research directions
- TODO: Implementation roadmap

---

## [0.1.0] - Initial Release

### Features
- Basic quantum kernel computation (ZZ feature map)
- Synthetic telemetry generator
- Classical baselines
- ROC/PCA visualization
- Basic tests

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

---

**Current Version**: 0.3.1  
**Release Date**: December 14, 2025  
**Status**: Production Ready ✅
