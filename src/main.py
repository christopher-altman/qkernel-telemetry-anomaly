"""Main entry point for quantum kernel telemetry anomaly detection (v0.3.1)."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split

# Fallback for classical-only mode
from sklearn.metrics.pairwise import rbf_kernel

from .baselines import (
    BaselineConfig,
    isolation_forest_scores,
    oneclass_svm_scores,
    precomputed_kernel_svm_scores,
    rbf_svm_scores,
)
from .eval import compute_metrics, roc_points
from .quantum_kernel import QuantumKernel, QuantumKernelConfig
from .telemetry_sim import TelemetryConfig, make_dataset
from .utils import (
    ensure_dir,
    minmax_to_0_2pi,
    save_json,
    window_features_mean_std,
    match_feature_dim,
    nystrom_factor,
    nystrom_approx,
    kernel_mean_embedding_scores,
    mmd_drift_score,
    get_spectral_signature,
    spectral_divergence
)

# NEW: Import from v0.3.1 modules
from .kernel_spectrum import spectral_stats
from .mmd import mmd2_unbiased, mmd
from .kernel_alignment import align_optimize, kta_score


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantum-kernel telemetry anomaly detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data generation
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=128)
    p.add_argument("--noise-std", type=float, default=0.03)
    p.add_argument("--anomaly-rate", type=float, default=0.15)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-test", type=int, default=200)

    # Quantum Kernel
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--feature-map", type=str, default="zz", choices=["zz", "pauli", "iqp"])
    p.add_argument("--noise-model", type=str, default="none", choices=["none", "depolarizing", "amplitude_damping"])
    p.add_argument("--noise-strength", type=float, default=0.01)
    p.add_argument("--centered", action="store_true")
    p.add_argument("--feature-dim-mode", type=str, default="tile", choices=["pad", "tile", "truncate"])
    
    # Smoke test / Baseline mode (NEW v0.3.1)
    p.add_argument("--skip-quantum", action="store_true", 
                   help="Skip quantum kernel (use RBF only) for fast baseline check")

    # Nyström
    p.add_argument("--nystrom-m", type=int, default=0)
    p.add_argument("--nystrom-jitter", type=float, default=1e-6)
    
    # Unsupervised
    p.add_argument("--mean-embedding", action="store_true")
    p.add_argument("--compute-mmd", action="store_true")
    
    # NEW v0.3.1: Spectral diagnostics
    p.add_argument("--spectral-diagnostics", action="store_true",
                   help="Enable detailed spectral analysis and per-anomaly signature plots")
    
    # Kernel alignment optimization
    p.add_argument("--align-kernel", action="store_true",
                   help="Optimize quantum kernel parameters via kernel-target alignment")
    p.add_argument("--align-iters", type=int, default=25,
                   help="Number of alignment optimization iterations")
    p.add_argument("--align-lr", type=float, default=0.15,
                   help="Learning rate for alignment optimization")
    p.add_argument("--align-batch", type=int, default=64,
                   help="Mini-batch size for alignment (0=full batch)")
    p.add_argument("--align-seed", type=int, default=0,
                   help="Random seed for alignment optimization")
    
    # Output
    p.add_argument("--outdir", type=str, default="results/run1")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-progress", action="store_true")

    return p.parse_args()


def plot_spectral_signatures(
    K_full_test: np.ndarray, 
    y_test: np.ndarray, 
    a_type_test: np.ndarray, 
    outpath: Path
):
    """Plot eigenvalue decay curves separated by anomaly type.
    
    This visualization shows how different anomaly types perturb the 
    kernel geometry differently, resulting in distinct spectral signatures.
    """
    plt.figure(figsize=(10, 6))
    
    # Map integer IDs to names (from telemetry_sim.py)
    anom_names = {0: 'Impulse', 1: 'Drift', 2: 'Dropout', 3: 'Spoofing'}
    
    # 1. Normal signature
    normal_idx = np.where(y_test == 0)[0]
    sig_norm = None
    if len(normal_idx) > 5:
        K_norm = K_full_test[np.ix_(normal_idx, normal_idx)]
        sig_norm = get_spectral_signature(K_norm)
        plt.plot(sig_norm, 'k-', linewidth=2.5, label='Normal', alpha=0.9)

    # 2. Anomaly signatures
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    divergences = {}
    for aid, name in anom_names.items():
        # Get indices for this specific anomaly type
        idx = np.where((y_test == 1) & (a_type_test == aid))[0]
        
        if len(idx) > 5:
            # Extract sub-kernel for this class
            K_sub = K_full_test[np.ix_(idx, idx)]
            sig = get_spectral_signature(K_sub)
            plt.plot(sig, color=colors[aid], linewidth=1.5, label=name, alpha=0.8)
            
            # Calculate divergence from normal if available
            if sig_norm is not None:
                div = spectral_divergence(sig_norm, sig)
                divergences[name] = div
                print(f"  Spectral Divergence (Normal vs {name}): {div:.4f}")

    plt.yscale('log')
    plt.xlabel('Eigenvalue Index', fontsize=12)
    plt.ylabel('Normalized Eigenvalue (Log Scale)', fontsize=12)
    plt.title('Per-Anomaly Spectral Signatures (Kernel Geometry)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    
    return divergences


def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    figdir = ensure_dir("figures")
    ensure_dir("data")
    
    print("="*60)
    print("QUANTUM KERNEL TELEMETRY ANOMALY DETECTION v0.3.1")
    print("="*60)

    # 1. Data Generation
    print("[1] Generating data...")
    cfg = TelemetryConfig(timesteps=args.timesteps, noise_std=args.noise_std)
    X, y, a_type = make_dataset(
        args.n_train + args.n_test, 
        args.anomaly_rate, 
        cfg, 
        args.seed
    )
    
    # 2. Feature Extraction
    F = window_features_mean_std(X)
    
    # 3. Train/Test Split (Tracking Anomaly Types!)
    F_train, F_test, y_train, y_test, a_train, a_test = train_test_split(
        F, y, a_type, 
        test_size=args.n_test/(args.n_train+args.n_test), 
        random_state=args.seed, 
        stratify=y
    )

    # 4. Feature Processing
    xmin, xmax = F_train.min(axis=0), F_train.max(axis=0)
    F_train_s = minmax_to_0_2pi(F_train, xmin, xmax)
    F_test_s = minmax_to_0_2pi(F_test, xmin, xmax)
    
    F_train_s = match_feature_dim(F_train_s, args.n_qubits, args.feature_dim_mode)
    F_test_s = match_feature_dim(F_test_s, args.n_qubits, args.feature_dim_mode)

    # 5. Kernel Computation
    K_train, K_test = None, None
    qk = None

    if not args.skip_quantum:
        print(f"[2] Computing Quantum Kernel ({args.feature_map})...")
        qcfg = QuantumKernelConfig(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            feature_map=args.feature_map,
            noise_model=args.noise_model,
            noise_strength=args.noise_strength,
            centered=args.centered
        )
        qk = QuantumKernel(qcfg)
        
        # Kernel alignment optimization (if enabled)
        if args.align_kernel:
            print("[2b] Optimizing Kernel-Target Alignment...")
            
            # Get initial parameters
            init_params = qk.get_params()
            init_kta = kta_score(qk.kernel_matrix(F_train_s[:50], show_progress=False), y_train[:50])
            print(f"    Initial params: {init_params}")
            print(f"    Initial KTA (subsample): {init_kta:.4f}")
            
            # Define kernel builder for alignment
            def build_kernel(params, X):
                return qk.build_kernel_for_alignment(params, X)
            
            # Optimize
            opt_params, align_hist = align_optimize(
                kernel_builder=build_kernel,
                params=init_params,
                X=F_train_s,
                y=y_train,
                iters=args.align_iters,
                lr=args.align_lr,
                batch=args.align_batch,
                seed=args.align_seed,
                verbose=not args.no_progress
            )
            
            # Set optimized parameters
            qk.set_params(opt_params)
            print(f"    Optimized params: {opt_params}")
            print(f"    Final KTA: {align_hist[-1]:.4f}")
            print(f"    Improvement: {align_hist[-1] - align_hist[0]:.4f}")
            
            # Save alignment results
            save_json(outdir / "alignment.json", {
                "initial_params": init_params.tolist(),
                "optimized_params": opt_params.tolist(),
                "kta_history": align_hist,
                "initial_kta": float(align_hist[0]),
                "final_kta": float(align_hist[-1]),
                "improvement": float(align_hist[-1] - align_hist[0])
            })
        
        # Nyström Logic
        if args.nystrom_m > 0 and args.nystrom_m < F_train_s.shape[0]:
            print(f"    Using Nyström approx (m={args.nystrom_m})")
            idx = np.random.choice(F_train_s.shape[0], args.nystrom_m, replace=False)
            lm = F_train_s[idx]
            
            K_mm = qk.kernel_matrix(lm, show_progress=not args.no_progress)
            K_nm_train = qk.kernel_matrix(F_train_s, lm, show_progress=not args.no_progress)
            K_nm_test = qk.kernel_matrix(F_test_s, lm, show_progress=not args.no_progress)
            
            K_train = nystrom_approx(K_nm_train, K_mm, args.nystrom_jitter)
            C = nystrom_factor(K_mm, args.nystrom_jitter)
            K_test = K_nm_test @ C @ K_nm_train.T
        else:
            K_train = qk.kernel_matrix(F_train_s, show_progress=not args.no_progress)
            K_test = qk.kernel_matrix(F_test_s, F_train_s, show_progress=not args.no_progress)
            
        # Spectral Analysis (always compute, show if --spectral-diagnostics)
        print("[3] Analyzing Spectral Geometry...")
        
        # Use new kernel_spectrum module
        stats = spectral_stats(K_train)
        ent = stats['spectral_entropy']
        rank = stats['effective_rank']
        
        print(f"    Spectral Entropy: {ent:.4f}")
        print(f"    Effective Rank:   {rank:.2f}")
        
        # Save spectral stats if diagnostics enabled
        if args.spectral_diagnostics:
            spectral_results = {
                "spectral_entropy": ent,
                "effective_rank": rank,
                "eigenvalues_top20": stats['eigenvalues'][:20].tolist()
            }
            save_json(outdir / "spectral_stats.json", spectral_results)

    else:
        print("[2] Skipping Quantum Kernel (Classical Smoke Test)...")
        # Use RBF for matrix shapes
        gamma = 1.0 / F_train_s.shape[1]
        K_train = rbf_kernel(F_train_s, gamma=gamma)
        K_test = rbf_kernel(F_test_s, F_train_s, gamma=gamma)

    # 6. Modeling
    print("[4] Training Models...")
    scores = {}
    
    # 6a. Kernel SVM (Quantum or RBF depending on flag)
    prefix = "RBF" if args.skip_quantum else "QKernel"
    scores[f"{prefix}-SVM"] = precomputed_kernel_svm_scores(
        K_train, y_train, K_test, C=3.0, random_state=args.seed
    )

    # 6b. Mean Embedding (Unsupervised)
    if args.mean_embedding and not args.skip_quantum:
        K_diag = qk.kernel_diag(F_test_s)
        scores["QKernel-MeanEmbed"] = kernel_mean_embedding_scores(K_train, K_test, K_diag)

    # 6c. Baselines
    scores["IsolationForest"] = isolation_forest_scores(
        F_train, F_test, BaselineConfig(random_state=args.seed)
    )

    # 7. Drift Detection (using new mmd module)
    if args.compute_mmd:
        print("[5] Computing MMD drift score...")
        # Require test-test kernel
        if not args.skip_quantum:
            K_test_test = qk.kernel_matrix(F_test_s, show_progress=False)
        else:
            K_test_test = rbf_kernel(F_test_s)
        
        # Use mmd2_unbiased from mmd module
        mmd_value = mmd2_unbiased(K_train[:len(K_train)//2, :len(K_train)//2],
                                   K_train[len(K_train)//2:, len(K_train)//2:],
                                   K_train[:len(K_train)//2, len(K_train)//2:])
        mmd_sqrt = mmd(K_train[:len(K_train)//2, :len(K_train)//2],
                       K_train[len(K_train)//2:, len(K_train)//2:],
                       K_train[:len(K_train)//2, len(K_train)//2:])
        
        print(f"    MMD² (unbiased): {mmd_value:.6f}")
        print(f"    MMD (distance):  {mmd_sqrt:.4f}")
        save_json(outdir / "drift.json", {"mmd2_unbiased": mmd_value, "mmd": mmd_sqrt})

    # 8. Evaluation & Plotting
    metrics = {k: asdict(compute_metrics(y_test, v)) for k, v in scores.items()}
    save_json(outdir / "metrics.json", metrics)
    
    if not args.no_plots:
        # ROC
        plt.figure(figsize=(8,6))
        for name, res in metrics.items():
            fpr, tpr, _ = roc_points(y_test, scores[name])
            plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Anomaly Detection Performance', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figdir / "roc.png", dpi=300)
        plt.close()
        
        # NEW v0.3.1: Spectral Signatures (if enabled and quantum)
        if args.spectral_diagnostics and not args.skip_quantum and qk:
            print("[6] Generating per-anomaly spectral signatures...")
            # Compute full test kernel for spectral analysis
            K_tt_full = qk.kernel_matrix(F_test_s, show_progress=False)
            divergences = plot_spectral_signatures(
                K_tt_full, y_test, a_test, figdir / "spectral_signatures.png"
            )
            
            # Save divergences
            save_json(outdir / "spectral_divergences.json", divergences)
            print(f"    Saved spectral signatures to {figdir / 'spectral_signatures.png'}")
        
        # Alignment curve (if enabled)
        if args.align_kernel and 'align_hist' in locals():
            print("    Generating alignment curve plot...")
            plt.figure(figsize=(8, 6))
            plt.plot(align_hist, 'o-', linewidth=2, markersize=6)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Kernel-Target Alignment (KTA)', fontsize=12)
            plt.title('Kernel Alignment Optimization', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(figdir / "alignment_curve.png", dpi=300)
            plt.close()
            print(f"    Saved alignment curve to {figdir / 'alignment_curve.png'}")

    # Summary
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    for name, m in metrics.items():
        print(f"  {name:<25}: ROC-AUC = {m['roc_auc']:.4f}")
    
    print(f"\nOutput saved to: {outdir}/")
    print("="*60)

if __name__ == "__main__":
    main()
