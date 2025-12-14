from __future__ import annotations

import dataclasses
import numpy as np


@dataclasses.dataclass
class TelemetryConfig:
    """Configuration for synthetic telemetry windows.

    The generator intentionally produces *manifold-structured* normal behavior:
    normals are controlled by a small latent vector (frequency, phases, coupling),
    then mapped through nonlinear channel mixings.

    Channels (default 4):
      0: "orbit-like" radial proxy
      1: "orbit-like" tangential proxy
      2: "attitude-like" proxy
      3: "power/thermal-like" proxy
    """
    timesteps: int = 128
    dt: float = 1.0
    n_channels: int = 4
    noise_std: float = 0.03


def _latent_params(rng: np.random.Generator, n: int):
    # A few latent degrees of freedom → manifold structure
    base_freq = rng.uniform(0.03, 0.10, size=n)          # quasi-periodic base
    phase0 = rng.uniform(0, 2*np.pi, size=n)
    phase1 = rng.uniform(0, 2*np.pi, size=n)
    coupling = rng.uniform(0.2, 0.9, size=n)
    drift = rng.uniform(-0.002, 0.002, size=n)           # slow trend in normals
    return base_freq, phase0, phase1, coupling, drift


def generate_normal_windows(
    n_samples: int,
    cfg: TelemetryConfig,
    seed: int = 0,
) -> np.ndarray:
    """Generate normal telemetry windows.

    Returns:
        X: (n_samples, timesteps, n_channels)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(cfg.timesteps) * cfg.dt
    base_freq, phase0, phase1, coupling, drift = _latent_params(rng, n_samples)

    X = np.zeros((n_samples, cfg.timesteps, cfg.n_channels), dtype=float)

    for k in range(n_samples):
        w = base_freq[k]
        c = coupling[k]
        # Latent oscillators (phase-coupled)
        s0 = np.sin(2*np.pi*w*t + phase0[k])
        s1 = np.cos(2*np.pi*(w*(1.0 + 0.2*c))*t + phase1[k])
        s2 = np.sin(2*np.pi*(w*(0.5 + 0.3*c))*t + phase0[k] - 0.7*phase1[k])

        # Channel mixings (nonlinear)
        ch0 = 1.2*s0 + 0.4*s1 + 0.15*np.tanh(2*s2) + drift[k]*t
        ch1 = 0.9*s1 - 0.3*s0 + 0.10*(s0*s1) + 0.5*np.sin(0.3*s2)
        ch2 = 0.8*s2 + 0.25*np.tanh(1.5*s0) - 0.15*np.tanh(1.0*s1)
        ch3 = 0.6*np.tanh(1.2*s0 + 0.8*s1) + 0.2*s2 + 0.1*np.cos(2*np.pi*w*t)

        X[k, :, 0] = ch0
        X[k, :, 1] = ch1
        X[k, :, 2] = ch2
        X[k, :, 3] = ch3

    X += rng.normal(0.0, cfg.noise_std, size=X.shape)
    return X


# --- Anomaly injection -----------------------------------------------------


def inject_impulse(
    X: np.ndarray,
    rng: np.random.Generator,
    magnitude: float = 1.5,
    channel_prob: float = 0.6,
) -> None:
    """Add a single-step impulse to random channels at a random time."""
    n, T, C = X.shape
    for k in range(n):
        t0 = rng.integers(low=T//8, high=7*T//8)
        mask = rng.random(C) < channel_prob
        X[k, t0, mask] += rng.normal(0, magnitude, size=mask.sum())


def inject_drift(
    X: np.ndarray,
    rng: np.random.Generator,
    slope: float = 0.02,
    channel_prob: float = 0.5,
) -> None:
    """Add a gradual drift starting mid-window."""
    n, T, C = X.shape
    t = np.arange(T)
    for k in range(n):
        t0 = rng.integers(low=T//4, high=T//2)
        mask = rng.random(C) < channel_prob
        n_masked = mask.sum()
        if n_masked == 0:
            continue
        
        drift_curve = (t - t0).clip(min=0) * slope
        signs = rng.choice([-1.0, 1.0], size=n_masked)
        
        # Apply drift to each masked channel
        for i, c_idx in enumerate(np.where(mask)[0]):
            X[k, :, c_idx] += drift_curve * signs[i]


def inject_dropout(
    X: np.ndarray,
    rng: np.random.Generator,
    drop_len: int = 12,
    channel_prob: float = 0.5,
) -> None:
    """Zero out a short segment (sensor dropout)."""
    n, T, C = X.shape
    for k in range(n):
        # Adjust drop_len if it's too long for the window
        effective_drop_len = min(drop_len, T // 2)
        low = T // 6
        high = 5 * T // 6 - effective_drop_len
        # Ensure valid range
        if high <= low:
            high = low + 1
        t0 = rng.integers(low=low, high=high)
        mask = rng.random(C) < channel_prob
        X[k, t0:t0+effective_drop_len, mask] = 0.0


def inject_spoofing(
    X: np.ndarray,
    rng: np.random.Generator,
    phase_shift: float = 1.0,
    noise: float = 0.15,
    channel_prob: float = 0.5,
) -> None:
    """Phase shift + structured noise (spoofing-like perturbation)."""
    n, T, C = X.shape
    t = np.arange(T)
    for k in range(n):
        mask = rng.random(C) < channel_prob
        n_masked = mask.sum()
        if n_masked == 0:
            continue
        # Add a coherent oscillatory component plus phase offset
        spoof = np.sin(2*np.pi*0.07*t + phase_shift * rng.uniform(0.5, 1.5))
        scale = rng.uniform(0.3, 0.8)
        # Apply to each masked channel
        for c_idx in np.where(mask)[0]:
            X[k, :, c_idx] += spoof * scale + rng.normal(0, noise, size=T)


def make_dataset(
    n_samples: int,
    anomaly_rate: float,
    cfg: TelemetryConfig,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate labeled dataset (X, y, anomaly_type_id).

    y: 0 = normal, 1 = anomalous
    anomaly_type_id: -1 for normal, else integer in [0..3]
    """
    rng = np.random.default_rng(seed)
    X = generate_normal_windows(n_samples, cfg, seed=seed)
    y = np.zeros(n_samples, dtype=int)
    a_type = np.full(n_samples, -1, dtype=int)

    n_anom = int(round(n_samples * anomaly_rate))
    if n_anom == 0:
        return X, y, a_type

    idx = rng.choice(n_samples, size=n_anom, replace=False)
    y[idx] = 1

    # Assign anomaly types
    types = rng.integers(low=0, high=4, size=n_anom)
    a_type[idx] = types

    X_anom = X[idx].copy()
    # Apply in-place to the anomalous subset
    for t_id in range(4):
        sel = np.where(types == t_id)[0]
        if sel.size == 0:
            continue
        X_sub = X_anom[sel]
        if t_id == 0:
            inject_impulse(X_sub, rng)
        elif t_id == 1:
            inject_drift(X_sub, rng)
        elif t_id == 2:
            inject_dropout(X_sub, rng)
        elif t_id == 3:
            inject_spoofing(X_sub, rng)
        X_anom[sel] = X_sub

    X[idx] = X_anom
    return X, y, a_type
