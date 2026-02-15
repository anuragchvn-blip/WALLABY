"""Validation metrics for reconstruction quality assessment."""

import numpy as np
from typing import Dict, Tuple
from scipy.stats import pearsonr


def correlation_coefficient(delta_true: np.ndarray, delta_rec: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between true and reconstructed density."""
    mask = np.isfinite(delta_true) & np.isfinite(delta_rec)
    if np.sum(mask) < 2:
        return 0.0
    return pearsonr(delta_true[mask], delta_rec[mask])[0]


def bias_quantification(delta_true: np.ndarray, delta_rec: np.ndarray) -> Dict[str, float]:
    """Compute systematic bias in reconstruction."""
    mask = np.isfinite(delta_true) & np.isfinite(delta_rec)
    residual = delta_rec[mask] - delta_true[mask]
    std_true = np.std(delta_true[mask])
    return {
        "mean_bias": float(np.mean(residual)),
        "std_bias": float(np.std(residual)),
        "bias_fraction": float(np.mean(residual) / std_true) if std_true > 0 else 0.0
    }


def compute_power_spectrum(delta: np.ndarray, box_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum P(k) from density field."""
    delta_k = np.fft.fftn(delta)
    n = delta.shape[0]
    k = np.fft.fftfreq(n, d=box_size/n) * 2 * np.pi
    k_mag = np.sqrt(sum(k**2 for k in np.meshgrid(k, k, k, indexing='ij')))
    Pk = np.abs(delta_k)**2
    return k_mag.flatten(), Pk.flatten()


def validate_bulk_flow(velocity_true: Dict, velocity_rec: Dict, r_max: float = 150.0) -> Dict:
    """Validate bulk flow reconstruction."""
    def bulk_flow_magnitude(v):
        vx = v.get("vx")
        if vx is None or vx.size == 0:
            return 0.0
        return np.sqrt(np.mean(v["vx"])**2 + np.mean(v["vy"])**2 + np.mean(v["vz"])**2)
    
    B_true = bulk_flow_magnitude(velocity_true)
    B_rec = bulk_flow_magnitude(velocity_rec)
    
    return {
        "B_true": float(B_true),
        "B_rec": float(B_rec),
        "difference": float(abs(B_true - B_rec)),
        "tolerance": 50.0
    }
