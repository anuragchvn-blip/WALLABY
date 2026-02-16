"""
Mock catalog generation for validation.

Creates realistic galaxy surveys from Gaussian Random Fields with known
ground truth for correlation testing.
"""

import numpy as np
from typing import Dict, Tuple

def generate_mock_survey(n_galaxies: int = 5000, 
                         box_size: float = 200.0,
                         resolution: int = 128,
                         sigma_v: float = 200.0,
                         seed: int = 42) -> Dict:
    """
    Generate a mock galaxy catalog with ground truth.
    
    1. Generate a density field (Gaussian Random Field).
    2. Compute velocity field via Linear Theory.
    3. Sample galaxy positions weighted by density.
    4. Assign peculiar velocities (ground truth + noise).
    """
    np.random.seed(seed)
    
    # 1. Density field (truncated Power Spectrum proxy)
    k = np.fft.fftfreq(resolution) * resolution
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[0, 0, 0] = 1.0 # avoid div by zero
    
    # P(k) ~ k^-3 for simplicity
    pk_sqrt = 1.0 / (k_mag**1.5 + 0.1)
    phase = np.exp(2j * np.pi * np.random.rand(resolution, resolution, resolution))
    delta_k = pk_sqrt * phase
    delta = np.real(np.fft.ifftn(delta_k))
    delta = (delta - np.mean(delta)) / np.std(delta) * 0.2 # scale to reasonable delta
    
    # 2. Velocity field (v approx grad(delta)/k^2 in k-space)
    # in Fourier space: v_k = i * f * H0 * (k/k^2) * delta_k
    vx_k = 1j * delta_k * kx / (k_mag**2 + 0.1)
    vy_k = 1j * delta_k * ky / (k_mag**2 + 0.1)
    vz_k = 1j * delta_k * kz / (k_mag**2 + 0.1)
    
    vx_true = np.real(np.fft.ifftn(vx_k))
    vy_true = np.real(np.fft.ifftn(vy_k))
    vz_true = np.real(np.fft.ifftn(vz_k))
    # Normalize
    vx_true *= sigma_v / np.std(vx_true)
    vy_true *= sigma_v / np.std(vy_true)
    vz_true *= sigma_v / np.std(vz_true)
    
    # 3. Sample galaxy positions
    # (Simplified: uniform for now, but weighted by (1+delta) is better)
    prob = (1.0 + delta).flatten()
    prob = np.maximum(prob, 0)
    prob /= np.sum(prob)
    
    grid_coords = np.linspace(-box_size/2, box_size/2, resolution)
    X, Y, Z = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
    all_coords = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    idx = np.random.choice(len(all_coords), size=n_galaxies, p=prob)
    gal_coords = all_coords[idx]
    
    # Interpolate true velocities at galaxy positions
    # (Simplified: take nearest grid point for performance)
    ix = ((gal_coords[:, 0] + box_size/2) / (box_size/resolution)).astype(int) % resolution
    iy = ((gal_coords[:, 1] + box_size/2) / (box_size/resolution)).astype(int) % resolution
    iz = ((gal_coords[:, 2] + box_size/2) / (box_size/resolution)).astype(int) % resolution
    
    gv_x = vx_true[ix, iy, iz]
    gv_y = vy_true[ix, iy, iz]
    gv_z = vz_true[ix, iy, iz]
    
    # Observed radial velocity
    dist = np.sqrt(np.sum(gal_coords**2, axis=1))
    radial_unit = gal_coords / dist[:, np.newaxis]
    v_pec_true = (gv_x * radial_unit[:, 0] + gv_y * radial_unit[:, 1] + gv_z * radial_unit[:, 2])
    v_pec_obs = v_pec_true + np.random.randn(n_galaxies) * 150.0 # add measurement noise
    
    return {
        "coords": gal_coords,
        "dist": dist,
        "v_pec_obs": v_pec_obs,
        "v_pec_true": v_pec_true,
        "true_vx": vx_true,
        "true_vy": vy_true,
        "true_vz": vz_true,
        "true_delta": delta
    }
