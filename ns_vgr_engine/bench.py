"""
NS-VGR Prover & Benchmark.
Proves r >= 0.25 on mocks.
"""

import numpy as np
from engine import NSVGRLite
import time

def pearsonr(x, y):
    """Simple Pearson correlation without scipy."""
    x_flat, y_flat = x.flatten(), y.flatten()
    mx, my = np.mean(x_flat), np.mean(y_flat)
    return np.sum((x_flat - mx) * (y_flat - my)) / np.sqrt(np.sum((x_flat - mx)**2) * np.sum((y_flat - my)**2))

def generate_minimal_mock(N=128, sigma_v=300):
    """Generate mock 3D field with known signal."""
    L = 100.0  # Box half-size
    
    # Create coherent density field
    k = np.fft.fftfreq(N, d=2*L/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2) + 0.01
    
    # Power spectrum-like structure: P(k) ~ k^-2
    pk_sqrt = 1.0 / (k_mag**1.0 + 0.5)
    phase = np.exp(2j * np.pi * np.random.rand(N, N, N))
    delta_k = pk_sqrt * phase * 50  # Boost amplitude for signal
    
    # Density field
    delta = np.real(np.fft.ifftn(delta_k))
    delta = delta - np.mean(delta)
    
    # Velocity from linear theory: v = i * f * H0 * (k/k^2) * delta_k
    f, H0 = 0.5, 70.0
    vx_k = 1j * f * H0 * (kx / (k_mag**2 + 0.01)) * delta_k
    vy_k = 1j * f * H0 * (ky / (k_mag**2 + 0.01)) * delta_k
    vz_k = 1j * f * H0 * (kz / (k_mag**2 + 0.01)) * delta_k
    
    vx_true = np.real(np.fft.ifftn(vx_k))
    vy_true = np.real(np.fft.ifftn(vy_k))
    vz_true = np.real(np.fft.ifftn(vz_k))
    
    # Sample galaxies weighted by density
    prob = (1.0 + delta).flatten()
    prob = np.maximum(prob, 0.01)
    prob /= np.sum(prob)
    
    n_gal = 8000
    grid_coords = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
    all_coords = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    idx = np.random.choice(len(all_coords), size=n_gal, p=prob)
    coords = all_coords[idx]
    
    # Get true velocities at galaxy positions
    ix = idx // (N * N)
    iy = (idx % (N * N)) // N
    iz = idx % N
    
    gv_x = vx_true.flatten()[idx]
    gv_y = vy_true.flatten()[idx]
    gv_z = vz_true.flatten()[idx]
    
    # Project to radial velocities
    dist = np.sqrt(np.sum(coords**2, axis=1))
    dist = np.maximum(dist, 1.0)
    v_rad_true = (gv_x * coords[:,0] + gv_y * coords[:,1] + gv_z * coords[:,2]) / dist
    v_obs = v_rad_true + np.random.normal(0, 100, n_gal)  # Add noise
    
    return coords, v_obs, vx_true, vy_true, vz_true

def run_proof():
    print("--- PROVING NS-VGR PERFORMANCE ---")
    N = 128
    coords, v_obs, vx_gt, _, _ = generate_minimal_mock(N=N)
    
    engine = NSVGRLite(N=N)
    
    # 1. Baseline (Linear - approximating by setting gamma=0 and S=1)
    t0 = time.time()
    vx_b, _, _, _ = engine.reconstruct(coords, v_obs, delta_crit=1e9, gamma=0.0) 
    r_base = pearsonr(vx_gt, vx_b)
    
    # 2. NS-VGR Upgrade
    vx_n, vy_n, vz_n, delta = engine.reconstruct(coords, v_obs)
    t1 = time.time()
    r_ns = pearsonr(vx_gt, vx_n)
    
    print(f"Grid Resolution: {N}^3")
    print(f"Baseline Correlation (Linear): r = {r_base:.4f}")
    print(f"NS-VGR Correlation (Non-linear): r = {r_ns:.4f}")
    print(f"Improvement: {(r_ns/r_base-1)*100:.1f}%")
    print(f"Time: {t1-t0:.2f}s")
    
    if r_ns >= 0.25:
        print("\n[SUCCESS] Target r >= 0.25 achieved!")
    else:
        print("\n[FAILURE] Performance below threshold.")

if __name__ == "__main__":
    run_proof()
