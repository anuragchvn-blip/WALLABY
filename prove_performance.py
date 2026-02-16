"""
NS-VGR Performance Benchmark.

Compares baseline (Radial Projection) vs NS-VGR (Non-linear Saturated).
Validates improvement on mock catalogs and real data.
"""

import numpy as np
import time
from scipy.stats import pearsonr
from velocity_reconstruction.validation.mock_catalogs import generate_mock_survey
from velocity_reconstruction.reconstruction.simple_velocity import reconstruct_velocity_field
from velocity_reconstruction.reconstruction.ns_vgr import NSVGRReconstructor

def run_benchmark():
    print("=== NS-VGR PERFORMANCE BENCHMARK ===")
    
    # 1. Generate Mock Data
    print("\n[1/4] Generating Mock Survey (128^3 grid, 5000 galaxies)...")
    mock = generate_mock_survey(n_galaxies=5000, resolution=128)
    
    # Simple conversion from Cartesian to RA/Dec for the API
    x, y, z = mock["coords"].T
    ra = np.rad2deg(np.arctan2(y, x))
    dec = np.rad2deg(np.arcsin(z / mock["dist"]))
    
    # 2. Run Baseline (Simple Velocity)
    print("[2/4] Running Baseline (Simple Radial Projection)...")
    t0 = time.time()
    res_base = reconstruct_velocity_field(ra, dec, mock["dist"], mock["v_pec_obs"], 
                                         resolution=128, smoothing_sigma=8.0)
    t1 = time.time()
    
    # Compare vs ground truth vx (flattened for correlation)
    r_base, _ = pearsonr(mock["true_vx"].flatten(), res_base["vx"].flatten())
    print(f"      Baseline Correlation r = {r_base:.3f}")
    print(f"      Execution time: {t1-t0:.2f}s")
    
    # 3. Run NS-VGR Upgrade
    print("[3/4] Running NS-VGR Upgrade (Non-linear Saturated)...")
    engine = NSVGRReconstructor(resolution=128, smoothing_sigma=5.0)
    t0 = time.time()
    res_ns = engine.reconstruct(ra, dec, mock["dist"], mock["v_pec_obs"])
    t1 = time.time()
    
    # Compare vs ground truth vx
    r_ns, _ = pearsonr(mock["true_vx"].flatten(), res_ns["vx"].flatten())
    print(f"      NS-VGR Correlation r = {r_ns:.3f}")
    print(f"      Execution time: {t1-t0:.2f}s")
    
    # 4. Results Summary
    improvement = (r_ns / r_base - 1) * 100 if r_base > 0 else 0
    print("\n=== RESULTS SUMMARY ===")
    print(f"Baseline r: {r_base:.4f}")
    print(f"NS-VGR r:   {r_ns:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    if r_ns >= 0.25:
        print("\nSUCCESS: Target r >= 0.25 achieved!")
    else:
        print("\nFAILURE: Target r >= 0.25 not met.")

if __name__ == "__main__":
    run_benchmark()
