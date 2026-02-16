"""
Real Data Benchmark: NS-VGR on CosmicFlows-4.

Tests the NS-VGR formula on real galaxy catalog data and compares
against baseline linear theory.
"""

import sys
sys.path.insert(0, 'e:/WALLABY')

import numpy as np
import time
from pathlib import Path
from formula import NSVGRReconstructor
from velocity_reconstruction.data_io.catalog_readers import read_cosmicflows4
from velocity_reconstruction.preprocessing.peculiar_velocity import compute_peculiar_velocity
from velocity_reconstruction.config import CosmologyConfig, GridConfig, AlgorithmConfig


def pearson_correlation(x, y):
    """Compute Pearson correlation coefficient."""
    x_valid = x[np.isfinite(x) & np.isfinite(y)]
    y_valid = y[np.isfinite(x) & np.isfinite(y)]
    
    if len(x_valid) < 10:
        return 0.0
    
    mx = np.mean(x_valid)
    my = np.mean(y_valid)
    
    cov = np.sum((x_valid - mx) * (y_valid - my))
    std_x = np.sqrt(np.sum((x_valid - mx)**2))
    std_y = np.sqrt(np.sum((y_valid - my)**2))
    
    if std_x == 0 or std_y == 0:
        return 0.0
    
    return cov / (std_x * std_y)


def rms_error(pred, obs):
    """Compute RMS error."""
    diff = pred - obs
    valid = np.isfinite(diff)
    if np.sum(valid) == 0:
        return np.nan
    return np.sqrt(np.mean(diff[valid]**2))


def run_real_data_test():
    """Run NS-VGR benchmark on CosmicFlows-4 catalog."""
    
    print("=" * 60)
    print("NS-VGR REAL DATA BENCHMARK - CosmicFlows-4")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/6] Loading configuration from config.py...")
    config_cosmo = CosmologyConfig()
    config_grid = GridConfig()
    config_algo = AlgorithmConfig()
    
    print(f"  Grid: {config_grid.resolution}³, extent=±{config_grid.extent} Mpc")
    print(f"  Smoothing: σ={config_grid.smoothing_sigma} Mpc")
    print(f"  NS-VGR: δ_crit={config_algo.ns_vgr_delta_crit}, γ={config_algo.ns_vgr_gamma}, L_NL={config_algo.ns_vgr_l_nl}")
    
    # Load CosmicFlows-4 catalog
    print("\n[2/6] Loading CosmicFlows-4 catalog...")
    cf4_path = Path("e:/WALLABY/data/cosmicflows4.csv")
    
    if not cf4_path.exists():
        print(f"ERROR: {cf4_path} not found!")
        return
    
    catalog = read_cosmicflows4(cf4_path)
    print(f"  Loaded {catalog.n_galaxies} galaxies")
    
    # Compute peculiar velocities
    print("\n[3/6] Computing peculiar velocities...")
    v_pec = compute_peculiar_velocity(
        catalog.cz, 
        catalog.distance, 
        H0=config_cosmo.H0
    )
    
    # Filter valid galaxies (finite distances and velocities, within grid)
    valid = (
        np.isfinite(catalog.ra) &
        np.isfinite(catalog.dec) &
        np.isfinite(catalog.distance) &
        np.isfinite(v_pec) &
        (catalog.distance > 1.0) &
        (catalog.distance < 2 * config_grid.extent)
    )
    
    ra = catalog.ra[valid]
    dec = catalog.dec[valid]
    distance = catalog.distance[valid]
    distance_error = catalog.distance_error[valid]
    v_pec_obs = v_pec[valid]
    
    print(f"  Valid galaxies for reconstruction: {len(ra)}")
    print(f"  Distance range: {distance.min():.1f} - {distance.max():.1f} Mpc")
    print(f"  v_pec range: {v_pec_obs.min():.1f} - {v_pec_obs.max():.1f} km/s")
    
    # Initialize reconstructor
    print("\n[4/6] Running BASELINE (Linear Theory)...")
    reconstructor = NSVGRReconstructor(config_cosmo, config_grid, config_algo)
    
    t0 = time.time()
    result_baseline = reconstructor.reconstruct(
        ra, dec, distance, v_pec_obs, distance_error,
        use_ns_vgr=False
    )
    t1 = time.time()
    
    v_pred_baseline = result_baseline['v_pred']
    r_baseline = pearson_correlation(v_pec_obs, v_pred_baseline)
    rms_baseline = rms_error(v_pred_baseline, v_pec_obs)
    
    print(f"  Correlation r = {r_baseline:.4f}")
    print(f"  RMS error = {rms_baseline:.1f} km/s")
    print(f"  Time: {t1-t0:.2f}s")
    
    # Run NS-VGR
    print("\n[5/6] Running NS-VGR (Non-linear Saturated)...")
    
    t0 = time.time()
    result_ns_vgr = reconstructor.reconstruct(
        ra, dec, distance, v_pec_obs, distance_error,
        use_ns_vgr=True
    )
    t1 = time.time()
    
    v_pred_ns_vgr = result_ns_vgr['v_pred']
    r_ns_vgr = pearson_correlation(v_pec_obs, v_pred_ns_vgr)
    rms_ns_vgr = rms_error(v_pred_ns_vgr, v_pec_obs)
    
    print(f"  Correlation r = {r_ns_vgr:.4f}")
    print(f"  RMS error = {rms_ns_vgr:.1f} km/s")
    print(f"  Time: {t1-t0:.2f}s")
    
    # Results summary
    print("\n" + "=" * 60)
    print("[6/6] RESULTS SUMMARY")
    print("=" * 60)
    
    results = {
        "correlation_baseline": float(r_baseline),
        "correlation_ns_vgr": float(r_ns_vgr),
        "improvement_percent": float((r_ns_vgr / max(r_baseline, 0.001) - 1) * 100),
        "rms_error_baseline": float(rms_baseline),
        "rms_error_ns_vgr": float(rms_ns_vgr),
        "n_galaxies_used": int(len(ra)),
        "grid_resolution": int(config_grid.resolution),
        "execution_time_seconds": float(t1 - t0)
    }
    
    print(f"\nBaseline (Linear):    r = {results['correlation_baseline']:.4f}, RMS = {results['rms_error_baseline']:.1f} km/s")
    print(f"NS-VGR (Non-linear):  r = {results['correlation_ns_vgr']:.4f}, RMS = {results['rms_error_ns_vgr']:.1f} km/s")
    print(f"Improvement:          Δr = {results['correlation_ns_vgr'] - results['correlation_baseline']:.4f} ({results['improvement_percent']:+.1f}%)")
    print(f"Galaxies used:        {results['n_galaxies_used']}")
    print(f"Grid resolution:      {results['grid_resolution']}³")
    print(f"Execution time:       {results['execution_time_seconds']:.2f}s")
    
    # Success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    print("=" * 60)
    
    success = True
    
    if results['correlation_ns_vgr'] >= 0.25:
        print("✅ Target correlation r >= 0.25 ACHIEVED")
    else:
        print(f"❌ Target correlation r >= 0.25 NOT MET (r = {results['correlation_ns_vgr']:.4f})")
        success = False
    
    if results['correlation_ns_vgr'] - results['correlation_baseline'] >= 0.10:
        print(f"✅ Improvement Δr >= 0.10 ACHIEVED (Δr = {results['correlation_ns_vgr'] - results['correlation_baseline']:.4f})")
    else:
        print(f"❌ Improvement Δr >= 0.10 NOT MET (Δr = {results['correlation_ns_vgr'] - results['correlation_baseline']:.4f})")
        success = False
    
    if results['execution_time_seconds'] < 60:
        print(f"✅ Execution time < 60s ACHIEVED ({results['execution_time_seconds']:.2f}s)")
    else:
        print(f"❌ Execution time < 60s NOT MET ({results['execution_time_seconds']:.2f}s)")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("OVERALL: ✅ SUCCESS - All criteria met!")
    else:
        print("OVERALL: ❌ FAILURE - Some criteria not met")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_real_data_test()
