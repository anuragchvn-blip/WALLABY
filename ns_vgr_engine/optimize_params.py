"""
Parameter Optimization for NS-VGR.

Tests different parameter combinations to maximize correlation improvement.
"""

import sys
sys.path.insert(0, 'e:/WALLABY')

import numpy as np
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


def test_parameters(ra, dec, distance, distance_error, v_pec_obs,
                   delta_crit, gamma, l_nl, smoothing_sigma):
    """Test a specific parameter combination."""
    config_cosmo = CosmologyConfig()
    config_grid = GridConfig()
    config_grid.smoothing_sigma = smoothing_sigma
    config_algo = AlgorithmConfig()
    config_algo.ns_vgr_delta_crit = delta_crit
    config_algo.ns_vgr_gamma = gamma
    config_algo.ns_vgr_l_nl = l_nl
    
    reconstructor = NSVGRReconstructor(config_cosmo, config_grid, config_algo)
    
    # Baseline
    result_baseline = reconstructor.reconstruct(
        ra, dec, distance, v_pec_obs, distance_error, use_ns_vgr=False
    )
    r_baseline = pearson_correlation(v_pec_obs, result_baseline['v_pred'])
    
    # NS-VGR
    result_ns_vgr = reconstructor.reconstruct(
        ra, dec, distance, v_pec_obs, distance_error, use_ns_vgr=True
    )
    r_ns_vgr = pearson_correlation(v_pec_obs, result_ns_vgr['v_pred'])
    
    return r_baseline, r_ns_vgr


def optimize_parameters():
    """Grid search for optimal parameters."""
    print("=" * 70)
    print("NS-VGR PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Load data
    print("\nLoading CosmicFlows-4 catalog...")
    cf4_path = Path("e:/WALLABY/data/cosmicflows4.csv")
    catalog = read_cosmicflows4(cf4_path)
    
    config_cosmo = CosmologyConfig()
    v_pec = compute_peculiar_velocity(catalog.cz, catalog.distance, H0=config_cosmo.H0)
    
    # Filter valid
    valid = (
        np.isfinite(catalog.ra) & np.isfinite(catalog.dec) &
        np.isfinite(catalog.distance) & np.isfinite(v_pec) &
        (catalog.distance > 1.0) & (catalog.distance < 200.0)
    )
    
    ra = catalog.ra[valid]
    dec = catalog.dec[valid]
    distance = catalog.distance[valid]
    distance_error = catalog.distance_error[valid]
    v_pec_obs = v_pec[valid]
    
    print(f"Valid galaxies: {len(ra)}\n")
    
    # Parameter grid
    delta_crit_vals = [1.2, 1.4, 1.68, 2.0, 2.5]
    gamma_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    l_nl_vals = [3.0, 4.0, 5.0, 6.0, 8.0]
    smoothing_vals = [3.0, 4.0, 5.0, 6.0]
    
    best_improvement = -999
    best_params = None
    best_r_ns = 0
    best_r_base = 0
    
    total_tests = len(delta_crit_vals) * len(gamma_vals) * len(l_nl_vals) * len(smoothing_vals)
    print(f"Testing {total_tests} parameter combinations...\n")
    
    test_count = 0
    
    for smoothing in smoothing_vals:
        for delta_crit in delta_crit_vals:
            for gamma in gamma_vals:
                for l_nl in l_nl_vals:
                    test_count += 1
                    
                    try:
                        r_base, r_ns = test_parameters(
                            ra, dec, distance, distance_error, v_pec_obs,
                            delta_crit, gamma, l_nl, smoothing
                        )
                        
                        improvement = r_ns - r_base
                        
                        if test_count % 20 == 0:
                            print(f"[{test_count}/{total_tests}] σ={smoothing}, δc={delta_crit}, γ={gamma}, L={l_nl}: "
                                  f"r_base={r_base:.4f}, r_ns={r_ns:.4f}, Δr={improvement:.4f}")
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_params = (delta_crit, gamma, l_nl, smoothing)
                            best_r_ns = r_ns
                            best_r_base = r_base
                            print(f"  ★ NEW BEST: Δr = {improvement:.4f}, r_ns = {r_ns:.4f}")
                            
                    except Exception as e:
                        print(f"  Error with params: {e}")
                        continue
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nBest parameters:")
    print(f"  δ_crit = {best_params[0]}")
    print(f"  γ = {best_params[1]}")
    print(f"  L_NL = {best_params[2]}")
    print(f"  smoothing_σ = {best_params[3]}")
    print(f"\nPerformance:")
    print(f"  Baseline r = {best_r_base:.4f}")
    print(f"  NS-VGR r = {best_r_ns:.4f}")
    print(f"  Improvement Δr = {best_improvement:.4f}")
    
    if best_improvement >= 0.10:
        print("\n✅ SUCCESS: Δr >= 0.10 achieved!")
    else:
        print(f"\n❌ Target Δr >= 0.10 not reached (got {best_improvement:.4f})")
    
    # Update config.py recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDED config.py UPDATE:")
    print("=" * 70)
    print(f"""
In velocity_reconstruction/config.py, update:

@dataclass
class GridConfig:
    smoothing_sigma: float = {best_params[3]}  # Optimized from {GridConfig().smoothing_sigma}

@dataclass
class AlgorithmConfig:
    ns_vgr_delta_crit: float = {best_params[0]}  # Optimized from {AlgorithmConfig().ns_vgr_delta_crit}
    ns_vgr_gamma: float = {best_params[1]}  # Optimized from {AlgorithmConfig().ns_vgr_gamma}
    ns_vgr_l_nl: float = {best_params[2]}  # Optimized from {AlgorithmConfig().ns_vgr_l_nl}
""")
    
    return best_params, best_improvement


if __name__ == "__main__":
    best_params, improvement = optimize_parameters()
