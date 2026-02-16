"""
Quick parameter test to identify promising directions.
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
    x_valid = x[np.isfinite(x) & np.isfinite(y)]
    y_valid = y[np.isfinite(x) & np.isfinite(y)]
    if len(x_valid) < 10:
        return 0.0
    mx, my = np.mean(x_valid), np.mean(y_valid)
    cov = np.sum((x_valid - mx) * (y_valid - my))
    std_x = np.sqrt(np.sum((x_valid - mx)**2))
    std_y = np.sqrt(np.sum((y_valid - my)**2))
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


print("Loading data...")
cf4_path = Path("e:/WALLABY/data/cosmicflows4.csv")
catalog = read_cosmicflows4(cf4_path)

config_cosmo = CosmologyConfig()
v_pec = compute_peculiar_velocity(catalog.cz, catalog.distance, H0=config_cosmo.H0)

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

# Test configurations
configs = [
    ("Current", 1.68, 0.4, 5.0, 5.0),
    ("Lower smoothing", 1.68, 0.4, 5.0, 3.0),
    ("Stronger saturation", 1.2, 0.4, 5.0, 3.0),
    ("Stronger entrainment", 1.2, 0.8, 5.0, 3.0),
    ("Larger L_NL", 1.2, 0.8, 8.0, 3.0),
    ("Aggressive", 1.0, 1.0, 10.0, 3.0),
]

print("Testing parameter configurations:\n")
print(f"{'Config':<25} {'δ_crit':<8} {'γ':<6} {'L_NL':<6} {'σ':<6} {'r_base':<8} {'r_ns':<8} {'Δr':<8}")
print("=" * 90)

for name, delta_crit, gamma, l_nl, smoothing in configs:
    config_grid = GridConfig()
    config_grid.smoothing_sigma = smoothing
    config_algo = AlgorithmConfig()
    config_algo.ns_vgr_delta_crit = delta_crit
    config_algo.ns_vgr_gamma = gamma
    config_algo.ns_vgr_l_nl = l_nl
    
    reconstructor = NSVGRReconstructor(config_cosmo, config_grid, config_algo)
    
    # Baseline
    result_base = reconstructor.reconstruct(ra, dec, distance, v_pec_obs, distance_error, use_ns_vgr=False)
    r_base = pearson_correlation(v_pec_obs, result_base['v_pred'])
    
    # NS-VGR
    result_ns = reconstructor.reconstruct(ra, dec, distance, v_pec_obs, distance_error, use_ns_vgr=True)
    r_ns = pearson_correlation(v_pec_obs, result_ns['v_pred'])
    
    delta_r = r_ns - r_base
    
    status = "✅" if delta_r >= 0.10 else ""
    print(f"{name:<25} {delta_crit:<8.2f} {gamma:<6.2f} {l_nl:<6.1f} {smoothing:<6.1f} {r_base:<8.4f} {r_ns:<8.4f} {delta_r:<8.4f} {status}")

print("\n" + "=" * 90)
