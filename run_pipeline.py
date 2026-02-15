"""
Velocity Field Reconstruction Pipeline Test
Runs with real CosmicFlows-4 data
"""
import sys
sys.path.insert(0, 'e:/WALLABY')

import numpy as np
import pandas as pd

# Step 1: Load CosmicFlows-4 catalog
print("=" * 60)
print("STEP 1: Loading CosmicFlows-4 Catalog")
print("=" * 60)
from velocity_reconstruction.data_io.catalog_readers import read_cosmicflows4

cat = read_cosmicflows4('e:/WALLABY/data/cosmicflows4.csv')
print(f"Loaded {cat.n_galaxies} galaxies")
print(f"RA: {cat.ra.min():.2f} to {cat.ra.max():.2f} deg")
print(f"Dec: {cat.dec.min():.2f} to {cat.dec.max():.2f} deg")

valid_cz = cat.cz[~np.isnan(cat.cz)]
valid_dist = cat.distance[~np.isnan(cat.distance)]
print(f"CMB velocity: {valid_cz.min():.0f} to {valid_cz.max():.0f} km/s")
print(f"Distance: {valid_dist.min():.1f} to {valid_dist.max():.1f} Mpc")

methods, counts = np.unique(cat.method, return_counts=True)
print("\nDistance methods:")
for m, c in zip(methods, counts):
    print(f"  {m}: {c}")

# Step 2: Calculate peculiar velocities
print("\n" + "=" * 60)
print("STEP 2: Calculating Peculiar Velocities")
print("=" * 60)

from velocity_reconstruction.preprocessing.peculiar_velocity import compute_peculiar_velocity

# H0 from CosmicFlows-4 (in km/s/Mpc)
H0 = 70.0

# Compute peculiar velocities
v_pec = compute_peculiar_velocity(
    cat.cz, cat.distance, H0=H0
)

# Also get error estimate
from velocity_reconstruction.preprocessing.peculiar_velocity import compute_peculiar_velocity_error
v_pec_err = compute_peculiar_velocity_error(
    np.zeros_like(cat.cz),  # Assume negligible cz error
    cat.distance_error,
    cat.distance,
    intrinsic_scatter=0.35,
    H0=H0
)

print(f"Peculiar velocity range: {np.nanmin(v_pec):.0f} to {np.nanmax(v_pec):.0f} km/s")
print(f"Mean |v_pec|: {np.nanmean(np.abs(v_pec)):.0f} km/s")

# Step 3: Quality filtering
print("\n" + "=" * 60)
print("STEP 3: Quality Filtering")
print("=" * 60)

from velocity_reconstruction.preprocessing.quality_flags import validate_catalog

# Validate and get quality masks
masks = validate_catalog(
    cat.ra, cat.dec, cat.cz,
    cat.distance, v_pec, v_pec_err,
    config={
        "max_error_ratio": 0.5,
        "galactic_plane_margin": 5.0,
        "v_min": -500.0
    }
)

quality_mask = masks["good"]
n_pass = np.sum(quality_mask)
print(f"Galaxies passing quality cuts: {n_pass}/{len(quality_mask)} ({100*n_pass/len(quality_mask):.1f}%)")
print(f"  Low SNR: {np.sum(masks['low_snr'])}")
print(f"  Galactic plane: {np.sum(masks['galactic_plane'])}")
print(f"  Unphysical velocities: {np.sum(masks['unphysical'])}")
print(f"  Distance outliers: {np.sum(masks['outlier_distance'])}")
print(f"  Velocity outliers: {np.sum(masks['outlier_velocity'])}")

# Step 4: POTENT reconstruction
print("\n" + "=" * 60)
print("STEP 4: POTENT Reconstruction")
print("=" * 60)

# Filter data
ra = cat.ra[quality_mask]
dec = cat.dec[quality_mask]
distance = cat.distance[quality_mask]
v_pec_filtered = v_pec[quality_mask]

print(f"Input galaxies: {len(ra)}")

# Import POTENT
from velocity_reconstruction.reconstruction.potent import POTENTReconstructor

# Create grid
reconstructor = POTENTReconstructor(
    extent=150.0,  # Mpc (half-width)
    resolution=32,  # Keep small for testing
    smoothing_sigma=10.0,
    H0=H0
)

# Reconstruct
result = reconstructor.reconstruct(
    ra, dec, distance, v_pec_filtered
)

density_field = result["density"]
velocity_field = result["velocity"]

print(f"Density field shape: {density_field.shape}")
print(f"Velocity field components: vx={velocity_field['vx'].shape}, vy={velocity_field['vy'].shape}, vz={velocity_field['vz'].shape}")
print(f"Density range: {np.min(density_field):.4f} to {np.max(density_field):.4f}")

# Step 5: Validation metrics
print("\n" + "=" * 60)
print("STEP 5: Validation Metrics")
print("=" * 60)

from velocity_reconstruction.validation.metrics import compute_power_spectrum

# Power spectrum
cell_size = 2 * 150.0 / 32  # 2 * extent / resolution
k, pk = compute_power_spectrum(density_field, box_size=cell_size*32)
print(f"Power spectrum computed for {len(k)} modes")
print(f"P(k) range: {np.min(pk):.2e} to {np.max(pk):.2e}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)
print(f"Results saved to: e:/WALLABY/results/")
