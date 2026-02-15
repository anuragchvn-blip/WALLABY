"""Debug closure test - find the conversion bug."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
H0 = 70.0
OMEGA_M = 0.3
f_growth = OMEGA_M ** 0.55
COEFF = H0 / f_growth  # = 70/0.55 = 127.27

print("="*60)
print("CLOSURE TEST DEBUGGING")
print("="*60)

# Load velocity field
vx = np.load('e:/WALLABY/results/velocity_vx.npy')
vy = np.load('e:/WALLABY/results/velocity_vy.npy')
vz = np.load('e:/WALLABY/results/velocity_vz.npy')
density = np.load('e:/WALLABY/results/density.npy')

extent = 250.0
nx, ny, nz = vx.shape
cell = 2 * extent / nx
print(f"\nGrid: {nx}^3, Box: {extent*2} Mpc, Cell: {cell:.2f} Mpc")

# ============================================================
# STEP 1: CHECK ROUND-TRIP CONVERSION
# ============================================================
print("\n" + "="*60)
print("STEP 1: ROUND-TRIP CONVERSION TEST")
print("="*60)

# Test velocity field at center
v_test = np.array([vx[nx//2, ny//2, nz//2], 
                   vy[nx//2, ny//2, nz//2], 
                   vz[nx//2, ny//2, nz//2]])
print(f"\nVelocity at center: {v_test} km/s")

# Convert to gravity
g_test = COEFF * v_test
print(f"Gravity (g = H0/f * v): {g_test} km/s/Mpc")
print(f"|g| = {np.linalg.norm(g_test):.1f} km/s/Mpc")

# Convert back
v_recovered = g_test / COEFF
print(f"Recovered velocity: {v_recovered} km/s")
print(f"Round-trip works: {np.allclose(v_test, v_recovered)}")

# ============================================================
# STEP 2: LOAD GALAXY DATA
# ============================================================
print("\n" + "="*60)
print("STEP 2: LOAD GALAXY DATA")
print("="*60)

import warnings
warnings.filterwarnings('ignore')

from velocity_reconstruction.data_io.catalog_readers import read_cosmicflows4
from velocity_reconstruction.preprocessing.malmquist import apply_malmquist_correction
from velocity_reconstruction.preprocessing.peculiar_velocity import compute_peculiar_velocity
from velocity_reconstruction.preprocessing.quality_flags import validate_catalog

cat = read_cosmicflows4('e:/WALLABY/data/cosmicflows4.csv')

# Process distances
distance_corr = cat.distance.copy()
for method in np.unique(cat.method):
    if method == 'Unknown':
        continue
    mask = cat.method == method
    mag_lim = {"TF": 14.5, "FP": 16.0, "SN Ia": 17.5}.get(method, 15.0)
    distance_corr[mask] = apply_malmquist_correction(
        cat.distance[mask], cat.distance_error[mask], method=method, magnitude_limit=mag_lim
    )

# Quality filter
valid_mask = (distance_corr > 1.5) & (distance_corr < 200) & np.isfinite(distance_corr)
v_pec = np.full(len(cat.cz), np.nan, dtype=np.float64)
v_pec[valid_mask] = compute_peculiar_velocity(cat.cz[valid_mask], distance_corr[valid_mask], H0=H0)

v_pec_err = np.full_like(v_pec, 300.0)
masks = validate_catalog(cat.ra, cat.cz, cat.cz, distance_corr, v_pec, v_pec_err,
    config={"max_error_ratio": 2.0, "galactic_plane_margin": 3.0, "v_min": -4000, "v_max": 5000})
good = masks["good"] & np.isfinite(v_pec) & (distance_corr > 1.5) & (distance_corr < 200) & (np.abs(cat.dec) > 3)

# ============================================================
# STEP 3: COORDINATE CONVERSION - THE CRITICAL PART
# ============================================================
print("\n" + "="*60)
print("STEP 3: COORDINATE CONVERSION")
print("="*60)

# Galaxy coordinates in degrees
ra = cat.ra[good]
dec = cat.dec[good]
dist = distance_corr[good]
v_obs = v_pec[good]

# Convert RA/Dec to Cartesian - need supergalactic coordinates!
# The grid uses supergalactic coordinates centered at (0,0,0)
# RA/Dec are in equatorial (J2000)

# First convert to galactic, then to supergalactic
# Simplified: use supergalactic projection

def equatorial_to_supergalactic(ra_deg, dec_deg):
    """Convert RA/Dec (deg) to supergalactic coordinates."""
    # North Galactic Pole in J2000: RA=192.85948, Dec=27.12825
    # Galactic anti-center direction
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    
    # Simple approximation: treat RA/Dec as if origin is at galactic center
    # This is NOT precise but gives relative positions
    
    # Convert to 3D Cartesian on unit sphere
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    return x, y, z

# Get unit vectors
x_unit, y_unit, z_unit = equatorial_to_supergalactic(ra, dec)

# Scale by distance to get positions in Mpc
x_gal = x_unit * dist
y_gal = y_unit * dist  
z_gal = z_unit * dist

print(f"\nGalaxy positions (first 5):")
for i in range(5):
    print(f"  Galaxy {i}: ({x_gal[i]:.1f}, {y_gal[i]:.1f}, {z_gal[i]:.1f}) Mpc, v_obs={v_obs[i]:.0f} km/s")

# Check if galaxies are within our grid
in_bounds = (np.abs(x_gal) < extent) & (np.abs(y_gal) < extent) & (np.abs(z_gal) < extent)
print(f"\nGalaxies within grid: {in_bounds.sum()} / {len(x_gal)}")

# ============================================================
# STEP 4: INTERPOLATE g FIELD
# ============================================================
print("\n" + "="*60)
print("STEP 4: INTERPOLATE g FIELD")
print("="*60)

# Compute gravity field
g_x = COEFF * vx
g_y = COEFF * vy
g_z = COEFF * vz
g_mag = np.sqrt(g_x**2 + g_y**2 + g_z**2)

print(f"|g| range: {np.nanmin(g_mag):.0f} to {np.nanmax(g_mag):.0f} km/s/Mpc")

# Create interpolators
from scipy.interpolate import RegularGridInterpolator

x_grid = np.linspace(-extent, extent, nx)
y_grid = np.linspace(-extent, extent, ny)
z_grid = np.linspace(-extent, extent, nz)

interp_gx = RegularGridInterpolator((x_grid, y_grid, z_grid), g_x, bounds_error=False, fill_value=np.nan)
interp_gy = RegularGridInterpolator((x_grid, y_grid, z_grid), g_y, bounds_error=False, fill_value=np.nan)
interp_gz = RegularGridInterpolator((x_grid, y_grid, z_grid), g_z, bounds_error=False, fill_value=np.nan)

# ============================================================
# STEP 5: MANUAL GALAXY CHECK
# ============================================================
print("\n" + "="*60)
print("STEP 5: MANUAL GALAXY CHECK")
print("="*60)

# Check first 10 galaxies in detail
n_check = 10
print(f"\n{'Galaxy':>6} {'Pos':>25} {'v_obs':>10} {'g_interp':>30} {'g_rad':>12} {'v_pred':>10} {'Ratio':>8}")
print("-"*110)

for i in range(n_check):
    pos = np.array([x_gal[i], y_gal[i], z_gal[i]])
    
    if np.any(np.abs(pos) >= extent):
        print(f"{i:6d} {'(outside grid)':>25} {v_obs[i]:>10.0f}")
        continue
    
    # Interpolate g
    g_vec = np.array([interp_gx(pos)[0], interp_gy(pos)[0], interp_gz(pos)[0]])
    
    # Line of sight direction
    r = np.linalg.norm(pos)
    if r < 0.1:
        print(f"{i:6d} {'(too close)':>25} {v_obs[i]:>10.0f}")
        continue
    
    los = pos / r
    g_radial = np.dot(g_vec, los)
    
    # Predict velocity: v = g_radial / (H0/f) = g_radial / 127
    v_pred = g_radial / COEFF
    
    ratio = v_pred / v_obs[i] if v_obs[i] != 0 else np.nan
    
    print(f"{i:6d} ({pos[0]:7.1f},{pos[1]:7.1f},{pos[2]:7.1f}) {v_obs[i]:>10.0f} ({g_vec[0]:>8.1f},{g_vec[1]:>8.1f},{g_vec[2]:>8.1f}) {g_radial:>12.1f} {v_pred:>10.0f} {ratio:>8.2f}")

# ============================================================
# STEP 6: FULL CLOSURE TEST
# ============================================================
print("\n" + "="*60)
print("STEP 6: FULL CLOSURE TEST")
print("="*60)

# Predict for all galaxies
v_pred = np.full(len(x_gal), np.nan)

for i in range(len(x_gal)):
    pos = np.array([x_gal[i], y_gal[i], z_gal[i]])
    
    if np.any(np.abs(pos) >= extent):
        continue
    
    r = np.linalg.norm(pos)
    if r < 0.1:
        continue
    
    g_vec = np.array([interp_gx(pos)[0], interp_gy(pos)[0], interp_gz(pos)[0]])
    los = pos / r
    g_radial = np.dot(g_vec, los)
    v_pred[i] = g_radial / COEFF

# Correlation
valid = np.isfinite(v_pred) & np.isfinite(v_obs)
n_valid = valid.sum()
print(f"\nValid predictions: {n_valid} / {len(v_obs)}")

if n_valid > 100:
    v_obs_valid = v_obs[valid]
    v_pred_valid = v_pred[valid]
    
    r = np.corrcoef(v_obs_valid, v_pred_valid)[0, 1]
    residual = v_obs_valid - v_pred_valid
    rms = np.sqrt(np.nanmean(residual**2))
    bias = np.nanmean(residual)
    
    print(f"\n*** CLOSURE TEST RESULTS ***")
    print(f"Correlation r = {r:.4f}")
    print(f"RMS residual = {rms:.0f} km/s")
    print(f"Bias = {bias:.0f} km/s")
    print(f"v_obs mean = {np.mean(v_obs_valid):.0f} km/s")
    print(f"v_pred mean = {np.mean(v_pred_valid):.0f} km/s")
    print(f"Ratio of means = {np.mean(v_pred_valid)/np.mean(v_obs_valid):.2f}")
    
    if np.mean(v_obs_valid) > 0:
        print(f"\n>>> If ratio is ~127 or ~0.008, conversion coefficient is WRONG!")
    
    # ============================================================
    # STEP 7: SCATTER PLOT
    # ============================================================
    plt.figure(figsize=(10, 10))
    plt.scatter(v_obs_valid, v_pred_valid, alpha=0.2, s=3)
    plt.plot([-2000, 2000], [-2000, 2000], 'r--', lw=2, label='1:1')
    plt.plot([-2000, 2000], [0, 0], 'k-', lw=0.5, alpha=0.3)
    plt.plot([0, 0], [-2000, 2000], 'k-', lw=0.5, alpha=0.3)
    plt.xlabel('Observed v_pec (km/s)', fontsize=12)
    plt.ylabel('Predicted v_pec (km/s)', fontsize=12)
    plt.title(f'Closure Test: r={r:.4f}, RMS={rms:.0f} km/s', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(-2000, 2000)
    plt.ylim(-2000, 2000)
    plt.axis('equal')
    plt.savefig('e:/WALLABY/results/closure_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: results/closure_scatter.png")

# ============================================================
# DIAGNOSIS
# ============================================================
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

# Check if velocities in the field are reasonable
print(f"\nVelocity field statistics:")
print(f"  vx: {np.nanmin(vx):.0f} to {np.nanmax(vx):.0f}, mean={np.nanmean(vx):.0f} km/s")
print(f"  vy: {np.nanmin(vy):.0f} to {np.nanmax(vy):.0f}, mean={np.nanmean(vy):.0f} km/s")
print(f"  vz: {np.nanmin(vz):.0f} to {np.nanmax(vz):.0f}, mean={np.nanmean(vz):.0f} km/s")

print(f"\nGalaxy velocity statistics:")
print(f"  v_obs: {np.nanmin(v_obs):.0f} to {np.nanmax(v_obs):.0f}, mean={np.nanmean(v_obs):.0f} km/s")

print(f"\n>>> The issue is likely:")
print(f"    1. Coordinate mismatch (galaxies not in same system as grid)")
print(f"    2. OR the velocity field itself has issues")
print(f"    3. OR we need to check the POTENT reconstruction")
