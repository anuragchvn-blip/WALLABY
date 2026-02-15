"""Fixed closure test with proper coordinate transformation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H0 = 70.0
OMEGA_M = 0.3
f_growth = OMEGA_M ** 0.55
COEFF = H0 / f_growth  # 127.27

print("="*60)
print("FIXED CLOSURE TEST")
print("="*60)

# Load velocity field
vx = np.load('e:/WALLABY/results/velocity_vx.npy')
vy = np.load('e:/WALLABY/results/velocity_vy.npy')
vz = np.load('e:/WALLABY/results/velocity_vz.npy')
density = np.load('e:/WALLABY/results/density.npy')

extent = 250.0
nx, ny, nz = vx.shape
cell = 2 * extent / nx

print(f"Grid: {nx}^3, Box: {extent*2} Mpc")

# ============================================================
# KEY INSIGHT: The velocity field is in SUPERGALACTIC coords
# The galaxies are in EQUATORIAL coords (RA/Dec)
# We need to transform RA/Dec → Supergalactic properly
# ============================================================

# Load galaxy data
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

# THE FIX: Use proper supergalactic coordinate transformation
# Supergalactic plane: SGL=0, SGB=0 is galactic plane
# North Super Galactic Pole: RA=283.8°, Dec=+15.7° (J2000)

SGP_RA = 283.8  # deg
SGP_DEC = 15.7  # deg

def eq_to_sg(ra_deg, dec_deg):
    """Transform equatorial to supergalactic coordinates."""
    # Rotation matrix from equatorial to supergalactic
    # Based on Lahav et al. (2000)
    
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    
    # Cartesian in equatorial
    x_eq = np.cos(dec_rad) * np.cos(ra_rad)
    y_eq = np.cos(dec_rad) * np.sin(ra_rad)
    z_eq = np.sin(dec_rad)
    
    # Rotate to supergalactic (reverse of the supergalactic transformation)
    # SGP in equatorial coords
    sgp_ra_rad = np.deg2rad(SGP_RA)
    sgp_dec_rad = np.deg2rad(SGP_DEC)
    
    # Rotation angles
    # First rotate around x-axis by -(90 - SGP_DEC)
    # Then rotate around z-axis by -SGP_RA
    
    # Simplified: use direct formula
    # Supergalactic latitude = arcsin(sin(dec)*sin(SGP_DEC) + cos(dec)*cos(SGP_DEC)*cos(ra-SGP_RA))
    # Supergalactic longitude = atan2(cos(dec)*sin(ra-SGP_RA), cos(dec)*cos(ra-SGP_RA)*cos(SGP_DEC) - sin(dec)*sin(SGP_DEC))
    
    sgb = np.arcsin(np.sin(dec_rad) * np.sin(sgp_dec_rad) + 
                    np.cos(dec_rad) * np.cos(sgp_dec_rad) * np.cos(ra_rad - sgp_ra_rad))
    
    sgl = np.arctan2(np.cos(dec_rad) * np.sin(ra_rad - sgp_ra_rad),
                     np.cos(dec_rad) * np.cos(ra_rad - sgp_ra_rad) * np.cos(sgp_dec_rad) - 
                     np.sin(dec_rad) * np.sin(sgp_dec_rad))
    
    return np.rad2deg(sgl), np.rad2deg(sgb)

# Get supergalactic coordinates for galaxies
sgl, sgb = eq_to_sg(cat.ra, cat.dec)
dist = distance_corr

# Convert to Cartesian (supergalactic)
x_gal = dist * np.cos(np.deg2rad(sgb)) * np.cos(np.deg2rad(sgl))
y_gal = dist * np.cos(np.deg2rad(sgb)) * np.sin(np.deg2rad(sgl))
z_gal = dist * np.sin(np.deg2rad(sgb))

# Apply filter
x_gal = x_gal[good]
y_gal = y_gal[good]
z_gal = z_gal[good]
v_obs = v_pec[good]
n_gal = len(x_gal)

print(f"Galaxies: {n_gal}")
print(f"Sample SG positions: ({x_gal[0]:.1f}, {y_gal[0]:.1f}, {z_gal[0]:.1f}) Mpc")

# Interpolate velocity field
from scipy.interpolate import RegularGridInterpolator

x_grid = np.linspace(-extent, extent, nx)
y_grid = np.linspace(-extent, extent, ny)
z_grid = np.linspace(-extent, extent, nz)

# Note: transpose for correct axis ordering
vx_T = np.transpose(vx, (2, 1, 0))
vy_T = np.transpose(vy, (2, 1, 0))
vz_T = np.transpose(vz, (2, 1, 0))

interp_vx = RegularGridInterpolator((x_grid, y_grid, z_grid), vx_T, bounds_error=False, fill_value=np.nan)
interp_vy = RegularGridInterpolator((x_grid, y_grid, z_grid), vy_T, bounds_error=False, fill_value=np.nan)
interp_vz = RegularGridInterpolator((x_grid, y_grid, z_grid), vz_T, bounds_error=False, fill_value=np.nan)

# Predict velocities at galaxy positions
v_pred = np.full(n_gal, np.nan)
v_pred_3d = np.full((n_gal, 3), np.nan)

for i in range(n_gal):
    pos = np.array([x_gal[i], y_gal[i], z_gal[i]])
    
    if np.all(np.abs(pos) < extent):
        v_3d = np.array([interp_vx(pos)[0], interp_vy(pos)[0], interp_vz(pos)[0]])
        v_pred_3d[i] = v_3d
        
        # Project onto line of sight
        r = np.linalg.norm(pos)
        if r > 0.1:
            los = pos / r
            v_pred[i] = np.dot(v_3d, los)

# Compute correlation
valid = np.isfinite(v_pred) & np.isfinite(v_obs)
n_valid = valid.sum()
print(f"Valid: {n_valid}")

if n_valid > 100:
    v_obs_v = v_obs[valid]
    v_pred_v = v_pred[valid]
    
    r = np.corrcoef(v_obs_v, v_pred_v)[0, 1]
    residual = v_obs_v - v_pred_v
    rms = np.sqrt(np.nanmean(residual**2))
    bias = np.nanmean(residual)
    
    print(f"\n{'='*50}")
    print("CLOSURE TEST RESULTS (FIXED COORDINATES)")
    print(f"{'='*50}")
    print(f"r = {r:.4f}")
    print(f"RMS = {rms:.0f} km/s")
    print(f"Bias = {bias:.0f} km/s")
    print(f"v_obs mean = {np.mean(v_obs_v):.0f} km/s")
    print(f"v_pred mean = {np.mean(v_pred_v):.0f} km/s")
    
    # Scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(v_obs_v, v_pred_v, alpha=0.2, s=3)
    plt.plot([-2000, 2000], [-2000, 2000], 'r--', lw=2, label='1:1')
    plt.xlabel('Observed v_pec (km/s)', fontsize=12)
    plt.ylabel('Predicted v_pec (km/s)', fontsize=12)
    plt.title(f'Closure Test (Fixed): r={r:.4f}, RMS={rms:.0f} km/s', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-2000, 2000)
    plt.ylim(-2000, 2000)
    plt.axis('equal')
    plt.savefig('e:/WALLABY/results/closure_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: results/closure_scatter.png")
    
    if r > 0.7:
        print("Status: VALIDATED!")
    elif r > 0.5:
        print("Status: MODERATE")
    else:
        print("Status: NEEDS WORK")

# Save diagnostic info
print(f"\nVelocity field sample (center):")
print(f"  v = ({vx[nx//2,ny//2,nz//2]:.1f}, {vy[nx//2,ny//2,nz//2]:.1f}, {vz[nx//2,ny//2,nz//2]:.1f}) km/s")
print(f"Galaxy sample (first):")
print(f"  v_obs = {v_obs[0]:.1f} km/s")
print(f"  position = ({x_gal[0]:.1f}, {y_gal[0]:.1f}, {z_gal[0]:.1f}) Mpc")
print(f"  v_pred (3D) = {v_pred_3d[0]} km/s")
