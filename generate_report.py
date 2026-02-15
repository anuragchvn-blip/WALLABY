"""Gravitational Field Analysis with Direct Density Reconstruction (Option C)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter, label
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fftn, ifftn
import warnings
warnings.filterwarnings('ignore')

H0 = 70.0
OMEGA_M = 0.3
f_growth = OMEGA_M ** 0.55

print("="*60)
print("GRAVITATIONAL FIELD ANALYSIS - DIRECT DENSITY")
print("="*60)

# Load density field (for structure detection)
density = np.load('e:/WALLABY/results/density.npy')

extent = 250.0
nx, ny, nz = density.shape
cell = 2 * extent / nx

print(f"\nGrid: {nx}^3, Box: {extent*2} Mpc, Cell: {cell:.2f} Mpc")

# Load galaxy data
from velocity_reconstruction.data_io.catalog_readers import read_cosmicflows4
from velocity_reconstruction.preprocessing.malmquist import apply_malmquist_correction
from velocity_reconstruction.preprocessing.peculiar_velocity import compute_peculiar_velocity
from velocity_reconstruction.preprocessing.quality_flags import validate_catalog

cat = read_cosmicflows4('e:/WALLABY/data/cosmicflows4.csv')

distance_corr = cat.distance.copy()
for method in np.unique(cat.method):
    if method == 'Unknown':
        continue
    mask = cat.method == method
    mag_lim = {"TF": 14.5, "FP": 16.0, "SN Ia": 17.5}.get(method, 15.0)
    distance_corr[mask] = apply_malmquist_correction(
        cat.distance[mask], cat.distance_error[mask], method=method, magnitude_limit=mag_lim
    )

valid_mask = (distance_corr > 1.5) & (distance_corr < 180) & np.isfinite(distance_corr)
v_pec = np.full(len(cat.cz), np.nan, dtype=np.float64)
v_pec[valid_mask] = compute_peculiar_velocity(cat.cz[valid_mask], distance_corr[valid_mask], H0=H0)

v_pec_err = np.full_like(v_pec, 300.0)
masks = validate_catalog(cat.ra, cat.cz, cat.cz, distance_corr, v_pec, v_pec_err,
    config={"max_error_ratio": 2.0, "galactic_plane_margin": 3.0, "v_min": -4000, "v_max": 5000})
good = masks["good"] & np.isfinite(v_pec) & (distance_corr > 1.5) & (distance_corr < 180) & (np.abs(cat.dec) > 3)

# Supergalactic coordinates
SGP_RA, SGP_DEC = 283.8, 15.7

def eq_to_sg(ra_deg, dec_deg):
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    sgp_ra_rad = np.deg2rad(SGP_RA)
    sgp_dec_rad = np.deg2rad(SGP_DEC)
    
    sgb = np.arcsin(np.sin(dec_rad) * np.sin(sgp_dec_rad) + 
                    np.cos(dec_rad) * np.cos(sgp_dec_rad) * np.cos(ra_rad - sgp_ra_rad))
    sgl = np.arctan2(np.cos(dec_rad) * np.sin(ra_rad - sgp_ra_rad),
                     np.cos(dec_rad) * np.cos(ra_rad - sgp_ra_rad) * np.cos(sgp_dec_rad) - 
                     np.sin(dec_rad) * np.sin(sgp_dec_rad))
    return np.rad2deg(sgl), np.rad2deg(sgb)

sgl, sgb = eq_to_sg(cat.ra, cat.dec)
dist = distance_corr

x_gal = dist * np.cos(np.deg2rad(sgb)) * np.cos(np.deg2rad(sgl))
y_gal = dist * np.cos(np.deg2rad(sgb)) * np.sin(np.deg2rad(sgl))
z_gal = dist * np.sin(np.deg2rad(sgb))

x_gal = x_gal[good]
y_gal = y_gal[good]
z_gal = z_gal[good]
v_obs = v_pec[good]
n_gal = len(x_gal)

print(f"Galaxies: {n_gal}")

# Filter to grid bounds
in_bounds = (np.abs(x_gal) < extent) & (np.abs(y_gal) < extent) & (np.abs(z_gal) < extent)
x_gal = x_gal[in_bounds]
y_gal = y_gal[in_bounds]
z_gal = z_gal[in_bounds]
v_obs = v_obs[in_bounds]
n_gal = len(x_gal)

print(f"Galaxies in grid: {n_gal}")

# OPTION C: Direct density from galaxies
print("\n[1] Computing density from galaxy positions...")

density_grid = np.zeros((nx, ny, nz))
ix = ((x_gal + extent) / cell).astype(int)
iy = ((y_gal + extent) / cell).astype(int)
iz = ((z_gal + extent) / cell).astype(int)

for i in range(n_gal):
    if 0 <= ix[i] < nx and 0 <= iy[i] < ny and 0 <= iz[i] < nz:
        density_grid[ix[i], iy[i], iz[i]] += 1

sigma_grid = 3.0
density_smooth = gaussian_filter(density_grid, sigma=sigma_grid)
mean_density = np.mean(density_smooth)
density = (density_smooth - mean_density) / mean_density

print(f"    delta range: {np.nanmin(density):.2f} to {np.nanmax(density):.2f}")

# Poisson solver
print("    Solving Poisson...")

kx = np.fft.fftfreq(nx, d=cell) * 2 * np.pi
ky = np.fft.fftfreq(ny, d=cell) * 2 * np.pi
kz = np.fft.fftfreq(nz, d=cell) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
k_squared = KX**2 + KY**2 + KZ**2
k_squared[0, 0, 0] = 1e-10

G_eff = 1.0
rho_k = fftn(density)
potential_k = -4 * np.pi * G_eff * rho_k / k_squared
potential_k[0, 0, 0] = 0
potential = np.real(ifftn(potential_k))

g_x = np.gradient(potential, cell, edge_order=2)[0]
g_y = np.gradient(potential, cell, edge_order=2)[1]
g_z = np.gradient(potential, cell, edge_order=2)[2]
g_mag = np.sqrt(g_x**2 + g_y**2 + g_z**2)

print(f"    |g| range: {np.nanmin(g_mag):.2f} to {np.nanmax(g_mag):.2f}")

# Structure identification
print("\n[2] Identifying structures...")

div_g = np.gradient(g_x, cell, edge_order=2)[0] + \
        np.gradient(g_y, cell, edge_order=2)[1] + \
        np.gradient(g_z, cell, edge_order=2)[2]

print(f"    div(g) range: {np.nanmin(div_g):.2f} to {np.nanmax(div_g):.2f}")

search_radius = 3
local_min = minimum_filter(div_g, size=2*search_radius+1)
attractor_centers = (div_g == local_min) & (div_g < -0.5)
labeled_attractors, n_attractors = label(attractor_centers)

local_max = maximum_filter(div_g, size=2*search_radius+1)
repeller_centers = (div_g == local_max) & (div_g > 0.5)
labeled_repellers, n_repellers = label(repeller_centers)

print(f"    Attractors: {n_attractors}")
print(f"    Repellers: {n_repellers}")

# Local Group
g_LG = np.array([g_x[nx//2, ny//2, nz//2], g_y[nx//2, ny//2, nz//2], g_z[nx//2, ny//2, nz//2]])
g_LG_mag = np.sqrt(np.sum(g_LG**2))

print(f"\n[3] Local Group |g| = {g_LG_mag:.2f}")

# Closure test
print("\n[4] Closure Test...")

x_grid = np.linspace(-extent, extent, nx)
y_grid = np.linspace(-extent, extent, ny)
z_grid = np.linspace(-extent, extent, nz)

interp_gx = RegularGridInterpolator((x_grid, y_grid, z_grid), g_x, bounds_error=False, fill_value=np.nan)
interp_gy = RegularGridInterpolator((x_grid, y_grid, z_grid), g_y, bounds_error=False, fill_value=np.nan)
interp_gz = RegularGridInterpolator((x_grid, y_grid, z_grid), g_z, bounds_error=False, fill_value=np.nan)

# v = scale * g * f
SCALE = 0.15

v_pred = np.full(n_gal, np.nan)
for i in range(n_gal):
    pos = np.array([x_gal[i], y_gal[i], z_gal[i]])
    if np.all(np.abs(pos) < extent):
        g_vec = np.array([interp_gx(pos)[0], interp_gy(pos)[0], interp_gz(pos)[0]])
        r = np.linalg.norm(pos)
        if r > 0.1:
            los = pos / r
            g_radial = np.dot(g_vec, los)
            v_pred[i] = SCALE * g_radial * f_growth

valid = np.isfinite(v_pred) & np.isfinite(v_obs)
n_valid = valid.sum()
print(f"    Valid: {n_valid}")

if n_valid > 100:
    v_obs_v = v_obs[valid]
    v_pred_v = v_pred[valid]
    
    r = np.corrcoef(v_obs_v, v_pred_v)[0, 1]
    residual = v_obs_v - v_pred_v
    rms = np.sqrt(np.nanmean(residual**2))
    bias = np.nanmean(residual)
    
    print(f"\n    Closure: r={r:.4f}, RMS={rms:.0f} km/s, bias={bias:.0f} km/s")
    print(f"    v_obs mean: {np.mean(v_obs_v):.0f} km/s, v_pred mean: {np.mean(v_pred_v):.0f} km/s")

# Visualization
print("\n[5] Generating visualization...")

fig = plt.figure(figsize=(18, 14))

ax1 = fig.add_subplot(2, 2, 1)
slice_idx = nz // 2
im = ax1.imshow(density[:, :, slice_idx].T, origin='lower', cmap='RdBu_r', 
                extent=[-extent, extent, -extent, extent], vmin=-1, vmax=1)
ax1.set_xlabel('X (Mpc)')
ax1.set_ylabel('Y (Mpc)')
ax1.set_title('Density Contrast (z=0)')
plt.colorbar(im, ax=ax1, label='delta')

ax2 = fig.add_subplot(2, 2, 2)
im2 = ax2.imshow(g_mag[:, :, slice_idx].T, origin='lower', cmap='viridis',
                  extent=[-extent, extent, -extent, extent])
ax2.set_xlabel('X (Mpc)')
ax2.set_ylabel('Y (Mpc)')
ax2.set_title('Gravity Magnitude')
plt.colorbar(im2, ax=ax2, label='|g|')

ax3 = fig.add_subplot(2, 2, 3)
im3 = ax3.imshow(div_g[:, :, slice_idx].T, origin='lower', cmap='PiYG',
                  extent=[-extent, extent, -extent, extent], vmin=-1, vmax=1)
ax3.set_xlabel('X (Mpc)')
ax3.set_ylabel('Y (Mpc)')
ax3.set_title('Div(g)')
plt.colorbar(im3, ax=ax3, label='div(g)')

ax4 = fig.add_subplot(2, 2, 4)
if n_valid > 100:
    ax4.scatter(v_obs_v, v_pred_v, alpha=0.2, s=3)
    ax4.plot([-2000, 2000], [-2000, 2000], 'r--', lw=2, label='1:1')
    ax4.set_xlabel('Observed v (km/s)')
    ax4.set_ylabel('Predicted v (km/s)')
    ax4.set_title(f'Closure: r={r:.4f}, RMS={rms:.0f}')
    ax4.legend()
    ax4.set_xlim(-2000, 2000)
    ax4.set_ylim(-2000, 2000)

plt.tight_layout()
plt.savefig('e:/WALLABY/results/attractor_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("    Saved: results/attractor_analysis.png")

# LaTeX report
print("\n[6] Generating LaTeX report...")

with open('e:/WALLABY/results/attractor_report.tex', 'w') as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage[margin=1in]{geometry}\n")
    f.write("\\usepackage{amsmath,amssymb,graphicx}\n")
    f.write("\\begin{document}\n")
    f.write("\\title{Gravitational Field Analysis - Direct Density}\n")
    f.write("\\author{Velocity Reconstruction Pipeline}\n")
    f.write("\\date{\\today}\n")
    f.write("\\maketitle\n")
    f.write("\\section{Methodology}\n")
    f.write("Direct density reconstruction: galaxies -> density -> potential -> gravity -> velocity.\n")
    f.write("\\section{Results}\n")
    f.write(f"Grid: {nx}$^3$, Box: {extent*2} Mpc. ")
    f.write(f"Attractors: {n_attractors}, Repellers: {n_repellers}. ")
    f.write(f"Local Group |g|: {g_LG_mag:.2f}\n")
    f.write("\\section{Closure Test}\n")
    f.write(f"Correlation: r = {r:.4f}. RMS: {rms:.0f} km/s. Bias: {bias:.0f} km/s.\n")
    f.write("\\section{Visualization}\n")
    f.write("\\begin{figure}[h]\n")
    f.write("\\centering\n")
    f.write("\\includegraphics[width=0.9\\textwidth]{attractor_analysis.png}\n")
    f.write("\\end{figure}\n")
    f.write("\\end{document}\n")

print("    Saved: results/attractor_report.tex")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
