"""Fix: Move density computation after x_gal is defined"""

with open('e:/WALLABY/generate_report.py', 'r', encoding='utf-8') as f:
    content = f.read()

# First, remove the current density computation from line ~65-90
old_density_section = """# Direct density from galaxies with Gaussian smoothing
from scipy.ndimage import gaussian_filter

print("    Computing density directly from galaxy positions...")

# Create density field from galaxy positions
density_grid = np.zeros((nx, ny, nz))
ix = ((x_gal + extent) / cell).astype(int)
iy = ((y_gal + extent) / cell).astype(int)
iz = ((z_gal + extent) / cell).astype(int)

for i in range(len(x_gal)):
    if 0 <= ix[i] < nx and 0 <= iy[i] < ny and 0 <= iz[i] < nz:
        density_grid[ix[i], iy[i], iz[i]] += 1

# Gaussian smoothing
sigma_grid = 3.0
density_smooth = gaussian_filter(density_grid, sigma=sigma_grid)
mean_density = np.mean(density_smooth)
density = (density_smooth - mean_density) / mean_density

rho = density

# FFT-based Poisson solver"""

# Replace with placeholder
content = content.replace(old_density_section, """# Density will be computed after galaxy positions are loaded
# (moved to closure test section)
rho_placeholder = None

# FFT-based Poisson solver""")

# Now add density computation after in_bounds filtering
old_inbounds = "x_gal = x_gal[in_bounds]"
new_inbounds = """x_gal = x_gal[in_bounds]

# ============================================================
# COMPUTE DENSITY FROM GALAXY POSITIONS (Option C)
# ============================================================
print("    Computing density directly from galaxy positions...")

from scipy.ndimage import gaussian_filter

density_grid = np.zeros((nx, ny, nz))
ix = ((x_gal + extent) / cell).astype(int)
iy = ((y_gal + extent) / cell).astype(int)
iz = ((z_gal + extent) / cell).astype(int)

for i in range(len(x_gal)):
    if 0 <= ix[i] < nx and 0 <= iy[i] < ny and 0 <= iz[i] < nz:
        density_grid[ix[i], iy[i], iz[i]] += 1

sigma_grid = 3.0
density_smooth = gaussian_filter(density_grid, sigma=sigma_grid)
mean_density = np.mean(density_smooth)
density = (density_smooth - mean_density) / mean_density

rho = density"""

content = content.replace(old_inbounds, new_inbounds)

# Also fix rho_placeholder reference
content = content.replace("rho = density", "rho = density  # Now properly set")

with open('e:/WALLABY/generate_report.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed density computation location!")
