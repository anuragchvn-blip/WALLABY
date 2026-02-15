"""Quick fix script to update generate_report.py"""

# Read the file
with open('e:/WALLABY/generate_report.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic section
old_text = """# Using dimensionless density contrast δ
# The Poisson equation gives: ∇²Φ = 4πG ρ̄ δ
# For g = -∇Φ, we use linear theory relation: g ≈ c² × δ / (H0⁻¹)
# Simplified: scale to get reasonable g values (~100 km/s/Mpc for δ~1)
# g = A * gradient of smoothed density, where A is a calibration factor

# Scale factor - calibrate to get |g| ~ 100 km/s/Mpc for δ ~ 1
# The density contrast δ ranges from -0.75 to 0.3, so we need larger scaling
SCALE_FACTOR = 50000.0  # Increased to get realistic gravity values

# Proper gravitational field from Poisson equation
# ∇²Φ = 4πGρ̄δ  ->  Φ ∝ δ/k² in Fourier space
# Then g = ∇Φ (acceleration)
# Then v = (f/H0) * g (linear theory)

# Use FFT to solve Poisson with proper scaling
# This gives potential in (km/s)² units
rho = density  # density contrast"""

new_text = """# Direct density from galaxies with Gaussian smoothing
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

rho = density"""

content = content.replace(old_text, new_text)

# Write back
with open('e:/WALLABY/generate_report.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed!")
