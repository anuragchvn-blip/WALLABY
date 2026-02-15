"""Completely restructure: move all gravity computation after data load"""

with open('e:/WALLABY/generate_report.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove early gravity computation section (lines 33-115 roughly)
old_grav_section = """# ============================================================
# METHOD 1: GRAVITATIONAL FIELD FROM DENSITY (CORRECT)
# Using Poisson equation: ∇²Φ = 4πGρ̄δ
# Then: g = -∇Φ
# ============================================================

print("\n[1] Computing gravitational potential from density...")

def solve_poisson_fft(density, cell_size):
    Solve Poisson equation using FFT (periodic BC).
    nx, ny, nz = density.shape
    kx = np.fft.fftfreq(nx, d=cell_size) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=cell_size) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=cell_size) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = KX**2 + KY**2 + KZ**2
    k_squared[0, 0, 0] = 1e-10  # Avoid division by zero
    
    density_fft = fftn(density)
    potential_fft = -4 * np.pi * G * density_fft / k_squared
    potential_fft[0, 0, 0] = 0  # Zero mean
    
    return np.real(ifftn(potential_fft))

def compute_gradient(field, cell_size):
    Compute 3D gradient of a scalar field.
    grad_x = np.gradient(field, cell_size, edge_order=2)[0]
    grad_y = np.gradient(field, cell_size, edge_order=2)[1]
    grad_z = np.gradient(field, cell_size, edge_order=2)[2]
    return grad_x, grad_y, grad_z

# Density will be computed after galaxy positions are loaded
# (moved to closure test section)
rho_placeholder = None

# FFT-based Poisson solver
kx = np.fft.fftfreq(nx, d=cell) * 2 * np.pi
ky = np.fft.fftfreq(ny, d=cell) * 2 * np.pi
kz = np.fft.fftfreq(nz, d=cell) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
k_squared = KX**2 + KY**2 + KZ**2
k_squared[0, 0, 0] = 1e-10  # Avoid division by zero

# Poisson: Φ_k = -4πG * ρ_k / k²
# Use calibrated G to get realistic g values (~100 km/s/Mpc for δ~1)
G_eff = 10.0  # Calibrated effective gravitational constant
rho_k = fftn(rho)
potential_k = -4 * np.pi * G_eff * rho_k / k_squared
potential_k[0, 0, 0] = 0
potential = np.real(ifftn(potential_k))

# Gravity: g = ∇Φ
g_x = np.gradient(potential, cell, edge_order=2)[0]
g_y = np.gradient(potential, cell, edge_order=2)[1]
g_z = np.gradient(potential, cell, edge_order=2)[2]
g_mag = np.sqrt(g_x**2 + g_y**2 + g_z**2)

print(f"    |g| range: {np.nanmin(g_mag):.0f} to {np.nanmax(g_mag):.0f} km/s/Mpc")
print(f"    |g| mean: {np.nanmean(g_mag):.0f} km/s/Mpc")

# Divergence of g (should match -density via Poisson)
div_g = np.gradient(g_x, cell, edge_order=2)[0] + \\
        np.gradient(g_y, cell, edge_order=2)[1] + \\
        np.gradient(g_z, cell, edge_order=2)[2]

print(f"\n[2] Divergence of g")
print(f"    div(g) range: {np.nanmin(div_g):.2f} to {np.nanmax(div_g):.2f}")

# Identify attractors and repellers
# div(g) < 0 = attractor (convergence, overdense)
# div(g) > 0 = repeller (divergence, underdense)
search_radius = 3
local_min = minimum_filter(div_g, size=2*search_radius+1)
attractor_centers = (div_g == local_min) & (div_g < -0.5)
labeled_attractors, n_attractors = label(attractor_centers)

local_max = maximum_filter(div_g, size=2*search_radius+1)
repeller_centers = (div_g == local_max) & (div_g > 0.5)
labeled_repellers, n_repellers = label(repeller_centers)

print(f"\n[3] Structures")
print(f"    Attractors (div<0): {n_attractors}")
print(f"    Repellers (div>0): {n_repellers}")

# Extract attractor positions
attractors = []
for i in range(1, n_attractors + 1):
    mask = labeled_attractors == i
    if mask.sum() < 10:
        continue
    indices = np.where(mask)
    weights = -div_g[mask]
    weights = np.maximum(weights, 0.001)
    
    cx = np.sum(indices[0] * weights) / np.sum(weights)
    cy = np.sum(indices[1] * weights) / np.sum(weights)
    cz = np.sum(indices[2] * weights) / np.sum(weights)
    
    sx = (cx - nx/2) * cell
    sy = (cy - ny/2) * cell
    sz = (cz - nz/2) * cell
    
    attractors.append({
        'x': sx, 'y': sy, 'z': sz,
        'distance': np.sqrt(sx**2 + sy**2 + sz**2),
        'div': div_g[mask].min(),
        'strength': -div_g[mask].sum() * cell**3
    })

attractors = sorted(attractors, key=lambda a: -a['strength'])

# Extract repellers
repellers = []
for i in range(1, n_repellers + 1):
    mask = labeled_repellers == i
    if mask.sum() < 10:
        continue
    indices = np.where(mask)
    cx = np.mean(indices[0])
    cy = np.mean(indices[1])
    cz = np.mean(indices[2])
    sx = (cx - nx/2) * cell
    sy = (cy - ny/2) * cell
    sz = (cz - nz/2) * cell
    repellers.append({
        'x': sx, 'y': sy, 'z': sz,
        'distance': np.sqrt(sx**2 + sy**2 + sz**2),
        'div': div_g[mask].max()
    })

repellers = sorted(repellers, key=lambda r: -r['div'])

print(f"\n    Top 5 Attractors:")
for i, att in enumerate(attractors[:5]):
    print(f"      {i+1}. ({att['x']:.0f}, {att['y']:.0f}, {att['z']:.0f}) Mpc, dist={att['distance']:.0f} Mpc")

# Local Group
g_LG = np.array([g_x[nx//2, ny//2, nz//2], 
                 g_y[nx//2, ny//2, nz//2], 
                 g_z[nx//2, ny//2, nz//2]])
g_LG_mag = np.sqrt(np.sum(g_LG**2))
theta = np.arctan2(np.sqrt(g_LG[0]**2 + g_LG[1]**2), g_LG[2])
phi = np.arctan2(g_LG[1], g_LG[0])
sgl = np.rad2deg(phi)
sgb = np.rad2deg(np.pi/2 - theta)
sgl_val = float(sgl)
sgb_val = float(sgb)

print(f"\n[4] Local Group")
print(f"    |g| = {g_LG_mag:.0f} km/s/Mpc")
print(f"    Direction: SGL={sgl_val:.0f}\u00b0, SGB={sgb_val:.0f}\u00b0")"""

# Replace with simpler placeholder - we'll compute gravity later
new_grav_section = """# ============================================================
# METHOD 1: GRAVITATIONAL FIELD FROM DENSITY (CORRECT)
# Using Poisson equation: ∇²Φ = 4πGρ̄δ
# Then: g = -∇Φ
# Will be computed after galaxy data is loaded
# ============================================================

print("\n[1] Computing gravitational potential from density...")

# Placeholder - will be computed later
g_x = g_y = g_z = g_mag = None
div_g = None
attractors = []
repellers = []
n_attractors = n_repellers = 0
g_LG_mag = 0.0
sgl_val = sgb_val = 0.0"""

content = content.replace(old_grav_section, new_grav_section)

# Now add gravity computation after density is computed
old_density_done = """rho = density"""

new_density_done = """rho = density

# ============================================================
# COMPUTE GRAVITY FROM DENSITY (Poisson + Gradient)
# ============================================================
print("    Solving Poisson equation...")

# FFT-based Poisson solver
kx = np.fft.fftfreq(nx, d=cell) * 2 * np.pi
ky = np.fft.fftfreq(ny, d=cell) * 2 * np.pi
kz = np.fft.fftfreq(nz, d=cell) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
k_squared = KX**2 + KY**2 + KZ**2
k_squared[0, 0, 0] = 1e-10

# Poisson: Φ_k = -4πG * ρ_k / k²
G_eff = 10.0
rho_k = fftn(rho)
potential_k = -4 * np.pi * G_eff * rho_k / k_squared
potential_k[0, 0, 0] = 0
potential = np.real(ifftn(potential_k))

# Gravity: g = ∇Φ
g_x = np.gradient(potential, cell, edge_order=2)[0]
g_y = np.gradient(potential, cell, edge_order=2)[1]
g_z = np.gradient(potential, cell, edge_order=2)[2]
g_mag = np.sqrt(g_x**2 + g_y**2 + g_z**2)

print(f"    |g| range: {np.nanmin(g_mag):.0f} to {np.nanmax(g_mag):.0f}")
print(f"    |g| mean: {np.nanmean(g_mag):.0f}")

# Divergence of g
div_g = np.gradient(g_x, cell, edge_order=2)[0] + \\
        np.gradient(g_y, cell, edge_order=2)[1] + \\
        np.gradient(g_z, cell, edge_order=2)[2]

print(f"\n[2] Divergence of g")
print(f"    div(g) range: {np.nanmin(div_g):.2f} to {np.nanmax(div_g):.2f}")

# Identify attractors and repellers
search_radius = 3
local_min = minimum_filter(div_g, size=2*search_radius+1)
attractor_centers = (div_g == local_min) & (div_g < -0.5)
labeled_attractors, n_attractors = label(attractor_centers)

local_max = maximum_filter(div_g, size=2*search_radius+1)
repeller_centers = (div_g == local_max) & (div_g > 0.5)
labeled_repellers, n_repellers = label(repeller_centers)

print(f"\n[3] Structures")
print(f"    Attractors (div<0): {n_attractors}")
print(f"    Repellers (div>0): {n_repellers}")

# Extract attractor positions
attractors = []
for i in range(1, n_attractors + 1):
    mask = labeled_attractors == i
    if mask.sum() < 10:
        continue
    indices = np.where(mask)
    weights = -div_g[mask]
    weights = np.maximum(weights, 0.001)
    
    cx = np.sum(indices[0] * weights) / np.sum(weights)
    cy = np.sum(indices[1] * weights) / np.sum(weights)
    cz = np.sum(indices[2] * weights) / np.sum(weights)
    
    sx = (cx - nx/2) * cell
    sy = (cy - ny/2) * cell
    sz = (cz - nz/2) * cell
    
    attractors.append({
        'x': sx, 'y': sy, 'z': sz,
        'distance': np.sqrt(sx**2 + sy**2 + sz**2),
        'div': div_g[mask].min(),
        'strength': -div_g[mask].sum() * cell**3
    })

attractors = sorted(attractors, key=lambda a: -a['strength'])

# Extract repellers
repellers = []
for i in range(1, n_repellers + 1):
    mask = labeled_repellers == i
    if mask.sum() < 10:
        continue
    indices = np.where(mask)
    cx = np.mean(indices[0])
    cy = np.mean(indices[1])
    cz = np.mean(indices[2])
    sx = (cx - nx/2) * cell
    sy = (cy - ny/2) * cell
    sz = (cz - nz/2) * cell
    repellers.append({
        'x': sx, 'y': sy, 'z': sz,
        'distance': np.sqrt(sx**2 + sy**2 + sz**2),
        'div': div_g[mask].max()
    })

repellers = sorted(repellers, key=lambda r: -r['div'])

print(f"\n    Top 5 Attractors:")
for i, att in enumerate(attractors[:5]):
    print(f"      {i+1}. ({att['x']:.0f}, {att['y']:.0f}, {att['z']:.0f}) Mpc, dist={att['distance']:.0f} Mpc")

# Local Group
g_LG = np.array([g_x[nx//2, ny//2, nz//2], 
                 g_y[nx//2, ny//2, nz//2], 
                 g_z[nx//2, ny//2, nz//2]])
g_LG_mag = np.sqrt(np.sum(g_LG**2))
theta = np.arctan2(np.sqrt(g_LG[0]**2 + g_LG[1]**2), g_LG[2])
phi = np.arctan2(g_LG[1], g_LG[0])
sgl = np.rad2deg(phi)
sgb = np.rad2deg(np.pi/2 - theta)
sgl_val = float(sgl)
sgb_val = float(sgb)

print(f"\n[4] Local Group")
print(f"    |g| = {g_LG_mag:.0f}")
print(f"    Direction: SGL={sgl_val:.0f}\u00b0, SGB={sgb_val:.0f}\u00b0")"""

content = content.replace(old_density_done, new_density_done)

with open('e:/WALLABY/generate_report.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Restructured!")
