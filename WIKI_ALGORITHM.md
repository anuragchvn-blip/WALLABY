# Algorithm Documentation

## Core Algorithms

### 1. Peculiar Velocity Computation

**Input:** Redshift (cz), Distance (D), Hubble constant (H0)

**Formula:**
```
v_pec = cz - H0 × D
```

**Implementation:**
```python
def compute_peculiar_velocity(cz, distance, H0=70.0):
    """Compute peculiar velocity from redshift and distance."""
    return cz - H0 * distance
```

**Outlier Removal:**
- 3σ clipping to remove unphysical values
- Range filtering: -4000 < v < 5000 km/s

---

### 2. Coordinate Transformation

**From Equatorial to Supergalactic:**

```python
def eq_to_sg(ra_deg, dec_deg):
    """Transform equatorial to supergalactic coordinates."""
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    
    sgp_ra = 283.8  # Super Galactic Pole RA
    sgp_dec = 15.7  # Super Galactic Pole Dec
    
    sgb = np.arcsin(np.sin(dec_rad) * np.sin(np.deg2rad(sgp_dec)) + 
                    np.cos(dec_rad) * np.cos(np.deg2rad(sgp_dec)) * 
                    np.cos(ra_rad - np.deg2rad(sgp_ra)))
    
    sgl = np.arctan2(np.cos(dec_rad) * np.sin(ra_rad - np.deg2rad(sgp_ra)),
                     np.cos(dec_rad) * np.cos(ra_rad - np.deg2rad(sgp_dec)) * 
                     np.sin(np.deg2rad(sgp_dec)) - np.sin(dec_rad) * np.cos(np.deg2rad(sgp_dec)))
    
    return np.rad2deg(sgl), np.rad2deg(sgb)
```

**From Spherical to Cartesian:**
```python
x = D × cos(SGB) × cos(SGL)
y = D × cos(SGB) × sin(SGL)
z = D × sin(SGB)
```

---

### 3. Density Field Construction

**Grid Assignment:**
```python
# Convert galaxy positions to grid indices
ix = ((x + extent) / cell_size).astype(int)
iy = ((y + extent) / cell_size).astype(int)
iz = ((z + extent) / cell_size).astype(int)

# Assign to 3D histogram
for i in range(n_galaxies):
    if in_bounds(ix[i], iy[i], iz[i]):
        density_grid[ix[i], iy[i], iz[i]] += 1
```

**Gaussian Smoothing:**
```python
from scipy.ndimage import gaussian_filter
density_smooth = gaussian_filter(density_grid, sigma=sigma_cells)

# Convert to density contrast
delta = (density_smooth - mean_density) / mean_density
```

---

### 4. Poisson Solver (FFT)

**Theoretical Basis:**
The Poisson equation relates gravitational potential to density:
```
∇²Φ = 4πGρ̄δ
```

**Fourier Space Solution:**
```
Φ(k) = -4πG × δ(k) / k²
```

**Implementation:**
```python
def solve_poisson_fft(density, cell_size, G_eff=1.0):
    """Solve Poisson equation using FFT."""
    nx, ny, nz = density.shape
    
    # Wave numbers
    kx = np.fft.fftfreq(nx, d=cell_size) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=cell_size) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=cell_size) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = KX**2 + KY**2 + KZ**2
    k_squared[0, 0, 0] = 1e-10  # Avoid division by zero
    
    # Solve in Fourier space
    rho_k = np.fft.fftn(density)
    potential_k = -4 * np.pi * G_eff * rho_k / k_squared
    potential_k[0, 0, 0] = 0  # Zero mean
    
    return np.real(np.fft.ifftn(potential_k))
```

---

### 5. Gravity Field Computation

**From Potential to Acceleration:**
```python
def compute_gravity(potential, cell_size):
    """Compute gravity field from potential."""
    g_x = np.gradient(potential, cell_size, edge_order=2)[0]
    g_y = np.gradient(potential, cell_size, edge_order=2)[1]
    g_z = np.gradient(potential, cell_size, edge_order=2)[2]
    
    g_mag = np.sqrt(g_x**2 + g_y**2 + g_z**2)
    return g_x, g_y, g_z, g_mag
```

**Physical Units:**
- Potential: (km/s)²
- Gravity: (km/s)²/Mpc

---

### 6. Structure Identification

**Divergence Calculation:**
```python
div_g = (np.gradient(g_x, cell)[0] + 
         np.gradient(g_y, cell)[1] + 
         np.gradient(g_z, cell)[2])
```

**Attractor/Repeller Detection:**
```python
from scipy.ndimage import minimum_filter, maximum_filter, label

# Attractors: local minima of div(g) where div(g) < 0
local_min = minimum_filter(div_g, size=search_radius)
attractor_mask = (div_g == local_min) & (div_g < threshold)
labeled_attractors, n_attractors = label(attractor_mask)

# Repellers: local maxima of div(g) where div(g) > 0
local_max = maximum_filter(div_g, size=search_radius)
repeller_mask = (div_g == local_max) & (div_g > threshold)
labeled_repellers, n_repellers = label(repeller_mask)
```

---

### 7. Velocity Prediction from Gravity

**Novel Formula:**
```python
def predict_velocity_from_gravity(galaxy_positions, g_field, scale=0.15):
    """
    Predict peculiar velocity from gravity field.
    
    v_pred = SCALE × g_radial × f(ΩM)
    
    Where:
    - g_radial = g · r̂ (dot product with unit position vector)
    - f(ΩM) = ΩM^0.55 (growth rate)
    - SCALE = empirical calibration factor
    """
    f_growth = OMEGA_M ** 0.55
    
    for i in range(n_galaxies):
        pos = galaxy_positions[i]
        r = np.linalg.norm(pos)
        
        if r > 0.1:  # Avoid origin
            los = pos / r  # Line of sight unit vector
            g_vec = interpolate_g_at_position(pos, g_field)
            g_radial = np.dot(g_vec, los)
            
            v_pred[i] = scale * g_radial * f_growth
    
    return v_pred
```

---

### 8. Closure Test / Validation

**Correlation Coefficient:**
```python
def compute_closure_metrics(v_observed, v_predicted):
    """Compute validation metrics."""
    valid = np.isfinite(v_observed) & np.isfinite(v_predicted)
    
    r = np.corrcoef(v_observed[valid], v_predicted[valid])[0, 1]
    residual = v_observed[valid] - v_predicted[valid]
    rms = np.sqrt(np.nanmean(residual**2))
    bias = np.nanmean(residual)
    
    return r, rms, bias
```

**Expected Values:**
- |r| > 0.6: Good reconstruction
- RMS < 500 km/s: Acceptable for nearby galaxies
- Bias ≈ 0: No systematic error

---

### 9. Friends-of-Friends Grouping

```python
def friends_of_friends(positions, linking_length):
    """Identify galaxy groups using FoF algorithm."""
    from scipy.spatial import cKDTree
    
    tree = cKDTree(positions)
    groups = tree.query_ball_tree(tree, r=linking_length)
    
    # Union-find to get unique groups
    parent = list(range(len(positions)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i, group in enumerate(groups):
        for j in group:
            if i != j:
                union(i, j)
    
    return parent
```

---

### 10. Malmquist Bias Correction

```python
def apply_malmquist_correction(distance, distance_error, method, magnitude_limit):
    """
    Apply Malmquist bias correction.
    
    The bias depends on:
    - Distance indicator quality (TF, FP, SN Ia)
    - True distance
    - Local density
    """
    # Simplified correction
    fractional_error = distance_error / distance
    
    # Bias increases with distance and error
    bias_factor = 1 + 0.5 * fractional_error**2
    
    return distance * bias_factor
```

---

## Parameter Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| H0 | 70 km/s/Mpc | Hubble constant |
| ΩM | 0.3 | Matter density |
| f(ΩM) | 0.55 | Growth rate |
| Box Size | 500 Mpc | Simulation box |
| Grid | 64³ | Resolution |
| Cell Size | 7.81 Mpc | Grid spacing |
| Smoothing | 3 cells | Gaussian σ |
| SCALE | 0.15 | g→v calibration |
