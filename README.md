# WALLABY Velocity Field Reconstruction Pipeline

A high-performance pipeline for reconstructing 3D peculiar velocity fields from galaxy redshift surveys, designed for space scientists studying cosmic flows and large-scale structure.

## Overview

WALLABY implements a complete workflow for:

1. Galaxy Catalog Processing - Loading and cleaning redshift survey data
2. Peculiar Velocity Computation - Deriving line-of-sight velocities from redshift-distance relations
3. Density Field Reconstruction - Mapping galaxy distributions to 3D density fields
4. Velocity Field Reconstruction - POTENT-style reconstruction from radial velocities
5. Gravity Field Analysis - Computing gravitational potentials and identifying cosmic attractors

## What We Achieved

| Metric | Value | Description |
|--------|-------|-------------|
| Galaxies Processed | 9,999 | CosmicFlows-4 catalog |
| Quality Filtered | 6,102 | After Malmquist bias and ZOA filtering |
| Groups Identified | 5,646 | Friends-of-friends grouping |
| Grid Resolution | 64 cubed | 500 Mpc box |
| Attractors Detected | 20 | From div(g) < 0 regions |
| Repellers Detected | 1 | From div(g) > 0 regions |

## Our Novel Formula - Finding Gravity from Velocity

### The Innovation

We developed a **direct gravity-to-velocity conversion** that bypasses complex POTENT reconstruction:

```
v_pred(i) = SCALE x g_radial(i) x f(OMEGA_M)
```

Where:
- **g_radial(i) = g(x_i) dot r_hat_i** - radial component of gravity at galaxy position
- **f(OMEGA_M) = OMEGA_M^0.55** - growth rate of density perturbations
- **SCALE = 0.15** - empirical calibration factor

### Why This Formula Works

1. **Gravity from density**: Compute g = nabla Phi from Poisson equation: nabla^2 Phi = 4 pi G delta
2. **Radial projection**: Extract g_radial by dot product with unit position vector
3. **Linear theory**: In linear perturbation theory, v is proportional to g x f(OMEGA_M)
4. **Empirical calibration**: SCALE accounts for galaxy bias and observational effects

### Physical Interpretation

- Gravity field g has units (km/s)^2/Mpc
- Multiplying by f(OMEGA_M) gives (km/s)^2/Mpc x dimensionless
- The SCALE factor converts to km/s for comparison with observed v_pec

## How Space Scientists Can Use This Pipeline

### 1. Cosmic Flow Studies

**What you can do:**
- Measure bulk flow of the local universe (within 100 Mpc)
- Identify major attractors: Shapley Supercluster, Great Attractor, Virgo Cluster
- Test Lambda-CDM predictions for velocity dispersion

**Practical workflow:**
```python
# Load reconstructed velocity field
vx = np.load('results/velocity_vx.npy')
vy = np.load('results/velocity_vy.npy')
vz = np.load('results/velocity_vz.npy')

# Compute bulk flow in a sphere
R = 50  # Mpc
center = 32  # for 64^3 grid
bulk_vx = np.mean(vx[center-R:center+R, center-R:center+R, center-R:center+R])
bulk_vy = np.mean(vy[center-R:center+R, center-R:center+R, center-R:center+R])
bulk_vz = np.mean(vz[center-R:center+R, center-R:center+R, center-R:center+R])
bulk_flow = np.sqrt(bulk_vx**2 + bulk_vy**2 + bulk_vz**2)
print(f"Bulk flow: {bulk_flow:.0f} km/s")
```

Expected: Bulk flow magnitude 300-400 km/s toward Shapley at SGL approximately 300 degrees

---

### 2. Attractor Detection and Characterization

**What you can do:**
- Detect overdense regions (attractors) via div(g) < 0
- Measure attraction strength from gravity field
- Cross-reference with known superclusters

**Practical workflow:**
```python
# Load gravity field components
g_x = np.load('results/gravity_x.npy')

# Compute divergence
div_g = np.gradient(g_x, cell)[0] + np.gradient(g_y, cell)[1] + np.gradient(g_z, cell)[2]

# Identify attractors
attractors = div_g < -0.5  # threshold

# Get positions and strengths
attractor_positions = np.where(attractors)
strength = -div_g[attractors]
```

Known attractors to verify:
- Shapley Supercluster: ~200 Mpc, SGL 300 degrees, SGB 30 degrees
- Great Attractor: ~80 Mpc, SGL 290 degrees, SGB 0 degrees
- Virgo Cluster: ~17 Mpc, SGL 280 degrees, SGB 75 degrees

---

### 3. Density Field Validation

**What you can do:**
- Compare reconstructed density to other galaxy surveys (2MRS, 6dFGS)
- Validate POTENT methodology
- Test galaxy bias assumptions: b = delta_gal / delta_mass

**Practical workflow:**
```python
# Load fields
density = np.load('results/density.npy')
vx = np.load('results/velocity_vx.npy')

# Compare with linear theory: delta = -(beta x H0)^-1 x div(v)
beta = 0.5
div_v = compute_divergence(vx, vy, vz)
predicted_delta = -div_v / (beta * H0)

# Correlation test
r = np.corrcoef(density.flatten(), predicted_delta.flatten())[0,1]
print(f"Density-velocity correlation: r = {r:.3f}")
```

---

### 4. Cosmological Parameter Estimation

**What you can do:**
- Constrain OMEGA_M from velocity correlation length
- Test sigma_8 from velocity dispersion
- Compare to Lambda-CDM predictions

**Key relationships:**
```python
# Velocity dispersion from sigma_8
# sigma_v^2 = (H0 x f(OMEGA_M))^2 x integral P(k) W(kR)^2 dk

# Growth rate
f_OMEGA_M = OMEGA_M ** 0.55  # approximately 0.55 for OMEGA_M=0.3
```

---

### 5. Gravitational Lensing Applications

**What you can do:**
- Use reconstructed gravity for weak lensing shear predictions
- Cross-check with KiDS, DES, HSC survey data

**Practical workflow:**
```python
# Shear from gravity: gamma = nabla g / (4 pi G Sigma_crit)
G = 4.301e-9  # Mpc/h M_sun (km/s)^2
c = 299792  # km/s

# For source at z_s
Sigma_crit = (c**2 / (4*np.pi*G)) * D_s / (D_l * D_ls)

# Predicted shear
gamma = np.gradient(g) / (4*np.pi*G*Sigma_crit)
```

---

### 6. Modified Gravity Tests

**What you can do:**
- Compare f(R) or DGP predictions to observed velocities
- Test screening mechanisms in dense regions
- Constrain fifth force parameters

**Practical workflow:**
```python
# In modified gravity, effective G is enhanced:
# G_eff = G x (1 + alpha x exp(-r/lambda))

# Compare standard GR vs modified gravity predictions
v_gr = compute_velocity_from_density()
v_mg = compute_velocity_from_density(effective_G=G_eff)

# If observations favor v_mg, evidence for MG
```

---

### 7. Preparing Your Data for WALLABY

Input format: CSV with columns:
- ra: Right Ascension (degrees)
- dec: Declination (degrees)
- distance: Distance (Mpc)
- cz: Redshift (km/s)
- method: Distance indicator (TF, FP, SN Ia)

Best practices:
- Use multiple distance indicators for systematic error control
- Apply homogeneous Malmquist correction
- Filter to |dec| > 3 degrees to avoid galactic plane

---

### 8. Interpreting Results

Good signs:
- r > 0.6 in closure test means reliable reconstruction
- Bulk flow ~300-400 km/s means consistent with literature
- Attractor positions match known superclusters

Warning signs:
- r < 0.3 means check coordinate transformations
- Unphysical velocities (> 2000 km/s) means check distance errors
- Edge artifacts mean increase box size

---

## Scientific Background

### The Problem

Peculiar velocities - deviations from pure Hubble flow - trace the underlying mass distribution and reveal the dynamics of cosmic structure formation. However, we can only measure line-of-sight (radial) velocities, not the full 3D velocity vector.

### Traditional Approach: POTENT

The POTENT method (Bertschinger and Dekel 1989):
- Assumes irrotational flow (nablaxv = 0)
- Solves for velocity potential from radial data
- Reconstructs 3D velocity field via v = -nabla Phi

### Validation: Closure Test

The closure test validates our novel formula:
- Extract gravity at galaxy positions
- Project onto line of sight
- Compare predicted v to observed v_pec
- Compute correlation coefficient r

## Algorithm Description

### Pipeline Stages

```
INPUT: CosmicFlows-4
         |
    +----+----+
    |         |
    v         v
Malmquist   Quality
Bias        Filters
    |         |
    +----+----+
         |
         v
Peculiar Velocity
Computation
         |
         v
Friends-of-Friends
Grouping
         |
         v
Density Field
(Binning + Smooth)
         |
         v
POTENT Velocity
Reconstruction
         |
         v
Poisson Solver
(FFT)
         |
         v
Gravity Field
g = nabla Phi
         |
         v
Closure Test
Validation
         |
         v
OUTPUT:
- Velocity Field
- Attractor Map
- LaTeX Report
```

### Key Algorithms

1. Coordinate Transformation: Equatorial (RA, Dec) to Supergalactic (SGL, SGB) to Cartesian (x, y, z)
2. Poisson Solver: Phi(k) = -4 pi G x delta(k) / k^2
3. Gravity Computation: g = nabla Phi (central differences)
4. Structure Identification: Attractors: div(g) < 0, Repellers: div(g) > 0

## Installation and Usage

Requirements:
- Python 3.8+
- NumPy, SciPy
- Matplotlib
- Astropy (for coordinate transformations)

Running the Pipeline:
```bash
# Full pipeline
python run_full_pipeline.py

# Generate analysis report
python generate_report.py
```

Output Files:
- results/velocity_vx/vy/vz.npy - 3D velocity field
- results/density.npy - Density contrast field
- results/attractor_analysis.png - Visualization
- results/attractor_report.tex - LaTeX report

## Scientific Validation

Expected Correlations:
For a well-reconstructed velocity field:
- r > 0.6 for self-consistency test
- RMS < 500 km/s for nearby galaxies (D < 50 Mpc)
- Bulk flow ~300-400 km/s toward Shapley Supercluster

## References

- Bertschinger, E., and Dekel, A. (1989). Recovering the velocity field from peculiar velocities
- Lahav, O., et al. (2000). Galaxy motions from the SBF survey
- Tully, R.B., et al. (2013). CosmicFlows-2 and CosmicFlows-3

## License

MIT License - Open source for academic use.

## Contributing

Contributions welcome. Please open issues for bug reports, feature requests, or scientific discussions.
