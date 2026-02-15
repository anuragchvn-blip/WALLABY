# SCIENTIFIC SOFTWARE SPECIFICATION: VELOCITY FIELD TO MASS DISTRIBUTION RECONSTRUCTION SYSTEM

## PROJECT OBJECTIVE

Build a complete, production-grade pipeline that ingests galaxy peculiar velocity measurements from multiple surveys (CosmicFlows-4, WALLABY, FAST, DESI) and reconstructs the three-dimensional gravitational potential, velocity field, and mass density distribution of the local universe (z < 0.1, within ~300 Mpc), implementing peer-reviewed algorithms validated against N-body simulations.

---

## CORE REQUIREMENTS

### 1. DATA INGESTION AND PREPROCESSING

**Input Catalogs:**
- CosmicFlows-4: 38,065 grouped galaxies with distances from 8 methods (TF, FP, SNe Ia, TRGB, Cepheids, SBF, EPM, GCLF)
- WALLABY PDR2: HI 21cm measurements with baryonic Tully-Fisher distances
- FAST extragalactic HI survey: redshift-independent line widths for TF
- DESI peculiar velocity survey: Fundamental Plane distances for early-type galaxies
- 2MASS ZoA bright galaxy catalog: near-IR photometry for extinction correction

**Required Preprocessing Steps:**
1. **Peculiar velocity calculation** using v_pec = cz_obs - H₀D, where D is distance from indicator
2. **Malmquist bias assessment** for each distance method: quantify selection function as f(m_lim, z, δ) where m_lim is magnitude limit, incorporating both homogeneous (distance-dependent) and inhomogeneous (density-dependent) components
3. **Coordinate transformation** to supergalactic coordinates (SGL, SGB) following Lahav et al. (2000) convention for local structure analysis
4. **Galaxy grouping** using friends-of-friends algorithm with linking length b = 0.25 (Ω_m/0.3)^(-0.6) Mpc as per Cosmicflows methodology, then compute group-averaged velocities weighted by 1/σ²
5. **Extinction correction** for ZoA sources: apply Schlegel, Finkbeiner & Davis (1998) reddening maps, convert E(B-V) to A_K using Cardelli, Clayton & Mathis (1989) extinction law
6. **Error propagation**: preserve individual measurement uncertainties σ_v for downstream Bayesian analysis, including distance indicator intrinsic scatter (0.35 mag TF, 0.1 dex FP, 0.15 mag SNe Ia)

**Data Quality Flags:**
- Reject measurements with σ_v/v_pec > 0.5 (low signal-to-noise)
- Flag sources within 5° of Galactic plane where confusion dominates
- Identify and handle duplicate observations from multiple surveys using weighted averaging
- Mark incomplete redshift coverage regions for boundary condition treatment

---

### 2. VELOCITY FIELD RECONSTRUCTION ALGORITHMS

Implement three independent methods for cross-validation:

#### **METHOD A: POTENT (Bertschinger & Dekel 1989)**

**Algorithm Steps:**
1. **Grid setup**: Establish uniform Cartesian grid in supergalactic coordinates, 200³ cells covering ±100 Mpc, cell size 1 Mpc
2. **Radial velocity assignment**: For each galaxy i at position r_i with line-of-sight unit vector n_i and measured v_r,i, assign to nearest grid point
3. **Gaussian smoothing**: Apply 3D Gaussian kernel with σ = 10-12 Mpc (matching published POTENT analyses), handling edge effects with zero-padding
4. **Curl-free assumption**: Assume v = -∇Φ_v (potential flow), valid in linear regime where vorticity generation is negligible
5. **Radial to 3D reconstruction**: 
   - For each gridpoint r, collect radial velocities from all sightlines passing within smoothing radius
   - Solve for velocity potential Φ_v(r) using least-squares fit to ∂Φ_v/∂r = -v_r along multiple sightlines
   - Compute full velocity vector v = -∇Φ_v via finite differences
6. **Density from velocity**: Apply continuity equation in linear theory: δ(r) = -(1/H₀f) ∇·v, where growth rate f = Ω_m^0.55 for ΛCDM
7. **Gravitational potential**: Solve Poisson equation ∇²Φ = 4πG ρ̄ δ using FFT-based solver or multigrid relaxation

**Validation Criteria:**
- Reproduce published POTENT density maps from MARK III catalog (Dekel et al. 1999) to within systematic error ±0.13 and random error ±0.18 in δ
- Verify curl of reconstructed velocity field |∇×v|/|v| < 0.1 throughout volume
- Confirm convergence: changing smoothing scale from 10 to 15 Mpc should alter large-scale (>30 Mpc) structure by <20%

#### **METHOD B: BAYESIAN WIENER FILTER WITH BIAS GAUSSIANIZATION (BGc+WF, Lavaux 2016)**

**Prior Specification:**
- Power spectrum P(k) from Planck 2018 cosmology: Ω_m = 0.315, σ₈ = 0.811, n_s = 0.965
- Assume Gaussian prior on Fourier modes δ_k with variance P(k)
- Model distance modulus errors as lognormal to capture asymmetric scatter: ln(D_true/D_obs) ~ N(μ_bias, σ²_log)

**Likelihood Construction:**
- For each galaxy i with observed distance D_obs,i and redshift z_i:
  - True distance D_true,i relates to density field via ∫ δ(r) dr along sightline (integrated Sachs-Wolfe effect negligible at z<0.1)
  - Malmquist bias μ_bias,i = -σ²_log/2 + integral over selection function P(detect|D,δ)
  - Likelihood L_i = ∫ P(D_obs,i | D_true, σ_dist) × P(D_true | δ, z_i) dD_true analytically integrated per Boruah et al. (2022)

**Wiener Filter Reconstruction:**
1. Compute mean density field: δ̂_k = [P(k)⁻¹ + N⁻¹]⁻¹ N⁻¹ D_k, where D_k is data vector in Fourier space, N is noise covariance
2. Generate constrained realizations by adding random Gaussian field with covariance [P(k)⁻¹ + N⁻¹]⁻¹ to δ̂_k
3. Transform to velocity field: v_k = -i(k/k²) H₀f δ_k (Fourier space relation)
4. Inverse FFT to obtain real-space velocity and density fields

**Uncertainty Quantification:**
- Produce ensemble of 100+ constrained realizations
- Report median and 68% confidence intervals for δ(r) and v(r) at each grid point
- Compute correlation function ξ(r) = ⟨δ(r₁)δ(r₁+r)⟩ from realizations, compare to linear theory prediction

#### **METHOD C: HIERARCHICAL BAYESIAN INFERENCE (virbius framework, Lavaux 2016)**

**Model Parameters:**
- Cosmology: {Ω_m, σ₈, H₀} with priors from Planck+BAO
- Distance calibration: zero-point offsets μ_TF, μ_FP, μ_SNe and scatter parameters σ_TF, σ_FP, σ_SNe for each method
- Velocity field: Fourier coefficients v_k on grid
- Density field: δ_k = (k²/H₀f k) v_k (coupled via linear theory)

**Sampling Strategy:**
- Hamiltonian Monte Carlo (HMC) using Stan or PyMC for efficient exploration of high-dimensional parameter space
- 4 chains × 2000 iterations after 1000 warmup steps
- Monitor convergence via Gelman-Rubin R̂ < 1.01 for all parameters
- Thinning factor 5 to reduce autocorrelation

**Selection Function Modeling:**
- For each survey, define magnitude-limited sample: P(include | M, z) = Θ(m_lim - M - 5log₁₀D - 25)
- Incorporate surface brightness selection for extended sources
- Handle survey footprint via HEALPix masks

**Posterior Analysis:**
- Extract marginal posteriors for δ(r) and v(r) at grid points
- Compute Bayesian evidence for model comparison (e.g., ΛCDM vs. modified gravity predictions)
- Derive constraints on fσ₈ and bulk flow amplitudes with full covariance

---

### 3. ZONE OF AVOIDANCE HANDLING

**Multi-Wavelength Integration:**
- Merge optical (2MASS), HI (WALLABY, FAST, Parkes HIZOA), and X-ray (CIZA cluster catalog) detections
- Cross-match sources within 30" using topcat-style positional association
- Prioritize HI redshifts (extinction-free) over optical when both available

**Incomplete Coverage Treatment:**
- Generate HEALPix mask (Nside=64) marking regions with <10% completeness relative to expected galaxy density from luminosity function
- In reconstruction, down-weight contributions from incomplete regions using inverse completeness as noise inflation factor
- For Bayesian methods, model ZoA as regions with inflated measurement uncertainty rather than missing data

**Predictive Filling:**
- Train conditional generative model (e.g., variational autoencoder) on unobscured regions to predict expected galaxy density in ZoA given surrounding large-scale structure
- Use predictions to set informative priors in masked regions, but maintain high uncertainty
- Validate approach on artificially masked regions in complete sky areas

---

### 4. COMPUTATIONAL EFFICIENCY REQUIREMENTS

**Performance Targets:**
- Process full CF4++ catalog (50,000+ galaxies) in <30 minutes on 16-core workstation
- POTENT reconstruction on 200³ grid: <5 minutes
- Bayesian WF single realization: <2 minutes
- HMC full posterior sampling: <4 hours for 8000 samples

**Optimization Strategies:**
- FFT operations using FFTW library with SIMD optimizations
- Sparse matrix storage for Poisson solver (only 7-point stencil in 3D)
- Parallel tempering for HMC to improve mixing
- GPU acceleration (CUDA/OpenCL) for:
  - 3D convolutions (Gaussian smoothing)
  - FFT transforms
  - Matrix operations in Bayesian linear algebra
- Memory-mapped arrays for large catalogs to avoid loading entire dataset

**Numerical Precision:**
- Use float64 for all scientific computations (distances, velocities, fields)
- Gravitational constant G = 4.301 × 10⁻⁶ (km/s)² Mpc/M_☉ (exact SI conversion)
- Hubble constant H₀ = 70 km/s/Mpc (adjustable, propagate uncertainty)
- Validate numerical stability: reconstructed fields should not change by >1% when doubling grid resolution or halving timestep in iterative solvers

---

### 5. VALIDATION AND TESTING FRAMEWORK

**Mock Catalog Generation:**
- Use publicly available N-body simulations (e.g., MultiDark Planck 2, Illustris-TNG) to extract halo catalogs
- Apply realistic distance errors by adding lognormal scatter matching TF/FP/SNe distributions
- Impose survey selection functions (magnitude limits, sky coverage) on mock data
- Generate 10 independent realizations to quantify reconstruction uncertainty

**Validation Tests:**
1. **Recovery accuracy**: Measure correlation coefficient r between reconstructed δ_rec and true δ_true; require r > 0.8 for 10 Mpc smoothing
2. **Bias quantification**: Compute ⟨δ_rec - δ_true⟩ averaged over volume; systematic bias should be <5% of rms(δ_true)
3. **Bulk flow consistency**: Extract bulk flow B(R) = ⟨v(r)⟩ within radius R from reconstructed field; compare to input bulk flow, require agreement within 50 km/s
4. **Power spectrum preservation**: Compute P_rec(k) from reconstructed field, compare to input P_true(k); power should be recovered to within 20% for k < 0.1 h/Mpc
5. **Malmquist bias handling**: Reconstruct from biased mock catalogs; residuals δ_rec - δ_true should be uncorrelated with density (slope consistent with zero)

**Comparison to Published Results:**
- Reproduce bulk flow of 315 ± 40 km/s at 150 Mpc from CF4++ (2025 paper)
- Match density peaks at Shapley Supercluster (SGL ≈ 30°, SGB ≈ 30°, D ≈ 200 Mpc) and Norma cluster
- Confirm Dipole Repeller void location (opposite CMB dipole direction)

**Unit Testing:**
- Test individual functions with known inputs:
  - Coordinate transformations: round-trip equatorial → supergalactic → equatorial should recover original to machine precision
  - Poisson solver: input δ(r) = cos(2πx/L), verify Φ(r) = -(L/2π)² cos(2πx/L)
  - Velocity divergence: verify ∇·v = numerical derivative of v agrees with analytical for smooth test functions
- Edge case handling:
  - Empty grid cells (no galaxies nearby)
  - Extreme measurement errors (σ_v > v_pec)
  - Boundary regions (r approaching grid edge)

---

### 6. OUTPUT DATA PRODUCTS

**Primary Outputs (FITS format, COSMO convention):**
1. **3D density field**: δ(x,y,z) on uniform grid, header metadata: {grid spacing, origin coordinates, cosmology parameters, smoothing scale}
2. **3D velocity field**: v_x(x,y,z), v_y(x,y,z), v_z(x,y,z) components in km/s
3. **Gravitational potential**: Φ(x,y,z) in (km/s)² units
4. **Uncertainty cubes**: σ_δ(x,y,z), σ_v(x,y,z) from Bayesian posterior or bootstrap resampling

**Derived Products:**
1. **Bulk flow profiles**: B(R), B_x(R), B_y(R), B_z(R) as function of scale R, saved as ASCII table
2. **Convergence/divergence maps**: ∇·v(x,y,z) showing infall/outflow regions
3. **Isodensity contours**: δ = 0.5, 1.0, 2.0 surfaces exported as polygon meshes (STL or OBJ format)
4. **Streamlines**: integrated v(r) trajectories from Local Group position toward convergence points

**Diagnostic Outputs:**
1. **Residual maps**: v_obs - v_model for each galaxy, check for systematic patterns
2. **χ² statistics**: per-galaxy, per-survey, and global to assess fit quality
3. **Correlation functions**: ξ(r) measured from reconstructed δ vs. linear theory prediction
4. **Power spectrum**: P(k) measured, compared to input cosmology

**Visualization:**
- Slice plots through supergalactic plane showing δ(x,y,z=0) with velocity vectors overlaid
- 3D volume rendering of density field with adjustable isosurface threshold
- Interactive widget: click galaxy → show distance, velocity, reconstruction residual
- Mollweide projection of sky distribution colored by bulk flow magnitude

---

### 7. SOFTWARE ARCHITECTURE

**Module Structure:**
```
velocity_reconstruction/
├── data_io/
│   ├── catalog_readers.py      # Ingest CF4, WALLABY, FAST, DESI formats
│   ├── coordinate_transforms.py # Equatorial ↔ Galactic ↔ Supergalactic
│   └── extinction.py           # SFD98 dust maps, CCM89 reddening law
├── preprocessing/
│   ├── peculiar_velocity.py    # Compute v_pec from D and z
│   ├── grouping.py             # FoF algorithm, group averaging
│   ├── malmquist.py            # Selection function modeling
│   └── quality_flags.py        # Outlier rejection, error validation
├── reconstruction/
│   ├── potent.py               # Bertschinger & Dekel algorithm
│   ├── wiener_filter.py        # BGc + WF implementation
│   ├── bayesian_inference.py   # HMC with Stan/PyMC interface
│   └── field_operators.py      # ∇, ∇², FFT utilities
├── zone_avoidance/
│   ├── multiwavelength_merge.py
│   ├── completeness_mask.py
│   └── predictive_filling.py
├── validation/
│   ├── mock_catalogs.py        # Generate from simulations
│   ├── metrics.py              # r, bias, P(k), ξ(r) calculators
│   └── comparison_tests.py     # Against published results
├── visualization/
│   ├── slices.py               # 2D cuts through 3D fields
│   ├── volume_render.py        # Isosurfaces, ray tracing
│   └── interactive.py          # Web-based dashboard
└── outputs/
    ├── fits_writer.py
    ├── derived_products.py
    └── diagnostics.py
```

**Dependencies (version requirements):**
- NumPy ≥1.24 (vectorized operations, FFT)
- SciPy ≥1.11 (sparse matrices, optimization, interpolation)
- Astropy ≥5.3 (cosmology, coordinates, FITS I/O, units)
- healpy ≥1.16 (HEALPix masks for sky coverage)
- pandas ≥2.0 (catalog manipulation, groupby operations)
- PyMC ≥5.0 or PyStan ≥3.0 (Bayesian inference)
- scikit-learn (for cross-validation in ML components if implemented)
- matplotlib ≥3.7, plotly ≥5.14 (visualization)
- Optional: CuPy ≥12.0 (GPU acceleration), numba ≥0.57 (JIT compilation)

**Configuration Management:**
- YAML config file specifying:
  - Input catalog paths
  - Cosmological parameters {Ω_m, σ₈, H₀, n_s}
  - Grid parameters {extent, resolution, smoothing scales}
  - Algorithm selection (POTENT | WF | Bayesian | all)
  - Computational settings (num cores, GPU enable, precision)
  - Output directory and product selection
- Schema validation on config file load
- Version control config alongside code (Git)

**Logging and Provenance:**
- Log all parameters, data versions, code version (Git commit hash) to outputs
- Execution time profiling for each major step
- Warning system for:
  - Galaxies with v_pec < -500 km/s (unphysical infall)
  - Grid cells with no data within smoothing radius
  - Numerical instabilities (e.g., FFT aliasing, Poisson solver non-convergence)
  - Cosmology parameter values outside Planck 2σ

---

### 8. ERROR HANDLING AND EDGE CASES

**Data Quality Issues:**
- Missing redshifts: Skip galaxy, log warning, continue
- Negative peculiar velocities beyond -500 km/s: Flag as potential outlier, include with down-weighting
- Duplicate entries: Average if distance methods differ by <20%, else flag for manual review
- Coordinate singularities (poles): Use Cartesian internally, only convert to spherical for I/O

**Numerical Stability:**
- FFT aliasing: Apply anti-aliasing filter (low-pass k < k_Nyquist/2) before inverse transforms
- Poisson solver divergence: Implement successive over-relaxation with adaptive relaxation parameter ω, max 10,000 iterations
- Matrix conditioning: For ill-conditioned linear systems in Bayesian inference, use Tikhonov regularization with λ chosen via L-curve criterion
- Zero division: In density-to-velocity conversion, impose floor δ_min = -0.99 to prevent singularities in underdense regions

**Boundary Conditions:**
- Grid edges: Apply periodic boundary conditions for FFT, but flag as "extrapolation region" in outputs if |r| > 0.9 × r_max
- Survey boundaries: Taper window function smoothly over 10 Mpc at survey edge to avoid sharp discontinuities
- Zone of Avoidance: As described in §3, treat as high-uncertainty regions rather than hard mask

**Computational Resource Limits:**
- Memory overflow: If grid too large for RAM, implement out-of-core FFT using memory-mapped files
- Time limits: Implement checkpointing for HMC; save chain state every 100 iterations, allow resume
- Convergence failure: If R̂ > 1.05 after max iterations, log warning, continue with caveat flag in outputs

---

### 9. DOCUMENTATION REQUIREMENTS

**Scientific Documentation:**
- Document for each method: full mathematical derivation, assumptions, applicability limits
- Appendix listing all coordinate systems, transformations, and conventions (supergalactic longitude zero-point, velocity sign convention)
- Comparison table of methods: POTENT vs. WF vs. Bayesian (computational cost, assumptions, accuracy)
- Bibliography with DOIs for all referenced papers

**Technical Documentation:**
- Function-level docstrings following NumPy style guide:
  - Parameters with types and units
  - Returns with types and units  
  - Raises (exceptions and conditions)
  - Examples with expected outputs
  - Notes on algorithm complexity (O(N²), O(N log N), etc.)
- Module-level overview explaining scientific context and role in pipeline
- README with:
  - Installation instructions (conda environment YAML)
  - Quickstart: minimal working example from catalog to density map
  - Configuration guide: explanation of all YAML parameters
  - Performance benchmarks: expected runtime vs. dataset size

**User Guide:**
- Tutorial notebooks demonstrating:
  1. Loading and exploring CF4 catalog
  2. Computing peculiar velocities
  3. Running POTENT reconstruction
  4. Visualizing results
  5. Comparing to published bulk flows
- FAQ addressing common issues:
  - How to handle missing data
  - Choosing smoothing scale
  - Interpreting uncertainty estimates
  - When to trust reconstruction (signal-to-noise criteria)

**Developer Guide:**
- Code style: PEP 8 compliance, max line length 88 (Black formatter)
- Testing: pytest framework, target >80% coverage, include integration tests
- Contribution workflow: fork, branch, PR with review checklist
- Performance profiling: guide to using cProfile and line_profiler
- Adding new distance indicators: interface requirements, example implementation

---

### 10. DEPLOYMENT AND REPRODUCIBILITY

**Containerization:**
- Dockerfile defining complete environment: base image (python:3.11-slim), dependencies, code, config
- Docker Compose for multi-container setup if distributed computing implemented
- Singularity recipe for HPC environments without Docker support

**Data Management:**
- Provide script to download public catalogs from:
  - CF4: NED/IPAC
  - WALLABY: CSIRO ASKAP archive
  - FAST: China-VO
  - DESI: NERSC portal
- Checksum validation (SHA256) for downloaded files
- Cached preprocessed data to avoid re-computation (versioned by input catalog version + preprocessing code hash)

**Version Control:**
- Semantic versioning (MAJOR.MINOR.PATCH)
- CHANGELOG documenting all changes with scientific impact
- Tagged releases corresponding to published results or major algorithmic updates
- Git LFS for large data files (mock catalogs, test data)

**Continuous Integration:**
- Automated testing on push: unit tests, integration tests, mock catalog validation
- Nightly regression tests against reference outputs
- Performance benchmarks tracked over time to detect degradation
- Documentation build verification (Sphinx or similar)

**Licensing:**
- Open source license (MIT or BSD-3) for code
- Data: cite original survey papers, respect usage terms (typically public release after proprietary period)
- Third-party dependencies: verify license compatibility

---

### 11. FUTURE EXTENSIBILITY

**Design for Upcoming Data:**
- Euclid DR1 (October 2026): placeholder functions for photometric redshift catalog ingestion, weak lensing shear-to-density conversion
- SKA pathfinder integration: HI intensity mapping cube processing, RSD modeling
- JWST NIRCam ZoA: template for high-angular-resolution source extraction

**Algorithm Enhancements:**
- Plug-in architecture for new reconstruction methods (e.g., U-Net ML, BORG nonlinear)
- Interface specification: input (galaxy catalog with {RA, Dec, z, D, σ_D}), output (δ, v fields on grid)
- Benchmark suite for comparing methods: standardized mocks, metrics, plots

**Science Extensions:**
- Gravitational lensing convergence κ from projected density ∫ δ(r) dr
- Kinetic Sunyaev-Zel'dovich signal prediction from velocity field
- Constrained simulations: use reconstructed initial conditions as input to N-body code
- Growth rate fσ₈(z) inference from redshift evolution of velocity field

---

## DELIVERABLES

1. **Production software package** meeting all specifications above
2. **Validation report** documenting performance on mocks and comparison to published results, including all figures and tables
3. **Analysis notebooks** reproducing key science results: bulk flow measurements, Shapley/Norma/Dipole Repeller identification, velocity field convergence analysis
4. **User documentation** (installation, tutorials, API reference)
5. **Scientific publication draft** describing methodology, validation, and results on real data, formatted for submission to Astrophysical Journal Supplement Series

---

## SUCCESS CRITERIA

**Functional:**
- Successfully ingests CF4++, WALLABY, FAST, DESI catalogs without manual intervention
- Produces density and velocity field reconstructions using all three methods (POTENT, WF, Bayesian)
- Generates all specified output products automatically
- Handles ZoA regions appropriately
- Runs within performance targets on standard workstation

**Scientific:**
- Recovers true density field from mocks with r > 0.8
- Reproduces published bulk flow 315 ± 40 km/s at 150 Mpc
- Identifies expected structures (Shapley, Norma, Dipole Repeller) in reconstructed field
- Uncertainty estimates are well-calibrated: 68% of true values fall within 1σ intervals

**Engineering:**
- >80% test coverage, all tests pass
- Documentation complete and accurate
- Code review by independent astronomer confirms scientific correctness
- Runs reproducibly: same inputs → identical outputs across machines
- Performance profiling shows no obvious inefficiencies (>90% time in scientific computation, <10% overhead)

---

