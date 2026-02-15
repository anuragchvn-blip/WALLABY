# SCIENTIFIC SPECIFICATION: FIXING POTENT VELOCITY FIELD RECONSTRUCTION

## ROOT CAUSE DIAGNOSIS

**The Problem:**
POTENT reconstruction produces velocity field with:
- |v|_max ≈ 293 km/s (Expected: 400-600 km/s for bulk flow)
- |v|_mean ≈ 0 km/s (Expected: 200-400 km/s)
- Field variance too low by factor of 2-3

**Why This Matters:**
When you compute g = -∇Φ from weak velocity field, you get weak gravity field, which predicts weak velocities. The closure test shows r=0.11 not because the physics is wrong, but because you're comparing two independently-broken weak fields that happen to correlate slightly by chance.

**The True Test:**
Correlation between RECONSTRUCTED v_field and MEASURED v_radial should be r>0.7 BEFORE any gravity calculation. If this fails, everything downstream fails.

---

## POTENT METHOD REQUIREMENTS (Bertschinger & Dekel 1989)

### Critical Parameters That Must Be Correct:

**1. Growth Rate Parameter**
```
β = f/b where:
  f = Ω_m^0.55 = 0.55 (ΛCDM, Ω_m=0.3)
  b = galaxy bias ≈ 1.0-1.5 (depends on tracer)
  
Standard: β ≈ 0.5 (assuming b≈1.1)

VERIFY: Are you using β in velocity-density relation:
  δ = -(β H₀)^-1 ∇·v
```

**2. Smoothing Scale**
```
Current: σ = 12 Mpc Gaussian
Bertschinger & Dekel: σ = 10-12 h^-1 Mpc

For h=0.7: σ_physical = 14-17 Mpc

Too large smoothing → suppresses small-scale velocities
Too small smoothing → noise dominates

OPTIMAL: σ = 10 Mpc (physical, not h^-1)
```

**3. Hubble Constant Consistency**
```
All velocity calculations must use SAME H₀:
  v_pec = c×z_obs - H₀×D
  ∇·v = -β H₀ δ
  
VERIFY: Same H₀=70 km/s/Mpc used throughout?
Or mixing H₀=70 in one place, H₀=100 h in another?
```

**4. Radial to 3D Reconstruction**
```
POTENT assumes: v = -∇Φ_v (irrotational flow)

From radial measurements v_r = v·r̂:
Must solve: ∇²Φ_v = -∇·v
Subject to: ∂Φ_v/∂r = -v_r (observed)

CRITICAL: Boundary conditions matter!
- Φ_v → 0 as r → ∞
- ∂Φ_v/∂r → 0 at grid boundaries (not Φ_v = 0!)
```

**5. Malmquist Bias Correction**
```
Before reconstruction, must correct:
  v_true = v_obs - ⟨Δv_Malmquist⟩
  
Where bias depends on:
  - Distance indicator (TF worse than SNe)
  - Distance from observer
  - Local density field (inhomogeneous bias)
  
VERIFY: Applied BGc (Bias Gaussianization correction)?
Or at minimum: distance-dependent bias removal?
```

---

## DIAGNOSTIC CHECKLIST

### Test 1: Input Data Quality
```
BEFORE reconstruction, verify:

1. Peculiar velocity distribution:
   - Median v_pec ≈ 200-400 km/s ✓ (you have 486 km/s)
   - Std v_pec ≈ 400-600 km/s
   - NOT centered at zero (bulk flow!)
   
2. Radial velocity per galaxy:
   Check: v_r,i = v_pec,i × cos(θ_i)
   where θ_i = angle between galaxy position and velocity
   
   If all velocities radial (θ=0): v_r = v_pec
   This is WRONG - need to project measured v_pec onto sightline
   
3. Galaxy positions:
   Must be in SAME coordinate system as reconstruction grid
   Supergalactic? Equatorial? Galactic?
   Mismatch → reconstruction in wrong orientation
```

### Test 2: Grid Assignment
```
For each galaxy at (x,y,z) with v_r:

1. Assign to NEAREST grid cell
2. If multiple galaxies per cell: AVERAGE their v_r
3. Weight by 1/σ_v² if uncertainties available

COMMON ERROR:
  Assigning v_r to grid WITHOUT accounting for sightline direction
  
CORRECT:
  Grid stores: ∂Φ_v/∂r = -v_r along sightline to that cell
  NOT just v_r value
```

### Test 3: Smoothing Application
```
Apply Gaussian kernel:
  W(r) = (2πσ²)^(-3/2) exp(-r²/2σ²)
  
To RADIAL VELOCITY FIELD, not final 3D velocities

Smoothed: v_r^smooth(x) = ∫ v_r(x') W(|x-x'|) d³x'

VERIFY: Kernel normalization ∫ W d³x = 1
VERIFY: Effective smoothing scale matches σ (check FWHM)
```

### Test 4: Velocity Potential Solution
```
Solve: ∇²Φ_v = f(x,y,z)

Where f is constructed from smoothed v_r data

METHOD OPTIONS:
A. FFT Poisson solver (if using periodic boundaries)
B. Multigrid relaxation (if using realistic boundaries)
C. Green's function convolution

CRITICAL: Check ∇²Φ_v actually reproduces ∇·v from data
```

### Test 5: Gradient Calculation
```
v = -∇Φ_v

Numerical gradient:
  v_x = -(Φ[i+1,j,k] - Φ[i-1,j,k]) / (2Δx)
  
VERIFY: Grid spacing Δx used correctly
VERIFY: Boundary cells handled (forward/backward difference)
VERIFY: Sign convention (v = -∇Φ, not +∇Φ)
```

---

## EXPECTED OUTCOMES IF CORRECT

### Validation Metrics:

**1. Velocity Field Statistics**
```
AFTER reconstruction:
  |v|_mean = 200-400 km/s  (matches bulk flow)
  |v|_max = 600-1000 km/s  (cluster infall regions)
  |v|_median = 150-300 km/s
  
Direction: Peak toward Shapley/GA region (SGL~300°, SGB~20°)
```

**2. Divergence-Density Relation**
```
From continuity: δ = -(β H₀)^-1 ∇·v

Check: Correlation(δ_reconstructed, δ_from_galaxies) > 0.6

If low: β parameter wrong OR smoothing inappropriate
```

**3. Curl Test**
```
POTENT assumes irrotational: ∇×v = 0

After reconstruction:
  |∇×v| / |∇·v| < 0.1 everywhere
  
If ratio > 0.3: violation of irrotational assumption
Indicates: Non-linear effects, survey edge effects, or bugs
```

**4. Self-Consistency**
```
CRITICAL TEST (before any gravity calculation):

For each galaxy:
  1. Extract v_field at galaxy position (interpolate)
  2. Project onto sightline: v_pred,i = v_field · r̂_i
  3. Compare to measured: v_obs,i
  
Correlation: r(v_pred, v_obs) > 0.6
RMS: σ(v_pred - v_obs) < 500 km/s

IF THIS FAILS: POTENT reconstruction is broken
No point proceeding to gravity calculation
```

---

## SYSTEMATIC ERROR SOURCES

### Issue 1: Distance Errors Propagate to Velocities
```
If distance D has 20% error (typical for TF):
  σ_D/D = 0.2
  
Then: σ_v = H₀ × σ_D ≈ 70 × 0.2D = 14D km/s

At D=100 Mpc: σ_v ≈ 1400 km/s (huge!)

SOLUTION: Weight galaxies by 1/σ_v² in reconstruction
Low-weight distant/uncertain galaxies
High-weight nearby/precise galaxies
```

### Issue 2: Sparse Sampling Creates Bias
```
Galaxies not uniformly distributed:
  - Clusters: many galaxies
  - Voids: few galaxies
  
POTENT smoothing averages over empty regions → artificially suppresses velocities

SOLUTION: 
  - Use density-dependent smoothing (smaller σ in clusters)
  - OR: Explicitly model selection function
  - OR: Use Bayesian methods (virbius) instead of POTENT
```

### Issue 3: Zone of Avoidance
```
Missing data at |b| < 10° creates systematic errors

If ZOA intersects major attractor (Great Attractor does!):
  - Missing radial velocities in critical direction
  - Reconstruction under-estimates velocities there
  
SOLUTION: Inflate uncertainties in ZOA-adjacent cells
OR: Use multi-wavelength data (HI, X-ray) to fill gaps
```

### Issue 4: Coordinate System Errors
```
MOST COMMON BUG:

Galaxy positions in (RA, Dec, D) → Cartesian (x,y,z)
Reconstruction in Supergalactic coordinates
Gravity field back to (RA, Dec) for comparison

If coordinate transforms inconsistent:
  - Rotation/flip of reconstructed field
  - Magnitude correct but direction wrong
  
VERIFY: All coordinates in SAME SYSTEM throughout
Recommend: Supergalactic (X,Y,Z) for local universe studies
```

---

## ALTERNATIVE APPROACHES TO CONSIDER

### If POTENT Continues to Fail:

**Option A: Wiener Filter (Simpler)**
```
Assumes: Known power spectrum P(k)
Direct linear inversion of radial → 3D velocities
Less prone to numerical errors than POTENT

SOFTWARE: Use published WF codes from CosmicFlows team
```

**Option B: Bayesian (virbius, BORG)**
```
Jointly fits:
  - 3D velocity field
  - Distance calibration
  - Malmquist bias
  - Cosmological parameters
  
More robust but computationally expensive
Properly handles uncertainties

RECOMMENDED if POTENT unfixable
```

**Option C: Direct Density Reconstruction**
```
Skip velocity field entirely:

Galaxy positions → density field δ(r) [via kernel density estimation]
δ(r) → gravitational potential Φ [Poisson solver]  
Φ → gravity g = -∇Φ
g → predicted velocities v_pred = (f/H₀) g

Compare v_pred to measured v_obs

This bypasses POTENT's radial→3D reconstruction problems
```

---

## RECOMMENDED ACTION PLAN

### Phase 1: Diagnose POTENT (1 week)

**Run all 5 diagnostic tests above**

**Critical outputs to examine:**
1. Histogram of |v_field| values → should peak at 200-400 km/s
2. Vector field plot → should show coherent flow toward GA/Shapley
3. v_field vs v_obs scatter (BEFORE gravity) → r should be >0.6
4. Curl magnitude map → should be <10% of divergence
5. Boundary behavior → velocities shouldn't drop sharply at edges

**Likely findings:**
- Smoothing scale wrong (try σ=8, 10, 15 Mpc)
- β parameter wrong (try β=0.4, 0.5, 0.6)
- Boundary conditions wrong (Φ_v vs ∂Φ_v/∂n at edges)
- Coordinate system mismatch
- Missing Malmquist correction

### Phase 2: Fix Most Probable Issue (2-3 days)

**Based on diagnostics, implement ONE fix:**

If |v|_mean too low → increase β from 0.5 to 0.6-0.7
If |v| has small-scale noise → increase smoothing σ
If edge artifacts → fix boundary conditions
If r<0.3 in self-test → coordinate system bug

**Re-run self-consistency test**
Target: r(v_field_projected, v_obs) > 0.6

### Phase 3: Validate Physics (1 day)

**Once velocity field reasonable:**

Calculate g = -∇Φ from density
Predict v from g
Compare to measured v
Should get r > 0.7 if velocity field correct

### Phase 4: Scientific Analysis (ongoing)

**Only proceed if r>0.6 achieved:**

Catalog attractors
Measure bulk flows
Closure test
Push-pull decomposition
Compare to published structures

---

## SUCCESS CRITERIA

### Minimum Viable Reconstruction:
- |v|_mean > 200 km/s ✓
- Correlation r(v_field, v_obs) > 0.5 ✓
- Curl/div ratio < 0.2 ✓
- Bulk flow toward Shapley region ✓

### Publication Quality:
- |v|_mean = 300-400 km/s ✓
- r > 0.7 ✓
- Curl/div < 0.1 ✓
- Matches published bulk flow 315±40 km/s ✓
- Identifies known attractors within 30 Mpc ✓

---

## BOTTOM LINE

**Your current issue:**
```
POTENT → weak v_field (|v|≈300 km/s, should be 600)
     ↓
  g = -∇Φ also weak
     ↓
  v_pred from g also weak
     ↓
  r = 0.11 (two weak fields barely correlate)
```

**Must fix at source:**
```
FIX POTENT RECONSTRUCTION FIRST
     ↓
Strong v_field (|v|≈600 km/s) ✓
     ↓
Correct g field
     ↓
Accurate v_pred
     ↓
r > 0.7 achieved
```

**The closure test will ONLY work if the velocity field reconstruction works.**

**Test BEFORE gravity: Does v_field match v_obs? If no, stop and fix POTENT.**