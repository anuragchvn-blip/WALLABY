# WALLABY-VR: Velocity Reconstruction Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A high-performance cosmic velocity field reconstruction pipeline featuring the novel **NS-VGR (Non-linear Saturated Velocity-Gravity Relation)** formula for improved peculiar velocity predictions.

## Overview

WALLABY-VR reconstructs three-dimensional velocity fields from galaxy peculiar velocity surveys. The pipeline processes distance measurements from Tully-Fisher, Fundamental Plane, and Type Ia supernova observations to map the cosmic flow field in the local universe.

**Key Achievement:** Our novel NS-VGR formula achieves **r = 0.46** correlation with observed velocities, a **55% improvement** over standard linear theory (r = 0.30).

## The NS-VGR Formula

We introduce a novel velocity-gravity relation that addresses the failure of linear perturbation theory in non-linear regimes:

$$\mathbf{v}_{\mathrm{NS-VGR}} = f(\Omega_m) H_0 \left[ \mathcal{S}(\delta) \cdot \frac{\mathbf{g}}{H_0^2} + \gamma \mathcal{L}_{\mathrm{NL}} \frac{\nabla\delta}{1+\delta} \right]$$

### Formula Components

| Component | Expression | Physical Meaning |
|-----------|------------|------------------|
| **Saturation Kernel** | $\mathcal{S}(\delta) = \exp(-\|\delta\|/\delta_{\mathrm{crit}})$ | Prevents velocity divergence in clusters |
| **Gradient Entrainment** | $\gamma \mathcal{L}_{\mathrm{NL}} \nabla\delta/(1+\delta)$ | Captures void expansion dynamics |
| **Critical Threshold** | $\delta_{\mathrm{crit}} = 1.68$ | Spherical collapse virialization threshold |
| **Entrainment Coupling** | $\gamma = 0.4$ | Non-linear momentum flux strength |
| **Non-linear Scale** | $\mathcal{L}_{\mathrm{NL}} = 5$ Mpc | Shell-crossing characteristic length |

### Why NS-VGR Works

1. **Cluster Saturation:** Linear theory predicts unbounded velocities in overdense regions. The saturation kernel exponentially suppresses contributions where structures have virialized.

2. **Void Expansion:** Standard methods ignore momentum flux from density gradients. The entrainment term captures super-linear outflows from cosmic voids.

## Performance Results

| Metric | Linear Theory | NS-VGR | Improvement |
|--------|---------------|--------|-------------|
| Pearson Correlation | 0.297 | **0.462** | +55.3% |
| Improvement (Δr) | — | **0.165** | Exceeds 0.10 target |
| Execution Time | 1.92s | 1.91s | Equivalent |
| Computational Complexity | O(N log N) | O(N log N) | Optimal |

Validated on **CosmicFlows-4** catalog with 9,355 galaxies.

## Installation

```bash
# Clone the repository
git clone https://github.com/anuragchvn-blip/WALLABY.git
cd WALLABY

# Install dependencies
pip install -r velocity_reconstruction/requirements.txt
```

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Astropy
- Matplotlib

## Quick Start

### Run NS-VGR Reconstruction

```python
from ns_vgr_engine.formula import NSVGRReconstructor

# Initialize reconstructor
reconstructor = NSVGRReconstructor(
    grid_size=128,
    box_extent=100.0,  # Mpc
    smoothing_sigma=3.0,
    delta_crit=1.68,
    gamma=0.4,
    l_nl=5.0
)

# Load data and reconstruct
positions = ...  # Galaxy positions in Mpc
velocities = ...  # Observed peculiar velocities in km/s
errors = ...  # Distance errors

vx, vy, vz = reconstructor.reconstruct(positions, velocities, errors)
```

### Run Benchmark on CosmicFlows-4

```bash
cd ns_vgr_engine
python realdata_test.py
```

Expected output:
```
=== NS-VGR Real Data Test on CosmicFlows-4 ===
Loaded 9355 galaxies from CosmicFlows-4
Grid: 128^3, Extent: ±100.0 Mpc, Smoothing: 3.0 Mpc

Results:
  Baseline correlation: 0.2973
  NS-VGR correlation:   0.4615
  Improvement (Δr):     0.1642
  Execution time:       1.91s
```

## Project Structure

```
WALLABY/
├── ns_vgr_engine/           # NS-VGR implementation
│   ├── formula.py           # Core NSVGRReconstructor class
│   ├── realdata_test.py     # CosmicFlows-4 benchmark
│   ├── RESULTS.md           # Validation results
│   └── SUCCESS.md           # Achievement summary
├── velocity_reconstruction/ # Base reconstruction modules
│   ├── config.py            # Configuration parameters
│   ├── pipeline.py          # Main pipeline
│   ├── data_io/             # Data loading utilities
│   ├── preprocessing/       # Quality filtering, grouping
│   ├── reconstruction/      # Field operators, Poisson solver
│   ├── validation/          # Metrics, mock catalogs
│   └── visualization/       # Plotting utilities
├── paper/                   # Publication materials
│   ├── nsvgr_paper.tex      # LaTeX manuscript
│   └── references_nsvgr.bib # Bibliography
├── data/                    # Input catalogs
│   └── cosmicflows4.csv     # CosmicFlows-4 data
└── results/                 # Output products
    ├── velocity_vx.npy      # Reconstructed velocity field
    ├── velocity_vy.npy
    ├── velocity_vz.npy
    └── density.npy          # Reconstructed density field
```

## Algorithm Pipeline

```
Input: Galaxy catalog {(RA, Dec, distance, v_pec, σ_d)}
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Coordinate Transformation    │
              │  Equatorial → Supergalactic   │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Grid Assignment (CIC)        │
              │  Quality-weighted gridding    │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Gaussian Smoothing           │
              │  σ = 3 Mpc (FFT-based)        │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Density Reconstruction       │
              │  δ = -∇·v / (f H₀)            │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Poisson Solver (FFT)         │
              │  ∇²Φ = δ → g = -∇Φ            │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  NS-VGR Formula Application   │
              │  Saturation + Entrainment     │
              └───────────────────────────────┘
                              │
                              ▼
Output: 3D velocity field v(x, y, z)
```

## Configuration

Key parameters in `velocity_reconstruction/config.py`:

```python
# Grid settings
GRID_SIZE = 128
BOX_EXTENT = 100.0  # Mpc

# NS-VGR parameters
SMOOTHING_SIGMA = 3.0      # Mpc (critical for performance)
NS_VGR_DELTA_CRIT = 1.68   # Spherical collapse threshold
NS_VGR_GAMMA = 0.4         # Entrainment coupling
NS_VGR_L_NL = 5.0          # Non-linear scale (Mpc)

# Cosmology
H0 = 70.0                  # km/s/Mpc
OMEGA_M = 0.315
```

## Citation

If you use NS-VGR in your research, please cite:

```bibtex
@article{Chavan2026,
  author = {Chavan, Anurag},
  title = {NS-VGR: A Non-linear Saturated Velocity-Gravity Relation 
           for Cosmic Velocity Field Reconstruction},
  journal = {arXiv preprint},
  year = {2026},
  eprint = {2602.xxxxx},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.CO}
}
```

## Related Work

- [CosmicFlows-4](https://cosmicflows.iap.fr/) - Tully et al. (2023)
- [POTENT](https://ui.adsabs.harvard.edu/abs/1990ApJ...364..349D) - Dekel, Bertschinger & Faber (1990)
- [WALLABY Survey](https://wallaby-survey.org/) - Koribalski et al. (2020)

## Author

**Anurag Chavan**  
Email: anuragchvn1@gmail.com

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- CosmicFlows team for the publicly available peculiar velocity catalog
- WALLABY collaboration for survey data and infrastructure
- NASA Astrophysics Data System for literature access
