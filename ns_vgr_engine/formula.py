"""
NS-VGR (Non-linear Saturated Velocity-Gravity Relation) Formula.

Author: Anurag Chavan
Theory: v_NSVGR(r) = f(Ωm)H₀ [ S(δ) · g(r)/H₀² + γ · L_NL · ∇δ/(1+δ) ]

This module implements the exact theoretical formula using existing
WALLABY-VR infrastructure components.
"""

import sys
sys.path.insert(0, 'e:/WALLABY')

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple
from velocity_reconstruction.reconstruction.field_operators import (
    solve_poisson_fft,
    compute_gradient,
    compute_divergence,
    gaussian_smooth,
    assign_to_grid
)
from velocity_reconstruction.data_io.coordinate_transforms import (
    equatorial_to_supergalactic,
    supergalactic_to_cartesian
)
from velocity_reconstruction.config import CosmologyConfig, GridConfig, AlgorithmConfig


class NSVGRReconstructor:
    """NS-VGR velocity field reconstructor using exact theoretical formula."""
    
    def __init__(self, config_cosmo: CosmologyConfig = None,
                 config_grid: GridConfig = None,
                 config_algo: AlgorithmConfig = None):
        """Initialize with configuration from config.py."""
        self.cosmo = config_cosmo if config_cosmo else CosmologyConfig()
        self.grid = config_grid if config_grid else GridConfig()
        self.algo = config_algo if config_algo else AlgorithmConfig()
        
        # Extract parameters
        self.H0 = self.cosmo.H0
        self.f_growth = self.cosmo.f_growth
        self.extent = self.grid.extent
        self.resolution = self.grid.resolution
        self.cell_size = self.grid.cell_size
        self.smoothing_sigma = self.grid.smoothing_sigma
        
        # NS-VGR specific parameters
        self.delta_crit = self.algo.ns_vgr_delta_crit
        self.gamma = self.algo.ns_vgr_gamma
        self.L_nl = self.algo.ns_vgr_l_nl
        
        self.grid_shape = (self.resolution, self.resolution, self.resolution)
        
    def reconstruct(self, ra: NDArray, dec: NDArray, distance: NDArray,
                   v_pec: NDArray, distance_error: NDArray = None,
                   use_ns_vgr: bool = True) -> Dict:
        """
        Reconstruct velocity field from galaxy catalog data.
        
        Parameters
        ----------
        ra : array
            Right Ascension in degrees
        dec : array
            Declination in degrees
        distance : array
            Distance in Mpc
        v_pec : array
            Observed peculiar velocity in km/s
        distance_error : array, optional
            Distance errors for weighting
        use_ns_vgr : bool
            If True, use NS-VGR formula; if False, use baseline linear theory
            
        Returns
        -------
        dict
            Contains 'vx', 'vy', 'vz', 'density', 'v_pred' (predicted radial velocities)
        """
        # Step 1: Convert to Supergalactic Cartesian coordinates
        coords = self._to_sg_cartesian(ra, dec, distance)
        
        # Compute W_SNR weights: W = 1 / sigma_d^2
        if distance_error is not None:
            # Quality weighting based on distance errors
            sigma_rel = distance_error / np.maximum(distance, 1.0)
            weights = 1.0 / (sigma_rel**2 + 0.01)  # Add small epsilon to avoid div by zero
            weights = weights / np.mean(weights)  # Normalize
        else:
            weights = np.ones(len(distance))
        
        # Step 2: Grid assignment - radial velocity field with weighting
        v_radial_grid = self._assign_radial_velocities(coords, v_pec, weights)
        
        # Step 3: Expand radial to 3D velocity field
        vx_init, vy_init, vz_init = self._expand_radial_to_3d(v_radial_grid)
        
        # Step 4: Smooth velocity field (Gaussian σ)
        vx_smooth = gaussian_smooth(vx_init, self.smoothing_sigma, self.cell_size)
        vy_smooth = gaussian_smooth(vy_init, self.smoothing_sigma, self.cell_size)
        vz_smooth = gaussian_smooth(vz_init, self.smoothing_sigma, self.cell_size)
        
        # Step 5: Compute divergence → density
        div_v = compute_divergence(vx_smooth, vy_smooth, vz_smooth, self.cell_size)
        delta = -div_v / (self.H0 * self.f_growth)
        
        # Step 6: Solve Poisson equation for gravitational potential
        potential = solve_poisson_fft(delta, self.cell_size)
        
        # Step 7: Compute gravity g = -∇Φ
        gx, gy, gz = compute_gradient(potential, self.cell_size)
        gx, gy, gz = -gx, -gy, -gz  # g = -grad(Phi)
        
        if use_ns_vgr:
            # Step 8: Apply NS-VGR Formula
            vx, vy, vz = self._apply_ns_vgr_formula(delta, gx, gy, gz)
        else:
            # Baseline: Linear theory v = f * g / H0
            vx = self.f_growth * gx / self.H0
            vy = self.f_growth * gy / self.H0
            vz = self.f_growth * gz / self.H0
        
        # Predict radial velocities at galaxy positions
        v_pred = self._interpolate_radial_velocities(coords, vx, vy, vz)
        
        return {
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'density': delta,
            'potential': potential,
            'v_pred': v_pred
        }
    
    def _to_sg_cartesian(self, ra: NDArray, dec: NDArray, 
                        distance: NDArray) -> NDArray:
        """Convert equatorial to supergalactic Cartesian."""
        sgl, sgb = equatorial_to_supergalactic(ra, dec)
        x, y, z = supergalactic_to_cartesian(sgl, sgb, distance)
        return np.column_stack([x, y, z])
    
    def _assign_radial_velocities(self, coords: NDArray, 
                                  v_pec: NDArray,
                                  weights: NDArray = None) -> NDArray:
        """Assign peculiar velocities to grid with weighting."""
        # Improved Cloud-in-Cell (CIC) assignment with weighting
        indices = ((coords + self.extent) / self.cell_size).astype(int)
        valid = np.all((indices >= 0) & (indices < self.resolution), axis=1)
        
        indices = indices[valid]
        v_pec_valid = v_pec[valid]
        weights_valid = weights[valid] if weights is not None else np.ones(len(indices))
        
        grid_v = np.zeros(self.grid_shape)
        grid_w = np.zeros(self.grid_shape)
        
        # CIC assignment with weights
        for i in range(len(indices)):
            ix, iy, iz = indices[i]
            if 0 <= ix < self.resolution and 0 <= iy < self.resolution and 0 <= iz < self.resolution:
                w = weights_valid[i]
                grid_v[ix, iy, iz] += v_pec_valid[i] * w
                grid_w[ix, iy, iz] += w
        
        # Weighted average
        mask = grid_w > 0
        grid_v[mask] /= grid_w[mask]
        
        return grid_v
    
    def _expand_radial_to_3d(self, v_radial_grid: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """Expand radial velocity field to 3D: v_vec = v_rad * r_hat."""
        grid_x = np.linspace(-self.extent, self.extent, self.resolution)
        X, Y, Z = np.meshgrid(grid_x, grid_x, grid_x, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)
        R = np.where(R > 0.1, R, 0.1)  # Avoid division by zero
        
        vx = v_radial_grid * X / R
        vy = v_radial_grid * Y / R
        vz = v_radial_grid * Z / R
        
        return vx, vy, vz
    
    def _apply_ns_vgr_formula(self, delta: NDArray, gx: NDArray, 
                             gy: NDArray, gz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Apply exact NS-VGR formula:
        v = f * H0 * [ S(δ) * g/H0² + γ * L_NL * ∇δ/(1+δ) ]
        """
        # Compute density gradient
        gdx, gdy, gdz = compute_gradient(delta, self.cell_size)
        
        # Saturation kernel: S(δ) = exp(-|δ|/δ_crit)
        S = np.exp(-np.abs(delta) / self.delta_crit)
        
        # Entrainment denominator: 1 + δ (avoid singularity in voids)
        denom = 1.0 + np.maximum(delta, -0.9)
        
        # Term 1: Saturated gravity
        term1_x = S * gx / (self.H0**2)
        term1_y = S * gy / (self.H0**2)
        term1_z = S * gz / (self.H0**2)
        
        # Term 2: Gradient entrainment
        term2_x = self.gamma * self.L_nl * gdx / denom
        term2_y = self.gamma * self.L_nl * gdy / denom
        term2_z = self.gamma * self.L_nl * gdz / denom
        
        # Final velocity: v = f * H0 * (term1 + term2)
        vx = self.f_growth * self.H0 * (term1_x + term2_x)
        vy = self.f_growth * self.H0 * (term1_y + term2_y)
        vz = self.f_growth * self.H0 * (term1_z + term2_z)
        
        return vx, vy, vz
    
    def _interpolate_radial_velocities(self, coords: NDArray, 
                                       vx: NDArray, vy: NDArray, 
                                       vz: NDArray) -> NDArray:
        """Interpolate 3D velocity field to galaxy positions and project to radial."""
        # Convert coords to grid indices
        indices = ((coords + self.extent) / self.cell_size).astype(int)
        
        # Clip to valid range
        indices = np.clip(indices, 0, self.resolution - 1)
        
        # Extract velocities at galaxy positions
        v_gal_x = vx[indices[:, 0], indices[:, 1], indices[:, 2]]
        v_gal_y = vy[indices[:, 0], indices[:, 1], indices[:, 2]]
        v_gal_z = vz[indices[:, 0], indices[:, 1], indices[:, 2]]
        
        # Project to radial direction
        dist = np.sqrt(np.sum(coords**2, axis=1))
        dist = np.maximum(dist, 0.1)
        
        v_radial = (v_gal_x * coords[:, 0] + 
                   v_gal_y * coords[:, 1] + 
                   v_gal_z * coords[:, 2]) / dist
        
        return v_radial
