"""
NS-VGR (Non-linear Saturated Velocity-Gravity Relation) Reconstructor.

Developed by Anurag Chavan.
Provides non-linear velocity field reconstruction with exponential saturation
and density-gradient entrainment.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional
from velocity_reconstruction.reconstruction.field_operators import (
    solve_poisson_fft, compute_gradient, gaussian_smooth, assign_to_grid
)

class NSVGRReconstructor:
    """NS-VGR velocity field reconstruction algorithm."""

    def __init__(self, extent: float = 100.0, resolution: int = 128,
                 smoothing_sigma: float = 5.0, H0: float = 70.0, Omega_m: float = 0.315,
                 delta_crit: float = 1.68, gamma: float = 0.4, l_nl: float = 5.0):
        self.extent = extent
        self.resolution = resolution
        self.smoothing_sigma = smoothing_sigma
        self.H0 = H0
        self.Omega_m = Omega_m
        self.f_growth = Omega_m ** 0.55
        self.delta_crit = delta_crit
        self.gamma = gamma
        self.l_nl = l_nl
        self.cell_size = 2 * extent / resolution
        
        self.density = None
        self.velocity = None

    def reconstruct(self, ra: NDArray, dec: NDArray, distance: NDArray,
                    v_pec: NDArray) -> Dict:
        """Run NS-VGR reconstruction.
        
        Args:
            ra: Right Ascension (deg)
            dec: Declination (deg)
            distance: Distance (Mpc)
            v_pec: Peculiar velocity (km/s)
            
        Returns:
            Dictionary containing density, velocity field, and potential.
        """
        # 1. Coordinate conversion to SG Cartesian
        coords = self._equatorial_to_supergalactic_cartesian(ra, dec, distance)
        
        # 2. Assign to grid (density field estimation)
        # First, we need an initial density field from the peculiar velocities
        v_radial_grid = assign_to_grid(coords, v_pec, (self.resolution,)*3, self.extent, self.smoothing_sigma)
        v_radial_smooth = gaussian_smooth(v_radial_grid, self.smoothing_sigma, self.cell_size)
        
        # Estimate density from radial velocity divergence (preliminary)
        # In a real pipeline, one might iterate, but here we use the smoothed field
        vx_init, vy_init, vz_init = self._radial_to_3d_simple(v_radial_smooth)
        div_v = self._compute_divergence(vx_init, vy_init, vz_init)
        density = -div_v / (self.H0 * self.f_growth)
        
        # 3. Apply NS-VGR Formula
        # Solve for gravity first
        potential = solve_poisson_fft(density, self.cell_size)
        gx, gy, gz = compute_gradient(potential, self.cell_size)
        
        # Saturation Kernel
        S = np.exp(-np.abs(density) / self.delta_crit)
        
        # Gradient Entrainment
        grad_dx, grad_dy, grad_dz = compute_gradient(density, self.cell_size)
        denom = 1.0 + np.maximum(density, -0.9)  # Avoid division by zero in voids
        
        # Final prediction: v = f * (S * g/H0 + gamma * l_nl * grad_delta / (1+delta)) * H0
        # Simplifies units: g/H0 has units km/s/Mpc * Mpc / (km/s) = 1? 
        # Actually g from Poisson typically has units (km/s)^2. 
        # Let's ensure dimensional consistency as per POTENT.
        
        # Linear velocity from gravity term: v_lin = (f/H0) * g
        vx_grav = (self.f_growth / self.H0) * gx * S
        vy_grav = (self.f_growth / self.H0) * gy * S
        vz_grav = (self.f_growth / self.H0) * gz * S
        
        # Entrainment term: v_ent = f * H0 * gamma * l_nl * grad(delta)/(1+delta)
        vx_ent = self.f_growth * self.H0 * self.gamma * self.l_nl * (grad_dx / denom)
        vy_ent = self.f_growth * self.H0 * self.gamma * self.l_nl * (grad_dy / denom)
        vz_ent = self.f_growth * self.H0 * self.gamma * self.l_nl * (grad_dz / denom)
        
        vx = vx_grav + vx_ent
        vy = vy_grav + vy_ent
        vz = vz_grav + vz_ent
        
        self.velocity = {"vx": vx, "vy": vy, "vz": vz}
        self.density = density
        
        return {
            "density": density,
            "vx": vx, "vy": vy, "vz": vz,
            "potential": potential
        }

    def _equatorial_to_supergalactic_cartesian(self, ra, dec, distance):
        # Constants from simple_velocity.py
        SGP_RA, SGP_DEC = 283.8, 15.7
        ra_rad, dec_rad = np.deg2rad(ra), np.deg2rad(dec)
        sgp_ra_rad, sgp_dec_rad = np.deg2rad(SGP_RA), np.deg2rad(SGP_DEC)
        
        sgl = np.arctan2(
            np.cos(dec_rad) * np.sin(ra_rad - sgp_ra_rad),
            np.cos(dec_rad) * np.cos(ra_rad - sgp_ra_rad) * np.cos(sgp_dec_rad) -
            np.sin(dec_rad) * np.sin(sgp_dec_rad)
        )
        sgb = np.arcsin(
            np.sin(dec_rad) * np.sin(sgp_dec_rad) +
            np.cos(dec_rad) * np.cos(sgp_dec_rad) * np.cos(ra_rad - sgp_ra_rad)
        )
        
        x = distance * np.cos(sgb) * np.cos(sgl)
        y = distance * np.cos(sgb) * np.sin(sgl)
        z = distance * np.sin(sgb)
        return np.column_stack([x, y, z])

    def _radial_to_3d_simple(self, v_radial_grid):
        nx = self.resolution
        grid_x = np.linspace(-self.extent, self.extent, nx)
        X, Y, Z = np.meshgrid(grid_x, grid_x, grid_x, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2)
        r = np.where(r > 0.1, r, 0.1)
        return v_radial_grid * X / r, v_radial_grid * Y / r, v_radial_grid * Z / r

    def _compute_divergence(self, vx, vy, vz):
        div = np.zeros_like(vx)
        dx = self.cell_size
        div[1:-1, 1:-1, 1:-1] = (
            (vx[2:, 1:-1, 1:-1] - vx[:-2, 1:-1, 1:-1]) +
            (vy[1:-1, 2:, 1:-1] - vy[1:-1, :-2, 1:-1]) +
            (vz[1:-1, 1:-1, 2:] - vz[1:-1, 1:-1, :-2])
        ) / (2 * dx)
        return div
