"""Simple velocity field reconstruction - direct radial to 3D."""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional
from velocity_reconstruction.reconstruction.field_operators import gaussian_smooth, assign_to_grid

# Supergalactic pole
SGP_RA = 283.8
SGP_DEC = 15.7


def equatorial_to_supergalactic(ra_deg, dec_deg):
    """Convert equatorial to supergalactic."""
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    sgp_ra_rad = np.deg2rad(SGP_RA)
    sgp_dec_rad = np.deg2rad(SGP_DEC)
    
    sgl = np.arctan2(
        np.cos(dec_rad) * np.sin(ra_rad - sgp_ra_rad),
        np.cos(dec_rad) * np.cos(ra_rad - sgp_ra_rad) * np.cos(sgp_dec_rad) -
        np.sin(dec_rad) * np.sin(sgp_dec_rad)
    )
    sgb = np.arcsin(
        np.sin(dec_rad) * np.sin(sgp_dec_rad) +
        np.cos(dec_rad) * np.cos(sgp_dec_rad) * np.cos(ra_rad - sgp_ra_rad)
    )
    return np.rad2deg(sgl), np.rad2deg(sgb)


def reconstruct_velocity_field(ra, dec, distance, v_pec, 
                               extent=250.0, resolution=64, smoothing_sigma=8.0,
                               H0=70.0, Omega_m=0.3):
    """Reconstruct 3D velocity field from radial peculiar velocities.
    
    Simple method: smooth the radial velocities and convert to 3D assuming
    flow is radial from density enhancements.
    """
    f_growth = Omega_m ** 0.55
    cell_size = 2 * extent / resolution
    
    # Convert to supergalactic coordinates
    sgl, sgb = equatorial_to_supergalactic(ra, dec)
    sgl_rad = np.deg2rad(sgl)
    sgb_rad = np.deg2rad(sgb)
    
    # 3D positions
    x = distance * np.cos(sgb_rad) * np.cos(sgl_rad)
    y = distance * np.cos(sgb_rad) * np.sin(sgl_rad)
    z = distance * np.sin(sgb_rad)
    coords = np.column_stack([x, y, z])
    
    # Assign radial velocities to grid
    grid_shape = (resolution, resolution, resolution)
    v_radial_grid = assign_to_grid(coords, v_pec, grid_shape, extent, smoothing_sigma)
    
    # Smooth
    v_radial_smooth = gaussian_smooth(v_radial_grid, smoothing_sigma, cell_size)
    
    # Convert to 3D: assume velocity is radial from each grid point
    nx = resolution
    grid_x = np.linspace(-extent, extent, nx)
    X, Y, Z = np.meshgrid(grid_x, grid_x, grid_x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r = np.where(r > cell_size, r, cell_size)  # Avoid division by zero
    
    # v_3D = v_radial * (position / r)
    # Keep small scale - the v_radial values already have proper magnitude
    SCALE = 1.0
    vx = SCALE * v_radial_smooth * X / r
    vy = SCALE * v_radial_smooth * Y / r
    vz = SCALE * v_radial_smooth * Z / r
    
    # Compute density from velocity divergence
    div_v = compute_divergence(vx, vy, vz, cell_size)
    density = -div_v / (H0 * f_growth)
    
    return {
        "vx": vx, "vy": vy, "vz": vz,
        "density": density,
        "v_radial": v_radial_smooth
    }


def compute_divergence(vx, vy, vz, dx):
    """Compute divergence."""
    nx, ny, nz = vx.shape
    div = np.zeros_like(vx)
    div[1:-1, 1:-1, 1:-1] = (
        (vx[2:, 1:-1, 1:-1] - vx[:-2, 1:-1, 1:-1]) +
        (vy[1:-1, 2:, 1:-1] - vy[1:-1, :-2, 1:-1]) +
        (vz[1:-1, 1:-1, 2:] - vz[1:-1, 1:-1, :-2])
    ) / (2 * dx)
    return div
