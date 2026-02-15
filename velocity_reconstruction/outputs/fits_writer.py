"""Output products module - FITS writer and derived products."""

import numpy as np
from typing import Dict, Optional
from pathlib import Path


def write_density_field(filename: str, density: np.ndarray, header: Optional[Dict] = None):
    """Write 3D density field to FITS file."""
    try:
        from astropy.io import fits
        hdu = fits.PrimaryHDU(density)
        if header:
            for key, value in header.items():
                hdu.header[key] = value
        hdu.writeto(filename, overwrite=True)
    except ImportError:
        np.save(filename.replace('.fits', '.npy'), density)


def write_velocity_field(filename: str, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray):
    """Write 3D velocity field components to FITS."""
    try:
        from astropy.io import fits
        hdu = fits.PrimaryHDU(np.stack([vx, vy, vz]))
        hdu.writeto(filename, overwrite=True)
    except ImportError:
        np.savez(filename.replace('.fits', '.npz'), vx=vx, vy=vy, vz=vz)


def write_potential(filename: str, potential: np.ndarray):
    """Write gravitational potential to FITS."""
    try:
        from astropy.io import fits
        hdu = fits.PrimaryHDU(potential)
        hdu.header['UNIT'] = '(km/s)^2'
        hdu.writeto(filename, overwrite=True)
    except ImportError:
        np.save(filename.replace('.fits', '.npy'), potential)


def compute_bulk_flow(velocity: Dict, r_max: Optional[float] = None) -> Dict:
    """Compute bulk flow profile."""
    vx = velocity.get("vx", np.array([]))
    vy = velocity.get("vy", np.array([]))
    vz = velocity.get("vz", np.array([]))
    
    if vx.size == 0:
        return {"B": 0.0, "B_x": 0.0, "B_y": 0.0, "B_z": 0.0}
    
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    return {"B": float(np.mean(v_mag)), "B_x": float(np.mean(vx)), "B_y": float(np.mean(vy)), "B_z": float(np.mean(vz))}


def compute_divergence_map(velocity: Dict, dx: float = 1.0) -> np.ndarray:
    """Compute velocity divergence map."""
    from velocity_reconstruction.reconstruction.field_operators import compute_divergence
    vx = velocity.get("vx")
    vy = velocity.get("vy")
    vz = velocity.get("vz")
    if vx is None:
        return np.array([])
    return compute_divergence(vx, vy, vz, dx)
