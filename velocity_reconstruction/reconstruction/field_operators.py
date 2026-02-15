"""Field operators for reconstruction algorithms."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple


def compute_gradient(
    field: NDArray[np.float64],
    dx: float = 1.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute 3D gradient using central differences.
    
    Returns df/dx, df/dy, df/dz.
    """
    nx, ny, nz = field.shape
    
    # Initialize output arrays
    dfdx = np.zeros_like(field)
    dfdy = np.zeros_like(field)
    dfdz = np.zeros_like(field)
    
    # Interior points (central differences)
    dfdx[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2 * dx)
    dfdy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2 * dx)
    dfdz[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2 * dx)
    
    # Boundary points (forward/backward differences)
    dfdx[0, :, :] = (field[1, :, :] - field[0, :, :]) / dx
    dfdx[-1, :, :] = (field[-1, :, :] - field[-2, :, :]) / dx
    dfdy[:, 0, :] = (field[:, 1, :] - field[:, 0, :]) / dx
    dfdy[:, -1, :] = (field[:, -1, :] - field[:, -2, :]) / dx
    dfdz[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / dx
    dfdz[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / dx
    
    return dfdx, dfdy, dfdz


def compute_divergence(
    vx: NDArray[np.float64],
    vy: NDArray[np.float64],
    vz: NDArray[np.float64],
    dx: float = 1.0,
) -> NDArray[np.float64]:
    """Compute 3D divergence using central differences.
    
    Returns div(v) = dvx/dx + dvy/dy + dvz/dz.
    """
    nx, ny, nz = vx.shape
    
    div = np.zeros_like(vx)
    
    # Interior points
    div[1:-1, 1:-1, 1:-1] = (
        (vx[2:, 1:-1, 1:-1] - vx[:-2, 1:-1, 1:-1]) +
        (vy[1:-1, 2:, 1:-1] - vy[1:-1, :-2, 1:-1]) +
        (vz[1:-1, 1:-1, 2:] - vz[1:-1, 1:-1, :-2])
    ) / (2 * dx)
    
    return div


def compute_curl(
    vx: NDArray[np.float64],
    vy: NDArray[np.float64],
    vz: NDArray[np.float64],
    dx: float = 1.0,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Compute 3D curl using central differences.
    
    Returns (curl_x, curl_y, curl_z).
    """
    nx, ny, nz = vx.shape
    
    curl_x = np.zeros_like(vx)
    curl_y = np.zeros_like(vx)
    curl_z = np.zeros_like(vx)
    
    # Interior points
    curl_x[1:-1, 1:-1, 1:-1] = (
        (vy[1:-1, 1:-1, 2:] - vy[1:-1, 1:-1, :-2]) -
        (vz[1:-1, 2:, 1:-1] - vz[1:-1, :-2, 1:-1])
    ) / (2 * dx)
    
    curl_y[1:-1, 1:-1, 1:-1] = (
        (vz[2:, 1:-1, 1:-1] - vz[:-2, 1:-1, 1:-1]) -
        (vx[1:-1, 1:-1, 2:] - vx[1:-1, 1:-1, :-2])
    ) / (2 * dx)
    
    curl_z[1:-1, 1:-1, 1:-1] = (
        (vx[1:-1, 2:, 1:-1] - vx[1:-1, :-2, 1:-1]) -
        (vy[2:, 1:-1, 1:-1] - vy[:-2, 1:-1, 1:-1])
    ) / (2 * dx)
    
    return curl_x, curl_y, curl_z


def compute_laplacian(
    field: NDArray[np.float64],
    dx: float = 1.0,
) -> NDArray[np.float64]:
    """Compute 3D Laplacian using central differences.
    
    Returns nabla^2 f = d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2.
    """
    nx, ny, nz = field.shape
    lap = np.zeros_like(field)
    
    # Interior points
    lap[1:-1, 1:-1, 1:-1] = (
        field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] +
        field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] +
        field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] -
        6 * field[1:-1, 1:-1, 1:-1]
    ) / (dx ** 2)
    
    return lap


def solve_poisson_fft(
    rho: NDArray[np.float64],
    dx: float = 1.0,
    G: float = 4.301e-6,
    rho_bar: float = 0.0,
) -> NDArray[np.float64]:
    """Solve Poisson equation using FFT.
    
    nabla^2 Phi = 4*pi*G*(rho - rho_bar)
    
    Parameters
    ----------
    rho : ndarray
        Density field.
    dx : float
        Grid spacing.
    G : float
        Gravitational constant.
    rho_bar : float
        Mean density.
        
    Returns
    -------
    ndarray
        Gravitational potential.
    """
    nx, ny, nz = rho.shape
    
    # Create k-space grid
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dx) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=dx) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = KX**2 + KY**2 + KZ**2
    k_squared[0, 0, 0] = 1.0  # Avoid division by zero
    
    # Transform to k-space
    rho_k = np.fft.fftn(rho - rho_bar)
    
    # Solve in k-space: Phi_k = -4*pi*G*rho_k / k^2
    phi_k = -4 * np.pi * G * rho_k / k_squared
    phi_k[0, 0, 0] = 0.0  # Set zero mode
    
    # Transform back
    phi = np.real(np.fft.ifftn(phi_k))
    
    return phi


def gaussian_smooth(
    field: NDArray[np.float64],
    sigma: float,
    dx: float = 1.0,
) -> NDArray[np.float64]:
    """Apply 3D Gaussian smoothing.
    
    Parameters
    ----------
    field : ndarray
        Input field.
    sigma : float
        Smoothing scale in grid units.
    dx : float
        Grid spacing.
        
    Returns
    -------
    ndarray
        Smoothed field.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(field, sigma=sigma / dx)


def assign_to_grid(
    positions: NDArray[np.float64],
    values: NDArray[np.float64],
    grid_shape: Tuple[int, int, int],
    extent: float,
    smoothing_sigma: Optional[float] = None,
) -> NDArray[np.float64]:
    """Assign point data to 3D grid.
    
    Parameters
    ----------
    positions : ndarray
        Position coordinates (N, 3).
    values : ndarray
        Values to assign (N,).
    grid_shape : tuple
        Shape of output grid (nx, ny, nz).
    extent : float
        Half-extent of grid.
    smoothing_sigma : float, optional
        Gaussian smoothing scale.
        
    Returns
    -------
    ndarray
        Field on grid.
    """
    nx, ny, nz = grid_shape
    
    # Filter out NaN positions and values
    valid_mask = np.isfinite(positions).all(axis=1) & np.isfinite(values)
    positions = positions[valid_mask]
    values = values[valid_mask]
    
    if len(values) == 0:
        return np.zeros(grid_shape)
    
    # Create coordinate arrays
    x = np.linspace(-extent, extent, nx)
    y = np.linspace(-extent, extent, ny)
    z = np.linspace(-extent, extent, nz)
    
    # Create 3D grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = np.zeros(grid_shape)
    count = np.zeros(grid_shape)
    
    # Assign each point to nearest grid cell - with robust error handling
    with np.errstate(invalid='ignore', over='ignore'):
        ix_raw = ((positions[:, 0] + extent) / (2 * extent) * (nx - 1))
        iy_raw = ((positions[:, 1] + extent) / (2 * extent) * (ny - 1))
        iz_raw = ((positions[:, 2] + extent) / (2 * extent) * (nz - 1))
        
    # Replace NaN/inf with valid indices
    ix_raw = np.nan_to_num(ix_raw, nan=nx//2, posinf=nx-1, neginf=0)
    iy_raw = np.nan_to_num(iy_raw, nan=ny//2, posinf=ny-1, neginf=0)
    iz_raw = np.nan_to_num(iz_raw, nan=nz//2, posinf=nz-1, neginf=0)
    
    ix = ix_raw.astype(int)
    iy = iy_raw.astype(int)
    iz = iz_raw.astype(int)
    
    # Clip to valid range
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)
    
    # Accumulate values
    for i in range(len(positions)):
        grid[ix[i], iy[i], iz[i]] += values[i]
        count[ix[i], iy[i], iz[i]] += 1
    
    # Average where there are points
    mask = count > 0
    grid[mask] = grid[mask] / count[mask]
    
    # Apply smoothing if requested
    if smoothing_sigma is not None:
        grid = gaussian_smooth(grid, smoothing_sigma, dx=2*extent/nx)
    
    return grid