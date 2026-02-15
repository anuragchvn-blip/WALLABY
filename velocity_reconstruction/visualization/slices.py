"""
Enhanced Visualization module for velocity field reconstruction.
Produces publication-quality figures matching prop.md specifications.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

def plot_density_slice(
    density: np.ndarray, 
    slice_index: Optional[int] = None,
    ax=None, 
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None, 
    title: str = 'Density Contrast $\\delta$',
    colorbar: bool = True,
    extent: float = 150.0,
    show_colorbar: bool = True,
    plane: str = 'XY'
) -> Optional['matplotlib.axes.Axes']:
    """Plot 2D slice of density field with proper coordinate labels."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
    except ImportError:
        return None
    
    # Get the data slice based on plane
    if plane == 'XY':
        data = density[:, :, slice_index] if slice_index is not None else density[:, :, density.shape[2]//2]
        xlabel = 'X (Mpc)'
        ylabel = 'Y (Mpc)'
    elif plane == 'XZ':
        data = density[:, slice_index if slice_index is not None else density.shape[1]//2, :] if slice_index is not None else density[:, density.shape[1]//2, :]
        xlabel = 'X (Mpc)'
        ylabel = 'Z (Mpc)'
    elif plane == 'YZ':
        data = density[slice_index if slice_index is not None else density.shape[0]//2, :, :] if slice_index is not None else density[density.shape[0]//2, :, :]
        xlabel = 'Y (Mpc)'
        ylabel = 'Z (Mpc)'
    else:
        data = density[:, :, slice_index] if slice_index is not None else density[:, :, density.shape[2]//2]
        xlabel = 'X (Mpc)'
        ylabel = 'Y (Mpc)'
    
    # Calculate extent in Mpc
    nx, ny = data.shape
    x_extent = np.linspace(-extent, extent, nx)
    y_extent = np.linspace(-extent, extent, ny)
    
    # Auto-set vmin/vmax if not provided
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    
    # Use symmetric log scale for density contrast
    if vmin < 0 and vmax > 0:
        # Don't pass vmin/vmax to imshow when using norm
        if vmin < 0:
            norm = SymLogNorm(linthresh=min(0.1, abs(vmax)*0.1), vmin=vmin, vmax=vmax)
            vmin_plot = None
            vmax_plot = None
        else:
            norm = None
            vmin_plot = vmin
            vmax_plot = vmax
    else:
        norm = None
        vmin_plot = vmin
        vmax_plot = vmax
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        data, 
        cmap=cmap, 
        vmin=vmin_plot, 
        vmax=vmax_plot,
        norm=norm,
        origin='lower',
        extent=[x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]],
        aspect='auto'
    )
    
    if show_colorbar and colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(r'$\delta = \rho/\bar{\rho} - 1$', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return ax


def plot_velocity_vectors(
    velocity: Dict[str, np.ndarray], 
    slice_index: int,
    ax=None,
    skip: int = 4,
    scale: float = 50.0,
    color: str = 'black',
    alpha: float = 0.6,
    extent: float = 150.0,
    plane: str = 'XY'
):
    """Plot velocity vectors on 2D slice with proper scaling."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ax
    
    vx = velocity.get("vx")
    vy = velocity.get("vy")
    
    if vx is None or vy is None:
        return ax
    
    # Get the slice based on plane
    if plane == 'XY':
        vx_slice = vx[:, :, slice_index]
        vy_slice = vy[:, :, slice_index]
    elif plane == 'XZ':
        vx_slice = vx[:, slice_index, :]
        vy_slice = vy[:, slice_index, :]
    elif plane == 'YZ':
        vx_slice = vx[slice_index, :, :]
        vy_slice = vy[slice_index, :, :]
    else:
        vx_slice = vx[:, :, slice_index]
        vy_slice = vy[:, :, slice_index]
    
    ny, nx = vx_slice.shape
    x = np.linspace(-extent, extent, nx)
    y = np.linspace(-extent, extent, ny)
    X, Y = np.meshgrid(x, y)
    
    # Downsample for visualization
    vx_ds = vx_slice[::skip, ::skip]
    vy_ds = vy_slice[::skip, ::skip]
    X_ds = X[::skip, ::skip]
    Y_ds = Y[::skip, ::skip]
    
    # Calculate magnitude for coloring
    magnitude = np.sqrt(vx_ds**2 + vy_ds**2)
    
    # Plot quiver
    quiv = ax.quiver(
        X_ds, Y_ds, 
        vx_ds, vy_ds,
        magnitude,
        scale=scale,
        scale_units='xy',
        cmap='viridis',
        alpha=alpha,
        width=0.003
    )
    
    return ax, quiv


def create_summary_figure(
    density: np.ndarray, 
    velocity: Dict[str, np.ndarray],
    output_path: str,
    extent: float = 150.0,
    box_size: float = 300.0
):
    """Create comprehensive summary figure with multiple panels."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 11
    except ImportError:
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Get center slices
    ix, iy, iz = density.shape[0]//2, density.shape[1]//2, density.shape[2]//2
    
    # Auto-compute vmax - use symmetric range
    d_min = np.nanmin(density)
    d_max = np.nanmax(density)
    vmax = max(abs(d_min), abs(d_max))
    vmin = -vmax
    
    # Panel 1: XY plane (Supergalactic plane)
    ax1 = fig.add_subplot(gs[0, 0])
    # Simple linear scale for compatibility
    im1 = ax1.imshow(
        density[:, :, iz],
        cmap='RdBu_r',
        vmin=vmin, vmax=vmax,
        origin='lower',
        extent=[-extent, extent, -extent, extent]
    )
    plt.colorbar(im1, ax=ax1, shrink=0.8, label=r'$\delta$')
    ax1.set_title('Supergalactic Plane (XY)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (Mpc)')
    ax1.set_ylabel('Y (Mpc)')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: XZ plane
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        density[:, iy, :],
        cmap='RdBu_r',
        vmin=vmin, vmax=vmax,
        origin='lower',
        extent=[-extent, extent, -extent, extent]
    )
    plt.colorbar(im2, ax=ax2, shrink=0.8, label=r'$\delta$')
    ax2.set_title('XZ Plane', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (Mpc)')
    ax2.set_ylabel('Z (Mpc)')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: YZ plane
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(
        density[ix, :, :],
        cmap='RdBu_r',
        vmin=vmin, vmax=vmax,
        origin='lower',
        extent=[-extent, extent, -extent, extent]
    )
    plt.colorbar(im3, ax=ax3, shrink=0.8, label=r'$\delta$')
    ax3.set_title('YZ Plane', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Y (Mpc)')
    ax3.set_ylabel('Z (Mpc)')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Histogram of density values
    ax4 = fig.add_subplot(gs[1, 0])
    flat_density = density.flatten()
    flat_density = flat_density[np.isfinite(flat_density)]
    ax4.hist(flat_density, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label=r'$\delta = 0$')
    ax4.set_xlabel(r'$\delta$', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Density Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Power spectrum
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        from velocity_reconstruction.validation.metrics import compute_power_spectrum
        k, pk = compute_power_spectrum(density, box_size=box_size)
        # Remove zero/negative k values
        valid = k > 0
        k = k[valid]
        pk = pk[valid]
        ax5.loglog(k, pk, 'b-', linewidth=1.5, alpha=0.8)
        ax5.set_xlabel('k (h/Mpc)', fontsize=12)
        ax5.set_ylabel('P(k)', fontsize=12)
        ax5.set_title('Power Spectrum', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, which='both')
    except Exception as e:
        ax5.text(0.5, 0.5, 'Power spectrum\nunavailable', 
                transform=ax5.transAxes, ha='center', va='center')
    
    # Panel 6: Statistics text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate statistics
    vx_max = np.nanmax(np.abs(velocity['vx'])) if 'vx' in velocity else 0
    
    stats_text = f"""
    Reconstruction Statistics
    {'='*35}
    
    Grid Size:     {density.shape[0]}³
    Box Size:      {box_size:.0f} Mpc
    Resolution:    {box_size/density.shape[0]:.1f} Mpc/cell
    
    Density Field:
      Min δ:      {np.nanmin(density):.4f}
      Max δ:      {np.nanmax(density):.4f}
      Mean δ:      {np.nanmean(density):.6f}
      Std δ:       {np.nanstd(density):.4f}
    
    Velocity Field:
      |v|_max:    {vx_max:.1f} km/s
    
    WALLABY Pipeline
    CosmicFlows-4
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle('WALLABY Velocity Field Reconstruction\nCosmicFlows-4 Catalog', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def create_3d_volume_visualization(
    density: np.ndarray,
    output_path: str,
    extent: float = 150.0,
    isosurfaces: Tuple[float, ...] = (-0.5, -0.2, 0.2, 0.5, 1.0)
):
    """Create 3D isosurface visualization (requires mayavi or plotly)."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get center slice for simple visualization
        ix, iy, iz = density.shape[0]//2, density.shape[1]//2, density.shape[2]//2
        
        # Simple 3D contour (simplified for matplotlib)
        x = np.linspace(-extent, extent, density.shape[0])
        y = np.linspace(-extent, extent, density.shape[1])
        z = np.linspace(-extent, extent, density.shape[2])
        
        # Just show a simple projection
        ax.contourf(density[:, :, iz], levels=20, offset=extent, cmap='RdBu_r')
        
        ax.set_xlabel('X (Mpc)')
        ax.set_ylabel('Y (Mpc)')
        ax.set_zlabel('Z (Mpc)')
        ax.set_title('3D Density Field Projection')
        
        plt.savefig(output_path, dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"3D visualization skipped: {e}")


def plot_supergalactic_plane(
    density: np.ndarray,
    velocity: Dict[str, np.ndarray],
    output_path: str,
    extent: float = 150.0
):
    """Create publication-quality supergalactic plane plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
    except ImportError:
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get center slice
    iz = density.shape[2] // 2
    data = density[:, :, iz]
    
    # Calculate coordinates
    nx, ny = data.shape
    x = np.linspace(-extent, extent, nx)
    y = np.linspace(-extent, extent, ny)
    
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    norm = SymLogNorm(linthresh=0.1, vmin=-vmax, vmax=vmax)
    
    # Plot density
    im = ax.imshow(
        data,
        cmap='RdBu_r',
        norm=norm,
        origin='lower',
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect='equal'
    )
    
    # Add velocity vectors
    vx = velocity['vx'][:, :, iz]
    vy = velocity['vy'][:, :, iz]
    
    skip = 8
    X, Y = np.meshgrid(x, y)
    
    # Scale velocity for visualization
    scale_factor = np.nanmax(np.sqrt(vx**2 + vy**2)) / 100.0
    
    ax.quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        vx[::skip, ::skip] / scale_factor,
        vy[::skip, ::skip] / scale_factor,
        np.sqrt(vx[::skip, ::skip]**2 + vy[::skip, ::skip]**2),
        cmap='viridis',
        scale=150,
        scale_units='xy',
        alpha=0.7,
        width=0.004
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r'$\delta = \rho/\bar{\rho} - 1$', fontsize=14)
    
    # Labels
    ax.set_xlabel('Supergalactic X (Mpc)', fontsize=14)
    ax.set_ylabel('Supergalactic Y (Mpc)', fontsize=14)
    ax.set_title('Supergalactic Plane Density with Velocity Field', fontsize=16, fontweight='bold')
    
    # Mark important structures
    # Virgo Cluster approx location
    ax.plot(0, 0, 'r*', markersize=15, label='Virgo (approx)')
    ax.legend(loc='upper right', fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
