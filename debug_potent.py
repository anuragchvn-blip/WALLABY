"""Debug script to trace through POTENT."""
import numpy as np
from velocity_reconstruction.reconstruction.potent import POTENTReconstructor

potent = POTENTReconstructor(extent=250.0, resolution=32, smoothing_sigma=8.0, H0=70.0)

# Simple test data
ra = np.array([10.0, 20.0, 30.0])
dec = np.array([5.0, 10.0, 15.0])
distance = np.array([20.0, 30.0, 40.0])
v_pec = np.array([100.0, -200.0, 150.0])

print("Input:")
print(f"  RA: {ra}")
print(f"  Dec: {dec}")
print(f"  Distance: {distance}")
print(f"  v_pec: {v_pec}")

# Step 1: Convert coordinates
coords = potent._equatorial_to_supergalactic_cartesian(ra, dec, distance)
print(f"\nCoords (supergalactic):")
print(coords)

# Step 2: Assign to grid
v_radial_grid = potent._assign_radial_velocities(coords, v_pec)
print(f"\nRadial velocity grid: min={np.min(v_radial_grid):.1f}, max={np.max(v_radial_grid):.1f}")

# Step 3: Smooth
v_radial_smooth = potent._smoothing = gaussian_smooth(v_radial_grid, potent.smoothing_sigma, potent.cell_size)
print(f"Smoothed: min={np.min(v_radial_smooth):.1f}, max={np.max(v_radial_smooth):.1f}")

# Step 4: 3D velocity
velocity_3d = potent._radial_to_3d_velocity(v_radial_smooth, coords, v_pec)
print(f"\n3D velocity:")
print(f"  vx: min={np.min(velocity_3d['vx']):.1f}, max={np.max(velocity_3d['vx']):.1f}")
print(f"  vy: min={np.min(velocity_3d['vy']):.1f}, max={np.max(velocity_3d['vy']):.1f}")
print(f"  vz: min={np.min(velocity_3d['vz']):.1f}, max={np.max(velocity_3d['vz']):.1f}")

# Step 5: Density
density = potent._velocity_to_density(velocity_3d)
print(f"\nDensity: min={np.min(density):.4f}, max={np.max(density):.4f}")

# Step 6: Potential
from velocity_reconstruction.reconstruction.field_operators import solve_poisson_fft
potential = solve_poisson_fft(density, potent.cell_size)
print(f"\nPotential: min={np.min(potential):.4f}, max={np.max(potential):.4f}")

# Step 7: Final velocity
vel_final = potent._compute_velocity_from_potential(potential)
print(f"\nFinal velocity:")
print(f"  vx: min={np.min(vel_final['vx']):.1f}, max={np.max(vel_final['vx']):.1f}")
print(f"  |v|: max={np.max(np.sqrt(vel_final['vx']**2 + vel_final['vy']**2 + vel_final['vz']**2)):.1f}")
