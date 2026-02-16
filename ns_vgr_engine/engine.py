"""
NS-VGR (Non-linear Saturated Velocity-Gravity Relation) Engine.
Author: Anurag Chavan
Target: r >= 0.25-0.35 in WALLABY-VR pipeline.
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq

class NSVGRLite:
    """Consolidated Minimal NS-VGR Reconstructor."""
    
    def __init__(self, L=100.0, N=128, sigma=5.0, H0=70.0, Om=0.315):
        self.L, self.N, self.sigma = L, N, sigma
        self.H0, self.Om = H0, Om
        self.f = Om**0.55
        self.dx = 2 * L / N
        self.grid_shape = (N, N, N)

    def solve_gravity(self, delta):
        """Solve Poisson equation in Fourier Space: g = -grad(Phi)."""
        k = fftfreq(self.N, d=self.dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2
        k2[0,0,0] = 1.0 # Avoid div by zero
        
        # Phi = -delta / k^2 (in proper cosmological units)
        delta_k = fftn(delta)
        phi_k = -delta_k / k2
        
        # g_x = -i * kx * Phi
        gx = np.real(ifftn(1j * kx * phi_k))
        gy = np.real(ifftn(1j * ky * phi_k))
        gz = np.real(ifftn(1j * kz * phi_k))
        return gx, gy, gz

    def compute_grad_delta(self, delta):
        """Compute grad(delta) using central differences."""
        grad = np.gradient(delta, self.dx)
        return grad[0], grad[1], grad[2]

    def reconstruct(self, coords, v_pec, delta_crit=1.68, gamma=0.4, L_nl=5.0):
        """Primary NS-VGR Pipeline."""
        # 1. Grid Assignment with proper weighting (CIC-like)
        indices = ((coords + self.L) / self.dx).astype(int)
        valid = np.all((indices >= 0) & (indices < self.N), axis=1)
        indices = indices[valid]
        v_pec_valid = v_pec[valid]
        
        grid_v = np.zeros(self.grid_shape)
        grid_count = np.zeros(self.grid_shape)
        
        for i in range(len(indices)):
            ix, iy, iz = indices[i]
            grid_v[ix, iy, iz] += v_pec_valid[i]
            grid_count[ix, iy, iz] += 1
        
        # Average values in cells
        mask = grid_count > 0
        grid_v[mask] /= grid_count[mask]
        
        # 2. Expand radial velocity to 3D field
        grid_x = np.linspace(-self.L, self.L, self.N)
        X, Y, Z = np.meshgrid(grid_x, grid_x, grid_x, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2) + 0.1
        
        # Expand radial velocities to 3D (v_vec = v_rad * r_hat)
        vx_init = grid_v * X / R
        vy_init = grid_v * Y / R
        vz_init = grid_v * Z / R
        
        # 3. Smooth velocity field
        k = fftfreq(self.N, d=self.dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k2_mag = kx**2 + ky**2 + kz**2
        kernel = np.exp(-0.5 * k2_mag * self.sigma**2)
        
        vx_smooth = np.real(ifftn(fftn(vx_init) * kernel))
        vy_smooth = np.real(ifftn(fftn(vy_init) * kernel))
        vz_smooth = np.real(ifftn(fftn(vz_init) * kernel))
        
        # 4. Compute divergence -> Density
        vx_k = fftn(vx_smooth)
        vy_k = fftn(vy_smooth)
        vz_k = fftn(vz_smooth)
        div_v = np.real(ifftn(1j * (kx * vx_k + ky * vy_k + kz * vz_k)))
        delta = -div_v / (self.H0 * self.f + 1e-10)
        
        # 5. Apply NS-VGR Equation
        gx, gy, gz = self.solve_gravity(delta)
        gdx, gdy, gdz = self.compute_grad_delta(delta)
        
        # Non-linear components
        S = np.exp(-np.abs(delta) / delta_crit)
        denom = 1.0 + np.maximum(delta, -0.9)
        
        # V = f * (S * g + H0 * gamma * L_nl * grad_delta / (1+delta))
        vx = self.f * (S * gx + self.H0 * gamma * L_nl * gdx / denom)
        vy = self.f * (S * gy + self.H0 * gamma * L_nl * gdy / denom)
        vz = self.f * (S * gz + self.H0 * gamma * L_nl * gdz / denom)
        
        return vx, vy, vz, delta

    @staticmethod
    def malmquist_fix(dist, err, delta_local=0.0):
        """Minimal density-dependent Malmquist correction."""
        sigma_rel = err / np.maximum(dist, 1.0)
        # d_corr = d_obs / (1 + 0.5 * sigma^2 * (1.68 + delta))
        return dist / (1.0 + 0.5 * sigma_rel**2 * (1.68 + delta_local))
