"""
Peculiar velocity calculation from distance indicators.

This module implements the computation of peculiar velocities from
distance measurements using various distance indicators (Tully-Fisher,
Fundamental Plane, Type Ia Supernovae, etc.).

The peculiar velocity is computed as:
    v_pec = cz_obs - H0 * D

where:
- cz_obs is the observed cosmological redshift (in km/s)
- H0 is the Hubble constant
- D is the distance (in Mpc)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, Tuple, Union

# Physical constants
H0_DEFAULT = 70.0  # km/s/Mpc (default Hubble constant)
C = 299792.458  # km/s (speed of light)

# Distance indicator intrinsic scatter (in quadrature with distance error)
TF_INTRINSIC_SCATTER = 0.35  # mag (Tully-Fisher)
FP_INTRINSIC_SCATTER = 0.1  # dex (Fundamental Plane)
SNE_INTRINSIC_SCATTER = 0.15  # mag (Type Ia Supernovae)
TRGB_INTRINSIC_SCATTER = 0.15  # mag (TRGB)
CEPHEIDS_INTRINSIC_SCATTER = 0.15  # mag (Cepheids)
SBF_INTRINSIC_SCATTER = 0.15  # mag (SBF)
EPM_INTRINSIC_SCATTER = 0.20  # mag (EPM)
GCLF_INTRINSIC_SCATTER = 0.20  # mag (GCLF)


class DistanceMethod:
    """Enumeration of distance indicator methods."""

    TF = "TF"           # Tully-Fisher
    FP = "FP"           # Fundamental Plane
    SNE_IA = "SNe Ia"   # Type Ia Supernovae
    TRGB = "TRGB"       # Tip of the Red Giant Branch
    CEPHEIDS = "Cepheids"
    SBF = "SBF"         # Surface Brightness Fluctuations
    EPM = "EPM"         # Expanding Photosphere Method
    GCLF = "GCLF"       # Globular Cluster Luminosity Function
    UNKNOWN = "Unknown"


def get_intrinsic_scatter(method: str) -> float:
    """
    Get intrinsic scatter for a distance method.

    Parameters
    ----------
    method : str
        Distance method name.

    Returns
    -------
    float
        Intrinsic scatter in magnitude or dex.
    """
    scatter_map = {
        DistanceMethod.TF: TF_INTRINSIC_SCATTER,
        DistanceMethod.FP: FP_INTRINSIC_SCATTER,
        DistanceMethod.SNE_IA: SNE_INTRINSIC_SCATTER,
        DistanceMethod.TRGB: TRGB_INTRINSIC_SCATTER,
        DistanceMethod.CEPHEIDS: CEPHEIDS_INTRINSIC_SCATTER,
        DistanceMethod.SBF: SBF_INTRINSIC_SCATTER,
        DistanceMethod.EPM: EPM_INTRINSIC_SCATTER,
        DistanceMethod.GCLF: GCLF_INTRINSIC_SCATTER,
    }
    return scatter_map.get(method.upper(), 0.0)


def compute_distance_from_modulus(
    distance_modulus: float,
    zero_point: float = 0.0,
) -> float:
    """
    Compute distance from distance modulus.

    D = 10^((mu - M0)/5) / 1e-5  # in Mpc

    where mu is the observed distance modulus and M0 is the zero-point.

    Parameters
    ----------
    distance_modulus : float
        Observed distance modulus (m - M).
    zero_point : float
        Zero-point offset.

    Returns
    -------
    float
        Distance in Mpc.
    """
    return 10.0 ** ((distance_modulus - zero_point) / 5.0) / 1e-5


def compute_distance_modulus(
    distance: float,
    zero_point: float = 0.0,
) -> float:
    """
    Compute distance modulus from distance.

    mu = M0 + 5*log10(D) + 25

    Parameters
    ----------
    distance : float
        Distance in Mpc.
    zero_point : float
        Zero-point offset.

    Returns
    -------
    float
        Distance modulus.
    """
    return zero_point + 5.0 * np.log10(distance) + 25.0


def compute_peculiar_velocity(
    cz_obs: NDArray[np.float64],
    distance: NDArray[np.float64],
    H0: float = H0_DEFAULT,
) -> NDArray[np.float64]:
    """
    Compute peculiar velocities from observed redshift and distance.

    v_pec = cz_obs - H0 * D

    Parameters
    ----------
    cz_obs : ndarray
        Observed redshift (cz) in km/s.
    distance : ndarray
        Distance in Mpc.
    H0 : float
        Hubble constant in km/s/Mpc.

    Returns
    -------
    ndarray
        Peculiar velocity in km/s.
    """
    return cz_obs - H0 * distance


def compute_peculiar_velocity_error(
    cz_error: NDArray[np.float64],
    distance_error: NDArray[np.float64],
    distance: NDArray[np.float64],
    intrinsic_scatter: float,
    H0: float = H0_DEFAULT,
) -> NDArray[np.float64]:
    """
    Compute error on peculiar velocity.

    Propagates errors from redshift, distance, and intrinsic scatter.

    sigma_v^2 = sigma_cz^2 + (H0 * sigma_D)^2 + (d v_pec/d mu * sigma_mu)^2

    where sigma_mu is the combined distance modulus error.

    Parameters
    ----------
    cz_error : ndarray
        Error on observed redshift in km/s.
    distance_error : ndarray
        Error on distance in Mpc.
    distance : ndarray
        Distance in Mpc.
    intrinsic_scatter : float
        Intrinsic scatter in distance indicator (mag or dex).
    H0 : float
        Hubble constant in km/s/Mpc.

    Returns
    -------
    ndarray
        Error on peculiar velocity in km/s.
    """
    # Error from redshift measurement
    sigma_v_cz = cz_error

    # Error from distance measurement
    sigma_v_dist = H0 * distance_error

    # Error from intrinsic scatter
    # d(mu)/d(D) = 5 / (D * ln(10))
    # sigma_v_intrinsic = |d v_pec/d D| * sigma_D
    #                  = H0 * sigma_D_intrinsic
    # Avoid division by zero
    safe_distance = np.where(distance > 0.01, distance, 0.01)
    sigma_D_intrinsic = safe_distance * intrinsic_scatter / (safe_distance * np.log(10))
    sigma_v_intrinsic = H0 * np.abs(sigma_D_intrinsic)

    # Combine in quadrature (handle NaN/inf)
    sigma_v = np.sqrt(sigma_v_cz**2 + sigma_v_dist**2 + sigma_v_intrinsic**2)
    sigma_v = np.clip(sigma_v, 0, 10000)  # Cap at reasonable value

    return sigma_v


def compute_distance_from_tf(
    magnitude: float,
    line_width: float,
    a: float = -11.0,  # TF intercept
    b: float = 10.0,   # TF slope
) -> Tuple[float, float]:
    """
    Compute distance from Tully-Fisher relation.

    M = a + b * log10(W)

    where M is the absolute magnitude and W is the line width.

    Parameters
    ----------
    magnitude : float
        Apparent magnitude.
    line_width : float
        HI line width in km/s.
    a : float
        TF intercept.
    b : float
        TF slope.

    Returns
    -------
    tuple of (float, float)
        Distance in Mpc and distance modulus error.
    """
    # Absolute magnitude from TF relation
    log_W = np.log10(line_width)
    M = a + b * log_W

    # Distance modulus: mu = m - M
    mu = magnitude - M

    # Distance
    D = compute_distance_from_modulus(mu)

    return D, mu


def compute_distance_from_fp(
    sigma: float,       # velocity dispersion
    I_e: float,         # surface brightness
    R_e: float,         # effective radius
    a: float = -8.5,    # FP intercept
    b: float = 4.0,     # FP slope
    c: float = -0.75,   # FP coefficient for surface brightness
) -> Tuple[float, float]:
    """
    Compute distance from Fundamental Plane relation.

    log10(R_e) = a + b * log10(sigma) + c * log10(I_e)

    Parameters
    ----------
    sigma : float
        Velocity dispersion in km/s.
    I_e : float
        Surface brightness at R_e.
    R_e : float
        Effective radius in kpc.
    a, b, c : float
        FP coefficients.

    Returns
    -------
    tuple of (float, float)
        Distance in Mpc and distance modulus error.
    """
    log_sigma = np.log10(sigma)
    log_I = np.log10(I_e)

    # FP relation
    log_R_e = a + b * log_sigma + c * log_I

    # Physical size
    R_e_phys = 10**log_R_e  # in kpc

    # Angular size (from effective radius in arcsec)
    # D = R_phys / (theta * 1e-3) for theta in arcsec
    # For now, return physical size - need angular size for distance

    # This is a simplified version - full implementation needs angular sizes
    return R_e_phys / 1000.0, 0.0  # placeholder


def process_galaxy_catalog(
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    cz: NDArray[np.float64],
    distance: NDArray[np.float64],
    distance_error: NDArray[np.float64],
    cz_error: Optional[NDArray[np.float64]] = None,
    method: Union[str, NDArray[np.str_]] = DistanceMethod.TF,
    H0: float = H0_DEFAULT,
) -> Dict[str, NDArray]:
    """
    Process a galaxy catalog to compute peculiar velocities.

    Parameters
    ----------
    ra : ndarray
        Right Ascension in degrees.
    dec : ndarray
        Declination in degrees.
    cz : ndarray
        Observed redshift in km/s.
    distance : ndarray
        Distance in Mpc.
    distance_error : ndarray
        Error on distance in Mpc.
    cz_error : ndarray, optional
        Error on redshift in km/s.
    method : str or ndarray
        Distance method(s).
    H0 : float
        Hubble constant.

    Returns
    -------
    dict
        Dictionary containing:
        - 'ra', 'dec': coordinates
        - 'distance', 'distance_error': distance data
        - 'v_pec': peculiar velocity
        - 'v_pec_error': error on peculiar velocity
        - 'method': distance method
    """
    n_galaxies = len(ra)

    # Default redshift error (assumed small)
    if cz_error is None:
        cz_error = np.zeros(n_galaxies)

    # Ensure method is array
    if isinstance(method, str):
        method = np.full(n_galaxies, method)

    # Compute peculiar velocity
    v_pec = compute_peculiar_velocity(cz, distance, H0)

    # Compute intrinsic scatter for each galaxy
    intrinsic_scatter = np.array([get_intrinsic_scatter(m) for m in method])

    # Compute error on peculiar velocity
    v_pec_error = compute_peculiar_velocity_error(
        cz_error,
        distance_error,
        distance,
        intrinsic_scatter,
        H0,
    )

    return {
        "ra": ra,
        "dec": dec,
        "cz": cz,
        "cz_error": cz_error,
        "distance": distance,
        "distance_error": distance_error,
        "v_pec": v_pec,
        "v_pec_error": v_pec_error,
        "method": method,
    }


def apply_velocity_corrections(
    v_pec: NDArray[np.float64],
    v_pec_error: NDArray[np.float64],
    velocity_corrections: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply corrections to peculiar velocities.

    Parameters
    ----------
    v_pec : ndarray
        Peculiar velocity in km/s.
    v_pec_error : ndarray
        Error on peculiar velocity.
    velocity_corrections : ndarray, optional
        Corrections to apply.

    Returns
    -------
    tuple of ndarray
        Corrected peculiar velocity and error.
    """
    if velocity_corrections is not None:
        v_pec = v_pec - velocity_corrections

    return v_pec, v_pec_error


def estimate_bulk_flow(
    v_pec: NDArray[np.float64],
    v_pec_error: NDArray[np.float64],
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    distance: NDArray[np.float64],
    r_max: Optional[float] = None,
) -> Dict[str, float]:
    """
    Estimate bulk flow from peculiar velocities.

    Parameters
    ----------
    v_pec : ndarray
        Peculiar velocities in km/s.
    v_pec_error : ndarray
        Errors on peculiar velocities.
    ra : ndarray
        Right Ascension in degrees.
    dec : ndarray
        Declination in degrees.
    distance : ndarray
        Distances in Mpc.
    r_max : float, optional
        Maximum distance to include.

    Returns
    -------
    dict
        Bulk flow components and magnitude.
    """
    # Filter by distance if specified
    if r_max is not None:
        mask = distance <= r_max
        v_pec = v_pec[mask]
        v_pec_error = v_pec_error[mask]
        ra = ra[mask]
        dec = dec[mask]
        distance = distance[mask]

    # Convert to Cartesian
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    # Unit vectors
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    # Weighted mean velocity (inverse variance weighting)
    weights = 1.0 / v_pec_error**2
    weights = weights / np.sum(weights)

    v_x = np.sum(weights * v_pec * x)
    v_y = np.sum(weights * v_pec * y)
    v_z = np.sum(weights * v_pec * z)

    # Bulk flow magnitude
    B = np.sqrt(v_x**2 + v_y**2 + v_z**2)

    # Error on bulk flow (simplified)
    B_error = np.sqrt(np.sum((weights * v_pec)**2))

    return {
        "B_x": v_x,
        "B_y": v_y,
        "B_z": v_z,
        "B": B,
        "B_error": B_error,
        "n_galaxies": len(v_pec),
    }
