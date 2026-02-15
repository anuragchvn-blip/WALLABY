"""
Coordinate transformations for velocity field reconstruction.

This module implements coordinate transformations between:
- Equatorial coordinates (RA, Dec) - J2000 epoch
- Galactic coordinates (l, b)
- Supergalactic coordinates (SGL, SGB) following Lahav et al. (2000)

The supergalactic coordinate system has its origin at the center of the
Local Group, with the supergalactic plane approximately coinciding with
the plane of the Local Supercluster.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, Tuple, Union

# Supergalactic north pole in J2000 equatorial coordinates
# From Lahav et al. (2000)
SGL_NORTH_RA = 283.825  # degrees
SGL_NORTH_DEC = 15.616  # degrees

# Supergalactic longitude zero-point (intersection of supergalactic
# and galactic planes in the region of the Galactic anti-center)
SGL_ZERO = 137.37  # degrees (SGL = 0 at SGB = 0)

# Conversion constants
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi


def _rotation_matrix_to_galactic() -> NDArray[np.float64]:
    """
    Compute rotation matrix from equatorial to galactic coordinates.

    Returns
    -------
    ndarray
        3x3 rotation matrix.
    """
    # North Galactic Pole (NGP) in J2000 equatorial coordinates
    alpha_p = 192.85948 * DEG_TO_RAD
    delta_p = 27.12825 * DEG_TO_RAD

    # Galactic longitude of the NCP
    l_cp = 122.93192 * DEG_TO_RAD

    # Build rotation matrix
    R = np.array([
        [
            -np.sin(alpha_p) * np.cos(l_cp) - np.cos(alpha_p) * np.sin(delta_p) * np.sin(l_cp),
            -np.sin(alpha_p) * np.sin(l_cp) + np.cos(alpha_p) * np.sin(delta_p) * np.cos(l_cp),
            np.cos(alpha_p) * np.cos(delta_p),
        ],
        [
            np.cos(alpha_p) * np.cos(l_cp) - np.sin(alpha_p) * np.sin(delta_p) * np.sin(l_cp),
            np.cos(alpha_p) * np.sin(l_cp) + np.sin(alpha_p) * np.sin(delta_p) * np.cos(l_cp),
            np.sin(alpha_p) * np.cos(delta_p),
        ],
        [
            np.sin(delta_p) * np.sin(l_cp),
            -np.sin(delta_p) * np.cos(l_cp),
            np.cos(delta_p),
        ],
    ])

    return R


def equatorial_to_galactic(
    ra: Union[float, NDArray[np.float64]],
    dec: Union[float, NDArray[np.float64]],
) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]:
    """
    Convert equatorial (RA, Dec) to Galactic (l, b) coordinates.

    Parameters
    ----------
    ra : float or ndarray
        Right Ascension in degrees (J2000).
    dec : float or ndarray
        Declination in degrees (J2000).

    Returns
    -------
    tuple of (float or ndarray, float or ndarray)
        Galactic longitude l and latitude b in degrees.
    """
    ra_rad = np.asarray(ra) * DEG_TO_RAD
    dec_rad = np.asarray(dec) * DEG_TO_RAD

    x_eq = np.cos(dec_rad) * np.cos(ra_rad)
    y_eq = np.cos(dec_rad) * np.sin(ra_rad)
    z_eq = np.sin(dec_rad)

    R = _rotation_matrix_to_galactic()
    x_gal = R[0, 0] * x_eq + R[0, 1] * y_eq + R[0, 2] * z_eq
    y_gal = R[1, 0] * x_eq + R[1, 1] * y_eq + R[1, 2] * z_eq
    z_gal = R[2, 0] * x_eq + R[2, 1] * y_eq + R[2, 2] * z_eq

    l_rad = np.arctan2(y_gal, x_gal)
    l_rad = np.mod(l_rad, 2 * np.pi)
    b_rad = np.arcsin(np.clip(z_gal, -1.0, 1.0))

    return l_rad * RAD_TO_DEG, b_rad * RAD_TO_DEG


def galactic_to_equatorial(
    l: Union[float, NDArray[np.float64]],
    b: Union[float, NDArray[np.float64]],
) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]:
    """
    Convert Galactic (l, b) to equatorial (RA, Dec) coordinates.

    Parameters
    ----------
    l : float or ndarray
        Galactic longitude in degrees.
    b : float or ndarray
        Galactic latitude in degrees.

    Returns
    -------
    tuple of (float or ndarray, float or ndarray)
        Right Ascension and Declination in degrees (J2000).
    """
    l_rad = np.asarray(l) * DEG_TO_RAD
    b_rad = np.asarray(b) * DEG_TO_RAD

    x_gal = np.cos(b_rad) * np.cos(l_rad)
    y_gal = np.cos(b_rad) * np.sin(l_rad)
    z_gal = np.sin(b_rad)

    R = _rotation_matrix_to_galactic()
    R_inv = R.T

    x_eq = R_inv[0, 0] * x_gal + R_inv[0, 1] * y_gal + R_inv[0, 2] * z_gal
    y_eq = R_inv[1, 0] * x_gal + R_inv[1, 1] * y_gal + R_inv[1, 2] * z_gal
    z_eq = R_inv[2, 0] * x_gal + R_inv[2, 1] * y_gal + R_inv[2, 2] * z_gal

    ra_rad = np.arctan2(y_eq, x_eq)
    ra_rad = np.mod(ra_rad, 2 * np.pi)
    dec_rad = np.arcsin(np.clip(z_eq, -1.0, 1.0))

    return ra_rad * RAD_TO_DEG, dec_rad * RAD_TO_DEG


def equatorial_to_supergalactic(
    ra: Union[float, NDArray[np.float64]],
    dec: Union[float, NDArray[np.float64]],
) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]:
    """
    Convert equatorial (RA, Dec) to supergalactic (SGL, SGB) coordinates.

    Following Lahav et al. (2000).

    Parameters
    ----------
    ra : float or ndarray
        Right Ascension in degrees (J2000).
    dec : float or ndarray
        Declination in degrees (J2000).

    Returns
    -------
    tuple of (float or ndarray, float or ndarray)
        Supergalactic longitude (SGL) and latitude (SGB) in degrees.
    """
    ra_rad = np.asarray(ra) * DEG_TO_RAD
    dec_rad = np.asarray(dec) * DEG_TO_RAD

    alpha_p = SGL_NORTH_RA * DEG_TO_RAD
    delta_p = SGL_NORTH_DEC * DEG_TO_RAD

    cos_sgb = (np.sin(dec_rad) * np.sin(delta_p) +
               np.cos(dec_rad) * np.cos(delta_p) * np.cos(ra_rad - alpha_p))
    sgb_rad = np.arcsin(np.clip(cos_sgb, -1.0, 1.0))

    numerator = np.cos(dec_rad) * np.sin(ra_rad - alpha_p)
    denominator = (np.cos(delta_p) * np.sin(dec_rad) -
                   np.sin(delta_p) * np.cos(dec_rad) * np.cos(ra_rad - alpha_p))

    sgl_rad = np.arctan2(numerator, denominator)

    sgl_zero_rad = SGL_ZERO * DEG_TO_RAD
    sgl_rad = sgl_rad + sgl_zero_rad - np.pi / 2

    return sgl_rad * RAD_TO_DEG, sgb_rad * RAD_TO_DEG


def supergalactic_to_equatorial(
    sgl: Union[float, NDArray[np.float64]],
    sgb: Union[float, NDArray[np.float64]],
) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]:
    """
    Convert supergalactic (SGL, SGB) to equatorial (RA, Dec) coordinates.

    Parameters
    ----------
    sgl : float or ndarray
        Supergalactic longitude in degrees.
    sgb : float or ndarray
        Supergalactic latitude in degrees.

    Returns
    -------
    tuple of (float or ndarray, float or ndarray)
        Right Ascension and Declination in degrees (J2000).
    """
    sgl_rad = np.asarray(sgl) * DEG_TO_RAD
    sgb_rad = np.asarray(sgb) * DEG_TO_RAD

    alpha_p = SGL_NORTH_RA * DEG_TO_RAD
    delta_p = SGL_NORTH_DEC * DEG_TO_RAD

    sin_dec = (np.sin(sgb_rad) * np.sin(delta_p) +
               np.cos(sgb_rad) * np.cos(delta_p) * np.cos(sgl_rad))
    dec_rad = np.arcsin(np.clip(sin_dec, -1.0, 1.0))

    numerator = np.sin(sgl_rad) * np.cos(sgb_rad)
    denominator = (np.cos(sgb_rad) * np.cos(sgl_rad) * np.sin(delta_p) -
                   np.sin(sgb_rad) * np.cos(delta_p))

    ra_rad = alpha_p + np.arctan2(numerator, denominator)
    ra_rad = np.mod(ra_rad, 2 * np.pi)

    return ra_rad * RAD_TO_DEG, dec_rad * RAD_TO_DEG


def galactic_to_supergalactic(
    l: Union[float, NDArray[np.float64]],
    b: Union[float, NDArray[np.float64]],
) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]:
    """
    Convert galactic (l, b) to supergalactic (SGL, SGB) coordinates.

    Parameters
    ----------
    l : float or ndarray
        Galactic longitude in degrees.
    b : float or ndarray
        Galactic latitude in degrees.

    Returns
    -------
    tuple of (float or ndarray, float or ndarray)
        Supergalactic longitude (SGL) and latitude (SGB) in degrees.
    """
    ra, dec = galactic_to_equatorial(l, b)
    return equatorial_to_supergalactic(ra, dec)


def supergalactic_to_galactic(
    sgl: Union[float, NDArray[np.float64]],
    sgb: Union[float, NDArray[np.float64]],
) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]:
    """
    Convert supergalactic (SGL, SGB) to galactic (l, b) coordinates.

    Parameters
    ----------
    sgl : float or ndarray
        Supergalactic longitude in degrees.
    sgb : float or ndarray
        Supergalactic latitude in degrees.

    Returns
    -------
    tuple of (float or ndarray, float or ndarray)
        Galactic longitude and latitude in degrees.
    """
    ra, dec = supergalactic_to_equatorial(sgl, sgb)
    return galactic_to_equatorial(ra, dec)


def equatorial_to_cartesian(
    ra: Union[float, NDArray[np.float64]],
    dec: Union[float, NDArray[np.float64]],
    distance: Union[float, NDArray[np.float64]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert spherical coordinates to Cartesian.

    Parameters
    ----------
    ra : float or ndarray
        Right Ascension in degrees.
    dec : float or ndarray
        Declination in degrees.
    distance : float or ndarray
        Distance in same units (typically Mpc).

    Returns
    -------
    tuple of ndarray
        x, y, z coordinates.
    """
    ra_rad = np.asarray(ra) * DEG_TO_RAD
    dec_rad = np.asarray(dec) * DEG_TO_RAD
    dist = np.asarray(distance)

    x = dist * np.cos(dec_rad) * np.cos(ra_rad)
    y = dist * np.cos(dec_rad) * np.sin(ra_rad)
    z = dist * np.sin(dec_rad)

    return x, y, z


def cartesian_to_equatorial(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert Cartesian coordinates to spherical.

    Parameters
    ----------
    x, y, z : ndarray
        Cartesian coordinates.

    Returns
    -------
    tuple of ndarray
        RA (degrees), Dec (degrees), Distance.
    """
    distance = np.sqrt(x**2 + y**2 + z**2)

    with np.errstate(invalid='ignore'):
        dec_rad = np.arcsin(z / np.where(distance > 0, distance, 1))
        ra_rad = np.arctan2(y, x)

    dec_rad = np.where(distance > 0, dec_rad, 0.0)
    ra_rad = np.where(distance > 0, ra_rad, 0.0)

    return ra_rad * RAD_TO_DEG, dec_rad * RAD_TO_DEG, distance


def supergalactic_to_cartesian(
    sgl: Union[float, NDArray[np.float64]],
    sgb: Union[float, NDArray[np.float64]],
    distance: Union[float, NDArray[np.float64]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert supergalactic spherical to Cartesian coordinates.

    Parameters
    ----------
    sgl : float or ndarray
        Supergalactic longitude in degrees.
    sgb : float or ndarray
        Supergalactic latitude in degrees.
    distance : float or ndarray
        Distance in same units (typically Mpc).

    Returns
    -------
    tuple of ndarray
        x, y, z in supergalactic Cartesian coordinates.
    """
    sgl_rad = np.asarray(sgl) * DEG_TO_RAD
    sgb_rad = np.asarray(sgb) * DEG_TO_RAD
    dist = np.asarray(distance)

    x = dist * np.cos(sgb_rad) * np.cos(sgl_rad)
    y = dist * np.cos(sgb_rad) * np.sin(sgl_rad)
    z = dist * np.sin(sgb_rad)

    return x, y, z


def cartesian_to_supergalactic(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert supergalactic Cartesian to spherical coordinates.

    Parameters
    ----------
    x, y, z : ndarray
        Cartesian coordinates in supergalactic system.

    Returns
    -------
    tuple of ndarray
        SGL (degrees), SGB (degrees), Distance.
    """
    distance = np.sqrt(x**2 + y**2 + z**2)

    with np.errstate(invalid='ignore'):
        sgb_rad = np.arcsin(z / np.where(distance > 0, distance, 1))
        sgl_rad = np.arctan2(y, x)

    sgb_rad = np.where(distance > 0, sgb_rad, 0.0)
    sgl_rad = np.where(distance > 0, sgl_rad, 0.0)

    return sgl_rad * RAD_TO_DEG, sgb_rad * RAD_TO_DEG, distance


def round_trip_test(
    n_points: int = 1000,
    seed: Optional[int] = 42,
) -> Dict[str, float]:
    """
    Test coordinate transformations for round-trip precision.

    Parameters
    ----------
    n_points : int
        Number of random points to test.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Maximum errors for each transformation pair.
    """
    if seed is not None:
        np.random.seed(seed)

    ra_orig = np.random.uniform(0, 360, n_points)
    dec_orig = np.random.uniform(-90, 90, n_points)

    l, b = equatorial_to_galactic(ra_orig, dec_orig)
    ra_rec, dec_rec = galactic_to_equatorial(l, b)

    eq_errors = {
        "ra_max_error": np.max(np.abs(ra_orig - ra_rec)),
        "dec_max_error": np.max(np.abs(dec_orig - dec_rec)),
    }

    l_rec, b_rec = galactic_to_equatorial(l, b)

    gal_errors = {
        "l_max_error": np.max(np.abs(l - l_rec)),
        "b_max_error": np.max(np.abs(b - b_rec)),
    }

    return {**eq_errors, **gal_errors}


CoordinateArray = NDArray[np.float64]
