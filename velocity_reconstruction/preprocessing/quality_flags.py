"""Quality flags and data validation module."""

import numpy as np
from typing import Dict, List, Optional, Tuple


def flag_low_snr(v_pec: np.ndarray, v_pec_error: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Flag measurements with low signal-to-noise ratio."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (v_pec_error / np.abs(v_pec)) > threshold
    return np.nan_to_num(result, nan=False)


def flag_galactic_plane(dec: np.ndarray, galactic_lat_threshold: float = 5.0) -> np.ndarray:
    """Flag sources within Galactic plane margin."""
    return np.abs(dec) < galactic_lat_threshold


def flag_unphysical_velocities(v_pec: np.ndarray, v_min: float = -2000.0, v_max: float = 2500.0) -> np.ndarray:
    """Flag unphysical negative peculiar velocities.
    
    Peculiar velocities can legitimately range from ~-2000 to +2500 km/s
    for distant galaxies in massive clusters or voids.
    """
    return (v_pec < v_min) | (v_pec > v_max)


def flag_outliers(data: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """Flag outliers beyond n-sigma from median."""
    # Handle NaN values
    clean_data = data[np.isfinite(data)]
    if len(clean_data) == 0:
        return np.zeros(len(data), dtype=bool)
    median = np.median(clean_data)
    std = np.std(clean_data)
    if std == 0:
        return np.zeros(len(data), dtype=bool)
    return np.abs(data - median) > n_sigma * std


def validate_catalog(ra: np.ndarray, dec: np.ndarray, cz: np.ndarray,
                     distance: np.ndarray, v_pec: np.ndarray, v_pec_error: np.ndarray,
                     config: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """
    Validate galaxy catalog and apply quality flags.
    
    Parameters
    ----------
    ra, dec : ndarray
        Coordinates in degrees.
    cz : ndarray
        Redshift in km/s.
    distance : ndarray
        Distance in Mpc.
    v_pec : ndarray
        Peculiar velocity in km/s.
    v_pec_error : ndarray
        Error on peculiar velocity.
    config : dict, optional
        Quality thresholds.
        
    Returns
    -------
    dict
        Dictionary of masks for each quality flag.
    """
    if config is None:
        config = {"max_error_ratio": 0.5, "galactic_plane_margin": 5.0, "v_min": -2000.0, "v_max": 2500.0}
    
    masks = {}
    masks["low_snr"] = flag_low_snr(v_pec, v_pec_error, config.get("max_error_ratio", 0.5))
    masks["galactic_plane"] = flag_galactic_plane(dec, config.get("galactic_plane_margin", 5.0))
    masks["unphysical"] = flag_unphysical_velocities(
        v_pec, 
        config.get("v_min", -2000.0),
        config.get("v_max", 2500.0)
    )
    masks["outlier_distance"] = flag_outliers(distance, n_sigma=4.0)
    masks["outlier_velocity"] = flag_outliers(v_pec, n_sigma=4.0)
    
    # Combined mask (True = good quality)
    masks["good"] = ~(masks["low_snr"] | masks["unphysical"] | masks["outlier_distance"] | masks["outlier_velocity"])
    
    return masks


def filter_catalog_by_quality(ra: np.ndarray, dec: np.ndarray, cz: np.ndarray,
                              distance: np.ndarray, v_pec: np.ndarray, v_pec_error: np.ndarray,
                              masks: Dict[str, np.ndarray]) -> Tuple:
    """Filter catalog by quality masks."""
    good = masks["good"]
    return ra[good], dec[good], cz[good], distance[good], v_pec[good], v_pec_error[good]


def compute_quality_summary(masks: Dict[str, np.ndarray], n_total: int) -> Dict[str, int]:
    """Compute summary of quality flags."""
    return {key: np.sum(flag) for key, flag in masks.items() if key != "good"}
