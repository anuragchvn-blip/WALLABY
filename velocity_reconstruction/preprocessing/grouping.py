"""Galaxy grouping using friends-of-friends algorithm."""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple
from scipy.spatial import cKDTree


def compute_linking_length(Omega_m: float = 0.315, b: float = 0.25) -> float:
    """Compute FoF linking length.
    
    b = 0.25 * (Omega_m / 0.3) ^ (-0.6) Mpc
    """
    return b * (Omega_m / 0.3) ** (-0.6)


def friends_of_friends(
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    cz: NDArray[np.float64],
    linking_length: float = 8.0,
    velocity_link: float = 500.0,
) -> NDArray[np.int64]:
    """Friends-of-friends galaxy grouping.
    
    Parameters
    ----------
    ra : ndarray
        Right Ascension in degrees.
    dec : ndarray
        Declination in degrees.
    cz : ndarray
        Redshift in km/s.
    linking_length : float
        Projected linking length in Mpc.
    velocity_link : float
        Velocity linking length in km/s.
        
    Returns
    -------
    ndarray
        Group ID for each galaxy.
    """
    n = len(ra)
    
    # Convert to 3D coordinates (simplified - assumes small angle)
    # For proper implementation, use angular diameter distance
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    
    x = cz * np.cos(dec_rad) * np.cos(ra_rad) / 70.0  # approx Mpc
    y = cz * np.cos(dec_rad) * np.sin(ra_rad) / 70.0
    z = cz * np.sin(dec_rad) / 70.0
    
    coords = np.column_stack([x, y, z])
    
    # Build KD-tree
    tree = cKDTree(coords)
    
    # Find all pairs within linking length
    pairs = tree.query_pairs(linking_length)
    
    # Union-find algorithm
    parent = np.arange(n)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i, j in pairs:
        # Check velocity separation
        if abs(cz[i] - cz[j]) < velocity_link:
            union(i, j)
    
    # Compress group IDs
    groups = np.array([find(i) for i in range(n)])
    unique_groups = np.unique(groups)
    group_map = {old: new for new, old in enumerate(unique_groups)}
    
    return np.array([group_map[g] for g in groups])


def compute_group_properties(
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    cz: NDArray[np.float64],
    cz_error: NDArray[np.float64],
    group_ids: NDArray[np.int64],
) -> Dict[str, NDArray]:
    """Compute group-averaged properties.
    
    Returns
    -------
    dict
        Group properties including ra, dec, cz, cz_error, n_members.
    """
    unique_groups = np.unique(group_ids)
    
    group_ra = []
    group_dec = []
    group_cz = []
    group_cz_err = []
    group_n = []
    
    for g in unique_groups:
        mask = group_ids == g
        w = 1.0 / cz_error[mask]**2
        w_sum = np.sum(w)
        
        group_ra.append(np.sum(ra[mask] * w) / w_sum)
        group_dec.append(np.sum(dec[mask] * w) / w_sum)
        group_cz.append(np.sum(cz[mask] * w) / w_sum)
        group_cz_err.append(np.sqrt(1.0 / w_sum))
        group_n.append(np.sum(mask))
    
    return {
        "group_id": unique_groups,
        "ra": np.array(group_ra),
        "dec": np.array(group_dec),
        "cz": np.array(group_cz),
        "cz_error": np.array(group_cz_err),
        "n_members": np.array(group_n),
    }