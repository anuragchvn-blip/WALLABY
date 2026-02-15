"""Extinction correction module using SFD98 and CCM89."""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def ebv_to_ak(ebv: float) -> float:
    """Convert E(B-V) to K-band extinction A_K = 0.112 * E(B-V)."""
    return 0.112 * ebv


def cardelli_extinction(wavelength: float) -> float:
    """Compute A_lambda/A_V using CCM89 law."""
    x = 10000.0 / wavelength
    if x < 0.3 or x > 30:
        return 0.0
    if 0.3 <= x < 1.1:
        y = x
        return 0.574 * y**1.61 - 0.527 * y**1.61 + 1
    if 1.1 <= x < 3.3:
        y = x
        return 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    return 1.0


def query_sfd98(ra: NDArray, dec: NDArray) -> NDArray:
    """Query SFD98 dust map at coordinates (placeholder)."""
    return np.zeros_like(ra, dtype=float)


def correct_extinction(mag: NDArray, ebv: NDArray, wavelength: float) -> NDArray:
    """Apply extinction correction to magnitudes."""
    a_lambda = ebv * cardelli_extinction(wavelength)
    return mag - a_lambda