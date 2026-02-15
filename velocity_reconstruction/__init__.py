"""
Velocity Field to Mass Distribution Reconstruction System

A production-grade pipeline that ingests galaxy peculiar velocity measurements
from multiple surveys and reconstructs the 3D gravitational potential,
velocity field, and mass density distribution of the local universe.
"""

__version__ = "0.1.0"
__author__ = "Velocity Reconstruction Team"

from velocity_reconstruction import config

__all__ = ["config"]
