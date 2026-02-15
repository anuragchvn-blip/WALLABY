"""
Catalog readers for galaxy surveys.

Provides readers for CosmicFlows-4, WALLABY, FAST, DESI, and 2MASS catalogs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class GalaxyCatalog:
    """Container for galaxy catalog data."""

    def __init__(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        galaxy_id: Optional[np.ndarray] = None,
        cz: Optional[np.ndarray] = None,
        distance: Optional[np.ndarray] = None,
        distance_error: Optional[np.ndarray] = None,
        method: Optional[np.ndarray] = None,
        **kwargs,
    ):
        self.ra = np.asarray(ra)
        self.dec = np.asarray(dec)
        self.n_galaxies = len(ra)
        self.galaxy_id = galaxy_id if galaxy_id is not None else np.arange(self.n_galaxies)
        self.cz = cz if cz is not None else np.zeros(self.n_galaxies)
        self.distance = distance if distance is not None else np.zeros(self.n_galaxies)
        self.distance_error = distance_error if distance_error is not None else np.zeros(self.n_galaxies)
        self.method = method if method is not None else np.full(self.n_galaxies, "Unknown")

        for key, value in kwargs.items():
            setattr(self, key, np.asarray(value))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            "ra": self.ra,
            "dec": self.dec,
            "galaxy_id": self.galaxy_id,
            "cz": self.cz,
            "distance": self.distance,
            "distance_error": self.distance_error,
            "method": self.method,
        })

    def filter(
        self,
        min_distance: Optional[float] = None,
        max_distance: Optional[float] = None,
        methods: Optional[List[str]] = None,
    ) -> "GalaxyCatalog":
        """Filter catalog by criteria."""
        mask = np.ones(self.n_galaxies, dtype=bool)

        if min_distance is not None:
            mask &= self.distance >= min_distance
        if max_distance is not None:
            mask &= self.distance <= max_distance
        if methods is not None:
            mask &= np.isin(self.method, methods)

        kwargs = {}
        for attr in dir(self):
            if not attr.startswith("_"):
                val = getattr(self, attr)
                if isinstance(val, np.ndarray) and len(val) == self.n_galaxies:
                    kwargs[attr] = val[mask]

        return GalaxyCatalog(**kwargs)


def read_cosmicflows4(filepath: Union[str, Path]) -> GalaxyCatalog:
    """
    Read CosmicFlows-4 catalog from file.

    Supports multiple column formats:
    - VizieR download: RAJ2000, DEJ2000, Vcmb, DM, e_DM
    - Standard format: RAdeg, DEdeg, cz, dist, dist_err
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() in [".fits", ".fit"]:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            return GalaxyCatalog(
                ra=data["RAdeg"],
                dec=data["DEdeg"],
                cz=data["cz"],
                distance=data["dist"],
                distance_error=data["dist_err"],
                galaxy_id=data["name"],
                method=data["method"],
            )
    else:
        df = pd.read_csv(filepath)

        # Handle VizieR format columns
        if "RAJ2000" in df.columns and "Vcmb" in df.columns:
            # VizieR format: convert RA/Dec and distance modulus
            ra = pd.to_numeric(df["RAJ2000"], errors="coerce").values
            dec = pd.to_numeric(df["DEJ2000"], errors="coerce").values
            vcmb = pd.to_numeric(df["Vcmb"], errors="coerce").values
            dm = pd.to_numeric(df["DM"], errors="coerce").values
            e_dm = pd.to_numeric(df["e_DM"], errors="coerce").values

            # Convert distance modulus to distance (Mpc)
            # DM = 5 * log10(D/10pc) => D = 10^((DM-25)/5) Mpc
            # (This is equivalent to D = 10^(DM/5) / 10^5 Mpc)
            distance = np.where(np.isfinite(dm), 10.0 ** ((dm - 25.0) / 5.0), np.nan)
            # Convert DM error to distance error
            # dD/D = ln(10)/5 * dDM
            distance_error = np.where(
                np.isfinite(e_dm),
                distance * (np.log(10.0) / 5.0) * e_dm,
                np.nan
            )

            # Determine method from available columns
            method = np.full(len(df), "Unknown", dtype=object)
            if "DMtf" in df.columns:
                tf_mask = pd.to_numeric(df["DMtf"], errors="coerce").notna()
                method[tf_mask] = "TF"
            if "DMfp" in df.columns:
                fp_mask = pd.to_numeric(df["DMfp"], errors="coerce").notna()
                method[fp_mask] = "FP"
            if "DMsnIa" in df.columns:
                sn_mask = pd.to_numeric(df["DMsnIa"], errors="coerce").notna()
                method[sn_mask] = "SN Ia"

            return GalaxyCatalog(
                ra=ra,
                dec=dec,
                cz=vcmb,
                distance=distance,
                distance_error=distance_error,
                galaxy_id=df["PGC"].values if "PGC" in df.columns else np.arange(len(df)),
                method=method,
            )
        else:
            # Standard format
            return GalaxyCatalog(
                ra=df["RAdeg"].values,
                dec=df["DEdeg"].values,
                cz=df["cz"].values,
                distance=df["dist"].values,
                distance_error=df["dist_err"].values,
                galaxy_id=df["name"].values,
                method=df["method"].values,
            )


def read_wallaby(filepath: Union[str, Path]) -> GalaxyCatalog:
    """Read WALLABY PDR2 catalog from file."""
    filepath = Path(filepath)
    if filepath.suffix.lower() in [".fits", ".fit"]:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            ra = data["ra"]
            dec = data["dec"]
    else:
        df = pd.read_csv(filepath)
        ra = df["ra"].values
        dec = df["dec"].values

    return GalaxyCatalog(
        ra=ra,
        dec=dec,
        galaxy_id=np.arange(len(ra)),
    )


def read_fast(filepath: Union[str, Path]) -> GalaxyCatalog:
    """Read FAST extragalactic HI survey catalog from file."""
    filepath = Path(filepath)
    if filepath.suffix.lower() in [".fits", ".fit"]:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            ra = data["RA"]
            dec = data["DEC"]
    else:
        df = pd.read_csv(filepath)
        ra = df["RA"].values
        dec = df["DEC"].values

    return GalaxyCatalog(ra=ra, dec=dec, galaxy_id=np.arange(len(ra)))


def read_desi(filepath: Union[str, Path]) -> GalaxyCatalog:
    """Read DESI peculiar velocity survey catalog from file."""
    filepath = Path(filepath)
    if filepath.suffix.lower() in [".fits", ".fit"]:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            ra = data["ra"]
            dec = data["dec"]
            z = data["z"]
            distance = data["distance"]
            distance_error = data["distance_err"]
    else:
        df = pd.read_csv(filepath)
        ra = df["ra"].values
        dec = df["dec"].values
        z = df["z"].values
        distance = df["distance"].values
        distance_error = df["distance_err"].values

    cz = z * 299792.458
    return GalaxyCatalog(
        ra=ra, dec=dec, cz=cz,
        distance=distance, distance_error=distance_error,
        method=np.full(len(ra), "FP"),
    )


def read_twomass(filepath: Union[str, Path]) -> GalaxyCatalog:
    """Read 2MASS ZoA bright galaxy catalog from file."""
    filepath = Path(filepath)
    if filepath.suffix.lower() in [".fits", ".fit"]:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            ra = data["RAJ2000"]
            dec = data["DEJ2000"]
            k_mag = data["k_m"]
    else:
        df = pd.read_csv(filepath)
        ra = df["RAJ2000"].values
        dec = df["DEJ2000"].values
        k_mag = df["k_m"].values

    return GalaxyCatalog(ra=ra, dec=dec, k_mag=k_mag)


def load_survey_catalog(survey: str, filepath: Union[str, Path]) -> GalaxyCatalog:
    """Load catalog by survey name."""
    survey = survey.lower()
    if survey == "cosmicflows4":
        return read_cosmicflows4(filepath)
    elif survey == "wallaby":
        return read_wallaby(filepath)
    elif survey == "fast":
        return read_fast(filepath)
    elif survey == "desi":
        return read_desi(filepath)
    elif survey == "twomass":
        return read_twomass(filepath)
    else:
        raise ValueError(f"Unknown survey: {survey}")


def merge_catalogs(catalogs: List[GalaxyCatalog]) -> GalaxyCatalog:
    """Merge multiple catalogs."""
    all_ra = np.concatenate([c.ra for c in catalogs])
    all_dec = np.concatenate([c.dec for c in catalogs])
    all_id = np.concatenate([c.galaxy_id for c in catalogs])
    all_cz = np.concatenate([c.cz for c in catalogs])
    all_dist = np.concatenate([c.distance for c in catalogs])
    all_dist_err = np.concatenate([c.distance_error for c in catalogs])
    all_method = np.concatenate([c.method for c in catalogs])

    return GalaxyCatalog(
        ra=all_ra, dec=all_dec, galaxy_id=all_id,
        cz=all_cz, distance=all_dist, distance_error=all_dist_err,
        method=all_method,
    )
