"""
Configuration management for velocity field reconstruction.

This module provides configuration loading, validation, and management
for all pipeline components following the prop.md specification.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


@dataclass
class CosmologyConfig:
    """Cosmological parameters from Planck 2018."""

    H0: float = 70.0  # Hubble constant in km/s/Mpc
    Omega_m: float = 0.315  # Matter density parameter
    Omega_Lambda: float = 0.685  # Dark energy density
    sigma8: float = 0.811  # RMS fluctuation amplitude
    n_s: float = 0.965  # Scalar spectral index
    f_growth: float = field(init=False)  # Growth rate f = Omega_m^0.55

    def __post_init__(self):
        """Compute derived quantities."""
        self.f_growth = self.Omega_m ** 0.55

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "H0": self.H0,
            "Omega_m": self.Omega_m,
            "Omega_Lambda": self.Omega_Lambda,
            "sigma8": self.sigma8,
            "n_s": self.n_s,
            "f_growth": self.f_growth,
        }


@dataclass
class GridConfig:
    """Grid configuration for reconstruction."""

    extent: float = 100.0  # Half-extent in Mpc (covers ±100 Mpc)
    resolution: int = 200  # Number of cells per dimension (200^3 total)
    cell_size: float = field(init=False)  # Cell size in Mpc
    smoothing_sigma: float = 10.0  # Gaussian smoothing scale in Mpc

    def __post_init__(self):
        """Compute derived quantities."""
        self.cell_size = 2 * self.extent / self.resolution

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "extent": self.extent,
            "resolution": self.resolution,
            "cell_size": self.cell_size,
            "smoothing_sigma": self.smoothing_sigma,
        }


@dataclass
class DataPathsConfig:
    """Paths to input data catalogs."""

    cosmicflows4: Optional[str] = None
    wallaby_pdr2: Optional[str] = None
    fast_survey: Optional[str] = None
    desi_pv: Optional[str] = None
    twomass_zoa: Optional[str] = None
    sfd98_maps: Optional[str] = None  # Schlegel-Finkbeiner-Davis 1998 dust maps

    output_dir: str = "./output"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "cosmicflows4": self.cosmicflows4,
            "wallaby_pdr2": self.wallaby_pdr2,
            "fast_survey": self.fast_survey,
            "desi_pv": self.desi_pv,
            "twomass_zoa": self.twomass_zoa,
            "sfd98_maps": self.sfd98_maps,
            "output_dir": self.output_dir,
        }


@dataclass
class AlgorithmConfig:
    """Algorithm selection and parameters."""

    use_potent: bool = True
    use_wiener_filter: bool = True
    use_bayesian: bool = True

    # POTENT parameters
    potent_smoothing_range: List[float] = field(default_factory=lambda: [10.0, 12.0])
    potent_max_iterations: int = 10000
    potent_convergence_tolerance: float = 1e-6

    # Wiener Filter parameters
    wf_num_realizations: int = 100
    wf_kmax: float = 0.5  # Max k in h/Mpc for reconstruction

    # Bayesian Inference parameters
    bayesian_chains: int = 4
    bayesian_iterations: int = 2000
    bayesian_warmup: int = 1000
    bayesian_thinning: int = 5

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "use_potent": self.use_potent,
            "use_wiener_filter": self.use_wiener_filter,
            "use_bayesian": self.use_bayesian,
            "potent_smoothing_range": self.potent_smoothing_range,
            "potent_max_iterations": self.potent_max_iterations,
            "potent_convergence_tolerance": self.potent_convergence_tolerance,
            "wf_num_realizations": self.wf_num_realizations,
            "wf_kmax": self.wf_kmax,
            "bayesian_chains": self.bayesian_chains,
            "bayesian_iterations": self.bayesian_iterations,
            "bayesian_warmup": self.bayesian_warmup,
            "bayesian_thinning": self.bayesian_thinning,
        }


@dataclass
class ComputationalConfig:
    """Computational settings."""

    num_cores: int = 1  # Number of cores for parallel processing
    use_gpu: bool = False  # GPU acceleration
    precision: str = "float64"  # float32 or float64
    memory_limit_gb: float = 16.0  # Memory limit in GB

    # FFT settings
    fft_library: str = "numpy"  # numpy, fftw, or scipy

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "num_cores": self.num_cores,
            "use_gpu": self.use_gpu,
            "precision": self.precision,
            "memory_limit_gb": self.memory_limit_gb,
            "fft_library": self.fft_library,
        }


@dataclass
class DistanceMethodConfig:
    """Distance indicator parameters."""

    # Tully-Fisher (TF)
    tf_scatter: float = 0.35  # Intrinsic scatter in mag
    tf_zero_point: float = 0.0  # Zero-point offset (calibrated)

    # Fundamental Plane (FP)
    fp_scatter: float = 0.1  # Intrinsic scatter in dex
    fp_zero_point: float = 0.0

    # Type Ia Supernovae (SNe Ia)
    sne_scatter: float = 0.15  # Intrinsic scatter in mag
    sne_zero_point: float = 0.0

    # TRGB, Cepheids, SBF, EPM, GCLF
    trgb_scatter: float = 0.15
    cepheids_scatter: float = 0.15
    sbf_scatter: float = 0.15
    epm_scatter: float = 0.20
    gclf_scatter: float = 0.20

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "tf_scatter": self.tf_scatter,
            "tf_zero_point": self.tf_zero_point,
            "fp_scatter": self.fp_scatter,
            "fp_zero_point": self.fp_zero_point,
            "sne_scatter": self.sne_scatter,
            "sne_zero_point": self.sne_zero_point,
            "trgb_scatter": self.trgb_scatter,
            "cepheids_scatter": self.cepheids_scatter,
            "sbf_scatter": self.sbf_scatter,
            "epm_scatter": self.epm_scatter,
            "gclf_scatter": self.gclf_scatter,
        }


@dataclass
class QualityConfig:
    """Data quality thresholds."""

    max_error_ratio: float = 0.5  # Reject if sigma_v / v_pec > 0.5
    galactic_plane_margin: float = 5.0  # Degrees from Galactic plane to flag
    duplicate_threshold: float = 0.2  # Distance method difference for averaging

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "max_error_ratio": self.max_error_ratio,
            "galactic_plane_margin": self.galactic_plane_margin,
            "duplicate_threshold": self.duplicate_threshold,
        }


@dataclass
class ValidationConfig:
    """Validation parameters."""

    num_mock_realizations: int = 10
    min_correlation_coefficient: float = 0.8
    max_bias_fraction: float = 0.05
    bulk_flow_tolerance: float = 50.0  # km/s
    power_spectrum_tolerance: float = 0.20  # 20%

    # Published results to compare
    cf4_bulk_flow: float = 315.0  # km/s
    cf4_bulk_flow_error: float = 40.0  # km/s
    cf4_bulk_flow_radius: float = 150.0  # Mpc

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "num_mock_realizations": self.num_mock_realizations,
            "min_correlation_coefficient": self.min_correlation_coefficient,
            "max_bias_fraction": self.max_bias_fraction,
            "bulk_flow_tolerance": self.bulk_flow_tolerance,
            "power_spectrum_tolerance": self.power_spectrum_tolerance,
            "cf4_bulk_flow": self.cf4_bulk_flow,
            "cf4_bulk_flow_error": self.cf4_bulk_flow_error,
            "cf4_bulk_flow_radius": self.cf4_bulk_flow_radius,
        }


@dataclass
class OutputConfig:
    """Output product configuration."""

    save_density_field: bool = True
    save_velocity_field: bool = True
    save_potential: bool = True
    save_uncertainty: bool = True

    # Derived products
    save_bulk_flow: bool = True
    save_divergence: bool = True
    save_isodensity: bool = False
    save_streamlines: bool = False

    # Diagnostics
    save_residuals: bool = True
    save_chi2: bool = True
    save_correlation_function: bool = True
    save_power_spectrum: bool = True

    # Visualization
    save_slice_plots: bool = True
    save_volume_render: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "save_density_field": self.save_density_field,
            "save_velocity_field": self.save_velocity_field,
            "save_potential": self.save_potential,
            "save_uncertainty": self.save_uncertainty,
            "save_bulk_flow": self.save_bulk_flow,
            "save_divergence": self.save_divergence,
            "save_isodensity": self.save_isodensity,
            "save_streamlines": self.save_streamlines,
            "save_residuals": self.save_residuals,
            "save_chi2": self.save_chi2,
            "save_correlation_function": self.save_correlation_function,
            "save_power_spectrum": self.save_power_spectrum,
            "save_slice_plots": self.save_slice_plots,
            "save_volume_render": self.save_volume_render,
        }


@dataclass
class PipelineConfig:
    """Main configuration container."""

    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    data_paths: DataPathsConfig = field(default_factory=DataPathsConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    computational: ComputationalConfig = field(default_factory=ComputationalConfig)
    distance_method: DistanceMethodConfig = field(default_factory=DistanceMethodConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Pipeline metadata
    run_name: str = "velocity_reconstruction_run"
    log_level: str = "INFO"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "cosmology": self.cosmology.to_dict(),
            "grid": self.grid.to_dict(),
            "data_paths": self.data_paths.to_dict(),
            "algorithm": self.algorithm.to_dict(),
            "computational": self.computational.to_dict(),
            "distance_method": self.distance_method.to_dict(),
            "quality": self.quality.to_dict(),
            "validation": self.validation.to_dict(),
            "output": self.output.to_dict(),
            "run_name": self.run_name,
            "log_level": self.log_level,
        }


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    PipelineConfig
        Configuration object.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    yaml.YAMLError
        If config file is invalid YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Create configuration objects from dictionary
    cosmology = CosmologyConfig(**config_dict.get("cosmology", {}))
    grid = GridConfig(**config_dict.get("grid", {}))
    data_paths = DataPathsConfig(**config_dict.get("data_paths", {}))
    algorithm = AlgorithmConfig(**config_dict.get("algorithm", {}))
    computational = ComputationalConfig(**config_dict.get("computational", {}))
    distance_method = DistanceMethodConfig(**config_dict.get("distance_method", {}))
    quality = QualityConfig(**config_dict.get("quality", {}))
    validation = ValidationConfig(**config_dict.get("validation", {}))
    output = OutputConfig(**config_dict.get("output", {}))

    config = PipelineConfig(
        cosmology=cosmology,
        grid=grid,
        data_paths=data_paths,
        algorithm=algorithm,
        computational=computational,
        distance_method=distance_method,
        quality=quality,
        validation=validation,
        output=output,
        run_name=config_dict.get("run_name", "velocity_reconstruction_run"),
        log_level=config_dict.get("log_level", "INFO"),
    )

    return config


def create_default_config(output_path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """
    Create default configuration.

    Parameters
    ----------
    output_path : str or Path, optional
        Path to save default configuration YAML file.

    Returns
    -------
    PipelineConfig
        Default configuration object.
    """
    config = PipelineConfig()

    if output_path is not None:
        output_path = Path(output_path)
        config_dict = config.to_dict()

        # Convert to YAML-friendly format (no underscores in keys)
        yaml_dict = {}
        for key, value in config_dict.items():
            yaml_dict[key] = value

        with open(output_path, "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    return config


def validate_config(config: PipelineConfig) -> List[str]:
    """
    Validate configuration parameters.

    Parameters
    ----------
    config : PipelineConfig
        Configuration to validate.

    Returns
    -------
    List[str]
        List of validation warnings (empty if no issues).
    """
    warnings = []

    # Cosmology validation
    if not 50 < config.cosmology.H0 < 100:
        warnings.append(f"Unusual H0 value: {config.cosmology.H0}")

    if not 0.2 < config.cosmology.Omega_m < 0.5:
        warnings.append(f"Unusual Omega_m value: {config.cosmology.Omega_m}")

    # Grid validation
    if config.grid.resolution < 50:
        warnings.append(f"Low grid resolution: {config.grid.resolution}")

    if config.grid.smoothing_sigma > config.grid.extent:
        warnings.append("Smoothing scale larger than grid extent")

    # Computational validation
    if config.computational.precision not in ["float32", "float64"]:
        warnings.append(f"Unknown precision: {config.computational.precision}")

    if config.computational.num_cores < 1:
        warnings.append("num_cores must be at least 1")

    # Algorithm validation
    if not any([config.algorithm.use_potent,
                config.algorithm.use_wiener_filter,
                config.algorithm.use_bayesian]):
        warnings.append("No reconstruction algorithm selected")

    return warnings
