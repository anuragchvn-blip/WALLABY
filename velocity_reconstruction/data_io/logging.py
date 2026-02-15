"""
Logging and provenance tracking for velocity field reconstruction.

This module provides structured logging with configurable levels,
provenance tracking (git commit, data versions, parameters),
and warning systems for data quality issues.
"""

import functools
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union


def get_git_commit_hash() -> Optional[str]:
    """
    Get current git commit hash.

    Returns
    -------
    str or None
        Git commit hash if in a git repository, None otherwise.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_git_branch() -> Optional[str]:
    """
    Get current git branch name.

    Returns
    -------
    str or None
        Git branch name if in a git repository, None otherwise.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


class ProvenanceLogger:
    """
    Provenance tracking for pipeline execution.

    Logs git commit hash, data versions, parameters,
    and execution metadata for reproducibility.
    """

    def __init__(self, log_dir: Optional[Union[str, Path]] = None):
        """
        Initialize provenance logger.

        Parameters
        ----------
        log_dir : str or Path, optional
            Directory for log files. Defaults to current directory.
        """
        self.log_dir = Path(log_dir) if log_dir is not None else Path(".")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Provenance information
        self.git_commit = get_git_commit_hash()
        self.git_branch = get_git_branch()
        self.start_time = None
        self.end_time = None

        # Execution metadata
        self.data_versions: Dict[str, str] = {}
        self.parameters: Dict[str, Any] = {}
        self.warnings: list = []

    def log_data_version(self, name: str, version: str) -> None:
        """
        Log data version.

        Parameters
        ----------
        name : str
            Name of the data source.
        version : str
            Version identifier.
        """
        self.data_versions[name] = version

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log parameters.

        Parameters
        ----------
        params : dict
            Parameter dictionary to log.
        """
        self.parameters.update(params)

    def log_warning(self, message: str, category: str = "general") -> None:
        """
        Log a warning with category.

        Parameters
        ----------
        message : str
            Warning message.
        category : str
            Warning category (e.g., 'data_quality', 'numerical', 'boundary').
        """
        self.warnings.append({
            "time": datetime.now().isoformat(),
            "category": category,
            "message": message,
        })

    def start_run(self, run_name: str) -> None:
        """
        Mark start of pipeline execution.

        Parameters
        ----------
        run_name : str
            Name of the run.
        """
        self.start_time = datetime.now()
        self.run_name = run_name

    def end_run(self) -> None:
        """Mark end of pipeline execution."""
        self.end_time = datetime.now()

    def get_provenance_dict(self) -> Dict[str, Any]:
        """
        Get complete provenance dictionary.

        Returns
        -------
        dict
            Provenance information including git, parameters, warnings.
        """
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "run_name": self.run_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "data_versions": self.data_versions,
            "parameters": self.parameters,
            "warnings": self.warnings,
        }

    def save_provenance(self, filename: str = "provenance.json") -> Path:
        """
        Save provenance to JSON file.

        Parameters
        ----------
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to saved provenance file.
        """
        import json

        output_path = self.log_dir / filename
        provenance = self.get_provenance_dict()

        with open(output_path, "w") as f:
            json.dump(provenance, f, indent=2)

        return output_path


class QualityWarningFilter(logging.Filter):
    """
    Custom logging filter for quality warnings.

    Categorizes warnings for later analysis.
    """

    def __init__(self, provenance_logger: ProvenanceLogger):
        """
        Initialize filter.

        Parameters
        ----------
        provenance_logger : ProvenanceLogger
            Logger to record warnings.
        """
        super().__init__()
        self.provenance = provenance_logger

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records and extract quality warnings.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to filter.

        Returns
        -------
        bool
            True to allow record through.
        """
        if record.levelno >= logging.WARNING:
            # Categorize based on message content
            msg = record.getMessage()
            if "v_pec < -500" in msg or "unphysical" in msg.lower():
                self.provenance.log_warning(msg, "data_quality")
            elif "no data" in msg.lower() or "empty cell" in msg.lower():
                self.provenance.log_warning(msg, "boundary")
            elif "non-converge" in msg.lower() or "instability" in msg.lower():
                self.provenance.log_warning(msg, "numerical")
            elif "cosmology" in msg.lower() and ("sigma" in msg.lower() or "omega" in msg.lower()):
                self.provenance.log_warning(msg, "parameter")
            else:
                self.provenance.log_warning(msg, "general")

        return True


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    provenance: Optional[ProvenanceLogger] = None,
) -> logging.Logger:
    """
    Set up structured logging.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str or Path, optional
        Path to log file.
    provenance : ProvenanceLogger, optional
        Provenance logger for warning tracking.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    # Create logger
    logger = logging.getLogger("velocity_reconstruction")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add quality warning filter (if provenance provided)
    if provenance is not None:
        warning_filter = QualityWarningFilter(provenance)
        for handler in logger.handlers:
            handler.addFilter(warning_filter)

    return logger


def log_execution_time(logger: logging.Logger) -> Callable:
    """
    Decorator to log function execution time.

    Parameters
    ----------
    logger : logging.Logger
        Logger for timing messages.

    Returns
    -------
    Callable
        Decorator function.

    Example
    -------
    >>> @log_execution_time(logger)
    ... def slow_function():
    ...     time.sleep(1)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        return wrapper
    return decorator


# Module-level logger instance
logger = logging.getLogger("velocity_reconstruction")


def get_logger(name: str = "velocity_reconstruction") -> logging.Logger:
    """
    Get logger instance.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return logging.getLogger(name)
