"""
Reproducibility utilities for consistent experiment results.

Provides:
- Random seed setting for all frameworks
- Deterministic mode configuration
- Experiment metadata tracking
"""

import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .logging_config import get_logger

logger = get_logger("reproducibility")

# Global flag to track if seeds have been set
_seeds_set = False


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all frameworks.

    Args:
        seed: Random seed value
        deterministic: Enable deterministic mode for PyTorch
    """
    global _seeds_set

    logger.info(f"Setting random seed: {seed}")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            if deterministic:
                # Enable deterministic algorithms
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                # For PyTorch >= 1.8
                if hasattr(torch, "use_deterministic_algorithms"):
                    try:
                        torch.use_deterministic_algorithms(True)
                    except RuntimeError:
                        # Some operations don't have deterministic implementations
                        logger.warning("Could not enable all deterministic algorithms")

        logger.debug(f"PyTorch seed set: {seed}")

    except ImportError:
        logger.debug("PyTorch not available, skipping torch seed")

    # TensorFlow (if available)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        logger.debug(f"TensorFlow seed set: {seed}")

    except ImportError:
        logger.debug("TensorFlow not available, skipping tf seed")

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    _seeds_set = True
    logger.info(f"All random seeds set to {seed}")


def get_experiment_metadata(
    experiment_name: str,
    config: Optional[Any] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Collect metadata for experiment tracking.

    Args:
        experiment_name: Name of the experiment
        config: Configuration object (will be serialized)
        extra_info: Additional metadata to include

    Returns:
        Metadata dictionary
    """
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "timestamp_utc": datetime.utcnow().isoformat(),
    }

    # System info
    import platform

    metadata["system"] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }

    # PyTorch info
    try:
        import torch

        metadata["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if torch.cuda.is_available():
            metadata["torch"]["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]

    except ImportError:
        pass

    # PyTorch Geometric info
    try:
        import torch_geometric

        metadata["torch_geometric"] = {
            "version": torch_geometric.__version__,
        }
    except ImportError:
        pass

    # Transformers info
    try:
        import transformers

        metadata["transformers"] = {
            "version": transformers.__version__,
        }
    except ImportError:
        pass

    # Configuration
    if config is not None:
        if hasattr(config, "to_dict"):
            metadata["config"] = config.to_dict()
        elif hasattr(config, "__dict__"):
            metadata["config"] = {
                k: v for k, v in config.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }

    # Extra info
    if extra_info:
        metadata["extra"] = extra_info

    return metadata


def save_experiment_metadata(
    metadata: Dict[str, Any],
    path: str,
    format: str = "json",
) -> None:
    """
    Save experiment metadata to file.

    Args:
        metadata: Metadata dictionary
        path: Output file path
        format: Output format ('json' or 'yaml')
    """
    import json
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    elif format == "yaml":
        try:
            import yaml

            with open(output_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False)
        except ImportError:
            logger.warning("PyYAML not installed, falling back to JSON")
            with open(output_path.with_suffix(".json"), "w") as f:
                json.dump(metadata, f, indent=2, default=str)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Experiment metadata saved to {output_path}")


def get_git_info() -> Optional[Dict[str, str]]:
    """
    Get current git commit info for experiment tracking.

    Returns:
        Dictionary with git info or None if not in git repo
    """
    import subprocess

    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        commit_short = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        return {
            "commit_hash": commit_hash,
            "commit_short": commit_short,
            "branch": branch,
            "has_uncommitted_changes": bool(status),
        }

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class ExperimentTracker:
    """
    Track experiment progress and metrics.

    Usage:
        tracker = ExperimentTracker("my_experiment")
        tracker.start()

        for i, sample in enumerate(samples):
            # Process sample
            tracker.log_metric("accuracy", accuracy)
            tracker.checkpoint(i)

        tracker.end()
        tracker.save()
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        output_dir: str = "results/experiments",
    ):
        self.name = name
        self.config = config
        self.output_dir = output_dir

        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.metrics: Dict[str, list] = {}
        self.checkpoints: list = []
        self.metadata: Dict[str, Any] = {}

    def start(self) -> "ExperimentTracker":
        """Start experiment tracking."""
        self.start_time = datetime.now()
        self.metadata = get_experiment_metadata(self.name, self.config)
        self.metadata["git"] = get_git_info()

        logger.info(f"Experiment '{self.name}' started at {self.start_time.isoformat()}")
        return self

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            "value": value,
            "step": step or len(self.metrics[name]),
            "timestamp": datetime.now().isoformat(),
        })

    def checkpoint(self, step: int, data: Optional[Dict[str, Any]] = None) -> None:
        """Record a checkpoint."""
        self.checkpoints.append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        })

    def end(self) -> None:
        """End experiment tracking."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0

        self.metadata["end_time"] = self.end_time.isoformat()
        self.metadata["duration_seconds"] = duration

        logger.info(f"Experiment '{self.name}' ended. Duration: {duration:.2f}s")

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        summary = {
            "name": self.name,
            "metadata": self.metadata,
            "metrics": {},
            "checkpoints_count": len(self.checkpoints),
        }

        # Compute metric summaries
        for name, values in self.metrics.items():
            vals = [v["value"] for v in values]
            summary["metrics"][name] = {
                "count": len(vals),
                "mean": np.mean(vals) if vals else None,
                "std": np.std(vals) if vals else None,
                "min": np.min(vals) if vals else None,
                "max": np.max(vals) if vals else None,
            }

        return summary

    def save(self, path: Optional[str] = None) -> str:
        """Save experiment results."""
        import json
        from pathlib import Path

        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{self.output_dir}/{timestamp}_{self.name}.json"

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "summary": self.get_summary(),
            "metrics": self.metrics,
            "checkpoints": self.checkpoints,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Experiment results saved to {output_path}")
        return str(output_path)
