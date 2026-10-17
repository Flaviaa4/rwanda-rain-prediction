"""Configuration loading, project paths and reproducibility helpers.

This module is the entry point of the pipeline. It reads the two YAML files that drive
every other stage, decides where data, models and results live (Google Drive when running
in Colab, a local folder otherwise) and provides the single seeding function that is called
before any random operation. Keeping these concerns in one place means no other module
needs to know whether it runs in Colab or on a laptop.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_file(path: str | Path) -> Path:
    """Locate a configuration file relative to the working directory or the repository.

    Args:
        path: Absolute path, or a path relative to the current directory or the repo root.

    Returns:
        The existing file path.

    The notebooks may be started from the repository root (Colab) or from the
    ``notebooks/`` folder (Jupyter), so relative paths are tried against both locations.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPO_ROOT / candidate


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    """Load the main project configuration.

    Args:
        path: Location of ``config.yaml``.

    Returns:
        The configuration as a nested dictionary.
    """
    with open(_resolve_file(path), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiments(path: str | Path = "configs/experiments.yaml") -> dict:
    """Load the experiment registry (baselines, ML runs, DL defaults and DL runs).

    Args:
        path: Location of ``experiments.yaml``.

    Returns:
        A dictionary with the keys ``baselines``, ``ml``, ``dl_defaults`` and ``dl``.
    """
    with open(_resolve_file(path), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def in_colab() -> bool:
    """Report whether the code is running inside Google Colab.

    Returns:
        True when the ``google.colab`` module is importable.
    """
    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class ProjectPaths:
    """Every filesystem location used at runtime.

    The dataclass is frozen so that no stage can silently redirect outputs after the paths
    have been resolved once at the start of a notebook.
    """

    project_dir: Path
    raw_dir: Path
    combined_csv: Path
    raw_zip: Path
    models_dir: Path
    experiment_models_dir: Path
    results_dir: Path
    histories_dir: Path
    figures_dir: Path


def resolve_paths(cfg: dict) -> ProjectPaths:
    """Decide where the project data lives and create the folder structure.

    Args:
        cfg: The main configuration dictionary.

    Returns:
        A ``ProjectPaths`` instance whose directories all exist.

    Precedence is: the ``PROJECT_DIR`` environment variable, then the Drive folder when in
    Colab, then the local folder. Drive is preferred in Colab because files written there
    survive runtime disconnects and can be shared through a single link.
    """
    path_cfg = cfg["paths"]

    # The environment variable wins so that a local run can point at any folder without
    # editing the configuration file.
    if os.environ.get("PROJECT_DIR"):
        project_dir = Path(os.environ["PROJECT_DIR"])
    elif in_colab():
        project_dir = Path(path_cfg["drive_project_dir"])
    else:
        local = Path(path_cfg["local_project_dir"])
        project_dir = local if local.is_absolute() else REPO_ROOT / local

    models_dir = project_dir / path_cfg["models_subdir"]
    results_dir = project_dir / path_cfg["results_subdir"]
    paths = ProjectPaths(
        project_dir=project_dir,
        raw_dir=project_dir / path_cfg["raw_subdir"],
        combined_csv=project_dir / path_cfg["combined_csv"],
        raw_zip=project_dir / f"{path_cfg['raw_zip_name']}.zip",
        models_dir=models_dir,
        experiment_models_dir=models_dir / "experiments",
        results_dir=results_dir,
        histories_dir=results_dir / "histories",
        figures_dir=project_dir / path_cfg["figures_subdir"],
    )

    # Creating every directory up front means later stages can write files without
    # checking for missing folders.
    for directory in (paths.project_dir, paths.raw_dir, paths.models_dir,
                      paths.experiment_models_dir, paths.results_dir,
                      paths.histories_dir, paths.figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def set_global_seed(seed: int) -> None:
    """Seed every random number generator used by the project.

    Args:
        seed: The project seed from ``config.yaml``.

    Python, NumPy and (when installed) TensorFlow/Keras are seeded together so that data
    shuffling, weight initialisation and dropout masks are reproducible. TensorFlow is
    imported lazily so the traditional ML part of the pipeline also runs without it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Keras offers one call that seeds Python, NumPy and TensorFlow at once; it is used
    # only if TensorFlow is available in the current environment.
    try:
        from tensorflow import keras
    except ImportError:
        return
    keras.utils.set_random_seed(seed)
