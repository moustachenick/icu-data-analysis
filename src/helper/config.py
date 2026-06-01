import os
import shutil
import tomllib
from dataclasses import dataclass


# Project root is two levels up from this file (src/helper/config.py -> project root).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.toml")
_EXAMPLE_PATH = os.path.join(_PROJECT_ROOT, "config.example.toml")

_VALID_MODES = {"regression", "classification"}


@dataclass
class AppConfig:
    """Flat, typed view of the settings that used to be interactive prompts."""

    mode: str
    hours: int
    test_size: float
    val_size: float
    add_pathology_one_hot: bool
    drop_high_missing_columns: bool
    impute_missing_values: bool
    drop_lagged_null_rows: bool
    filter_by_pathology: str
    apply_instance_filtering: bool
    run_cross_validation: bool


def load_config(cli_mode=None, cli_hours=None):
    """
    Load the run configuration from ``config.toml`` at the project root.

    If ``config.toml`` does not exist, it is created by copying the tracked
    ``config.example.toml`` template (the program only ever reads config, never writes
    it). CLI overrides take precedence over the file for ``mode`` and ``hours``.

    Args:
        cli_mode (str | None): Value of ``--mode`` if the user passed it, else None.
        cli_hours (int | None): Value of ``--hours`` if the user passed it, else None.

    Returns:
        AppConfig: The resolved configuration.
    """
    if not os.path.exists(_CONFIG_PATH):
        if not os.path.exists(_EXAMPLE_PATH):
            raise FileNotFoundError(
                f"Neither config.toml nor config.example.toml found at {_PROJECT_ROOT}. "
                "Restore config.example.toml from version control."
            )
        shutil.copyfile(_EXAMPLE_PATH, _CONFIG_PATH)
        print(f"No config.toml found — created one from config.example.toml at {_CONFIG_PATH}\n")

    with open(_CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)

    run = data.get("run", {})
    split = data.get("split", {})
    preprocessing = data.get("preprocessing", {})
    parsing = data.get("parsing", {})
    classification = data.get("classification", {})

    mode = cli_mode or run.get("mode", "regression")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Expected one of {sorted(_VALID_MODES)} "
            f"(set [run].mode in config.toml or pass --mode)."
        )

    hours = cli_hours if cli_hours is not None else run.get("hours", 5)

    test_size = split.get("test_size", 0.2)
    val_size = split.get("val_size", 0.2)
    if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
        raise ValueError(
            f"Invalid split sizes (test_size={test_size}, val_size={val_size}). "
            "Each must be in (0, 1) and test_size + val_size must be < 1."
        )

    return AppConfig(
        mode=mode,
        hours=hours,
        test_size=test_size,
        val_size=val_size,
        add_pathology_one_hot=preprocessing.get("add_pathology_one_hot", False),
        drop_high_missing_columns=preprocessing.get("drop_high_missing_columns", True),
        impute_missing_values=preprocessing.get("impute_missing_values", True),
        drop_lagged_null_rows=preprocessing.get("drop_lagged_null_rows", True),
        filter_by_pathology=parsing.get("filter_by_pathology", ""),
        apply_instance_filtering=parsing.get("apply_instance_filtering", True),
        run_cross_validation=classification.get("run_cross_validation", False),
    )
