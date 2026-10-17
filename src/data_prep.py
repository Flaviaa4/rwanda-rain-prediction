"""Data loading, cleaning, validation and target creation (the data contract).

This stage turns the raw NASA POWER table into a clean, validated daily panel (one row per
location per day) and attaches the binary target ``RainTomorrow``. Every later stage relies
on the guarantees established here: no duplicate grid cells, a continuous daily sequence
per location, no missing values, and a target that never crosses location boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import ProjectPaths
from src.data_download import assemble_raw, drop_duplicate_grid_cells


def load_raw(paths: ProjectPaths, cfg: dict | None = None) -> pd.DataFrame:
    """Load the combined raw table, rebuilding it from the per-location files if needed.

    Args:
        paths: Resolved project paths.
        cfg: The main configuration; only required when the combined CSV must be rebuilt.

    Returns:
        The raw table with ``date`` parsed as datetime.
    """
    if paths.combined_csv.exists():
        return pd.read_csv(paths.combined_csv, parse_dates=["date"])

    # The combined file is a convenience; the per-location files are the true source, so
    # the table can always be rebuilt from them without downloading again.
    if cfg is None:
        raise FileNotFoundError(
            f"{paths.combined_csv} not found. Pass cfg to rebuild it from {paths.raw_dir}, "
            "or run notebook 01 to download the data.")
    return assemble_raw(paths, cfg)


@dataclass
class CleaningReport:
    """A record of what each cleaning step did, for display in the notebook.

    Attributes:
        steps: One entry per step with the rows before and after and a short note.
        dropped_locations: Mapping of removed locations to the location they duplicated.
        missing_before: Missing values per column before cleaning.
        missing_after: Missing values per column after cleaning.
        date_range: First and last date kept.
    """

    steps: list[dict] = field(default_factory=list)
    dropped_locations: dict[str, str] = field(default_factory=dict)
    missing_before: pd.Series | None = None
    missing_after: pd.Series | None = None
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None

    def log(self, step: str, before: int, after: int, note: str = "") -> None:
        """Append one cleaning step to the report."""
        self.steps.append({"step": step, "rows_before": before, "rows_after": after,
                           "rows_removed": before - after, "note": note})

    def to_frame(self) -> pd.DataFrame:
        """Return the steps as a table, one row per cleaning step."""
        return pd.DataFrame(self.steps)

    def missing_frame(self) -> pd.DataFrame:
        """Return missing-value counts and shares before and after cleaning.

        Only columns that had missing values at some point are shown, which keeps the table
        focused on the variables that actually needed attention.
        """
        table = pd.DataFrame({"missing_before": self.missing_before,
                              "missing_after": self.missing_after}).fillna(0).astype(int)
        table = table[(table["missing_before"] > 0) | (table["missing_after"] > 0)]
        return table


def clean(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean and validate the raw table.

    Args:
        raw: The raw table from ``load_raw``.
        cfg: The main configuration dictionary.

    Returns:
        The clean table and a ``CleaningReport`` describing each step.

    Raises:
        ValueError: If a location has duplicate dates or gaps in its daily sequence.

    The steps run in a fixed order: sentinel replacement, duplicate grid-cell removal, date
    filtering, ordering and continuity checks, and a final missing-value check. Filtering to
    1984 onwards removes the only missing values in the dataset, so no imputation is needed.
    """
    report = CleaningReport()
    parameters = cfg["nasa_power"]["parameters"]
    data = raw.copy()

    # Step 1. The sentinel replacement is repeated defensively because a raw file could
    # have been produced by another script that kept NASA's missing-value code.
    rows = len(data)
    data[parameters] = data[parameters].replace(cfg["nasa_power"]["missing_value"], np.nan)
    report.missing_before = data.isna().sum()
    report.log("Replace missing-value code with NaN", rows, len(data),
               f"code {cfg['nasa_power']['missing_value']}")

    # Step 2. The duplicate grid-cell check is also repeated: if it was skipped upstream,
    # identical series would leak information between the training and test periods.
    rows = len(data)
    data, dropped = drop_duplicate_grid_cells(data)
    report.dropped_locations = dropped
    note = ", ".join(f"{a} duplicates {b}" for a, b in dropped.items()) or "none found"
    report.log("Drop duplicate NASA grid cells", rows, len(data), note)

    # Step 3. Solar radiation and cloud cover only exist from 1984, so starting the analysis
    # there removes every missing value without inventing data through imputation.
    rows = len(data)
    start = pd.Timestamp(cfg["cleaning"]["analysis_start"])
    data = data.loc[data["date"] >= start]
    report.log("Keep dates from analysis start", rows, len(data), f"from {start.date()}")

    # Step 4. Lag and rolling features assume consecutive rows are consecutive days, so any
    # duplicate date or gap would silently corrupt them; the pipeline stops instead.
    rows = len(data)
    data = data.sort_values(["location", "date"]).reset_index(drop=True)
    duplicated = data.duplicated(["location", "date"])
    if duplicated.any():
        raise ValueError(f"{duplicated.sum()} duplicate (location, date) rows found.")
    gaps = data.groupby("location")["date"].diff().dropna() != pd.Timedelta(days=1)
    if gaps.any():
        bad = data.loc[gaps[gaps].index, ["location", "date"]].head()
        raise ValueError(f"Daily sequence has gaps, for example:\n{bad}")
    report.log("Sort and verify continuous daily sequence", rows, len(data),
               "no gaps or duplicate dates")

    # Step 5. Any remaining missing value is reported rather than silently dropped, because
    # it would indicate that the data source changed and the cleaning rules need review.
    report.missing_after = data.isna().sum()
    remaining = int(report.missing_after.sum())
    if remaining:
        print(f"Warning: {remaining} missing values remain:\n"
              f"{report.missing_after[report.missing_after > 0]}")
    report.log("Verify no missing values remain", len(data), len(data),
               f"{remaining} missing values")

    report.date_range = (data["date"].min(), data["date"].max())
    return data, report


def add_target(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach the binary target ``RainTomorrow`` to every row.

    Args:
        data: The clean table from ``clean``.
        cfg: The main configuration dictionary (``target`` section is used).

    Returns:
        The table with ``tomorrow_mm`` (next day's rainfall, kept for error analysis only)
        and the integer target column, without the last day of each location.

    A day counts as rainy when corrected precipitation reaches the configured threshold
    (1 mm by default), a common agronomic definition that ignores trace amounts.
    """
    target = cfg["target"]["name"]
    labelled = data.copy()

    # The target is derived per location so that the last day of one district is never
    # paired with the first day of the next district, which would create false "tomorrows".
    labelled["tomorrow_mm"] = labelled.groupby("location")["PRECTOTCORR"].shift(-1)

    # The final day of each location has no observed tomorrow and cannot be labelled.
    labelled = labelled.dropna(subset=["tomorrow_mm"]).reset_index(drop=True)
    labelled[target] = (labelled["tomorrow_mm"] >= cfg["target"]["rain_day_mm"]).astype(int)
    return labelled
