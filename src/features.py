"""Feature engineering: per-location history features and same-day regional features.

This stage receives the labelled table from ``data_prep.add_target`` and produces the single
feature table shared by every model, together with a ``FeatureSets`` registry that names the
column lists for the ``base`` and ``regional`` experiments. Every feature uses only values
known at the end of the current day, so no information from tomorrow can leak in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Columns that identify a row or describe tomorrow. They must never be model inputs.
NON_FEATURE_COLUMNS = ("date", "location", "tomorrow_mm")


def days_since_rain(flags: pd.Series) -> pd.Series:
    """Count the days since the last rainy day.

    Args:
        flags: Binary rain-day indicator for one location, in date order.

    Returns:
        A series that is 0 on rainy days and increases by one on every dry day.

    Long dry spells and the first dry day after rain carry different information about
    the persistence of the weather regime, which a simple lag cannot express. Before the
    first rainy day of the record the counter simply counts from the start of the series.
    """
    flags = flags.astype(int)
    # Each rainy day starts a new group; the position inside the group is then the number
    # of dry days that followed it.
    groups = flags.cumsum()
    counter = flags.groupby(groups).cumcount()
    # Rows before the first rain belong to group 0, which did not start on a rainy day, so
    # their count is shifted by one to stay consistent with the definition.
    return counter + (groups == 0).astype(int)


def add_base_features(data: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Add per-location history, change, seasonal and wind features.

    Args:
        data: The labelled table from ``data_prep.add_target``.
        cfg: The main configuration dictionary.

    Returns:
        The table with the new columns (rows with incomplete history removed) and the list
        of base numeric feature names.

    All shifts, rolling windows and differences are computed within each location so that
    one district's history never spills into another's.
    """
    feat_cfg = cfg["features"]
    target = cfg["target"]["name"]
    frame = data.sort_values(["location", "date"]).reset_index(drop=True)
    by_location = frame.groupby("location")

    # Today's rain flag uses the same threshold as the target, so the model sees today's
    # state in exactly the terms it is asked to predict for tomorrow.
    frame["rain_today"] = (frame["PRECTOTCORR"] >= cfg["target"]["rain_day_mm"]).astype(int)

    # Recent daily totals capture short-term persistence of wet or dry spells.
    for lag in feat_cfg["rain_lags"]:
        frame[f"rain_lag{lag}"] = by_location["PRECTOTCORR"].shift(lag)

    # Rolling sums summarise how wet the past weeks were; min_periods=1 avoids discarding
    # the first month of every location for a feature that is meaningful from day one.
    for window in feat_cfg["rain_sum_windows"]:
        frame[f"rain_sum_{window}d"] = by_location["PRECTOTCORR"].transform(
            lambda s, w=window: s.rolling(w, min_periods=1).sum())

    window = feat_cfg["rain_days_window"]
    frame[f"rain_days_{window}d"] = frame.groupby("location")["rain_today"].transform(
        lambda s: s.rolling(window, min_periods=1).sum())
    frame["days_since_rain"] = frame.groupby("location")["rain_today"].transform(days_since_rain)

    # Day-to-day changes in pressure, humidity, temperature and cloud are classic signs of
    # an approaching weather system, which levels alone do not reveal.
    for column in feat_cfg["change_columns"]:
        frame[f"{column}_change"] = by_location[column].diff()

    # Diurnal temperature range and dewpoint depression describe how close the air is to
    # saturation and how cloudy the day was.
    frame["temp_range"] = frame["T2M_MAX"] - frame["T2M_MIN"]
    frame["dewpoint_depression"] = frame["T2M"] - frame["T2MDEW"]

    # Day of year is encoded on a circle so that 31 December and 1 January are neighbours.
    angle = 2 * np.pi * frame["date"].dt.dayofyear / 365.25
    frame["doy_sin"] = np.sin(angle)
    frame["doy_cos"] = np.cos(angle)
    frame["rainy_season"] = frame["date"].dt.month.isin(
        feat_cfg["rainy_season_months"]).astype(int)

    # Wind direction is circular (359 and 1 degree are almost the same), so the raw angle is
    # replaced by its sine and cosine.
    wind_column = feat_cfg["wind_direction_column"]
    radians = np.deg2rad(frame[wind_column])
    frame["wind_dir_sin"] = np.sin(radians)
    frame["wind_dir_cos"] = np.cos(radians)
    frame = frame.drop(columns=wind_column)

    # Lags and differences leave the first days of each location without history; those
    # rows are removed rather than filled, so every remaining value is genuine.
    excluded = {*NON_FEATURE_COLUMNS, target}
    base = [column for column in frame.columns if column not in excluded]
    frame = frame.dropna(subset=base).reset_index(drop=True)
    return frame, base


def add_regional_features(data: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Add same-day rainfall at every location and regional averages.

    Args:
        data: The table returned by ``add_base_features``.
        cfg: The main configuration dictionary.

    Returns:
        The table with regional columns merged on date and the list of new column names.

    Rain-bearing systems move across Rwanda within a day or two, so today's conditions at
    neighbouring sites are informative about tomorrow at a given site. Every value here is
    observed on the current day, so the features remain free of future information.
    """
    # Today's rainfall at every site becomes one column per site, shared by all rows of
    # the same date.
    rain_wide = data.pivot(index="date", columns="location", values="PRECTOTCORR")
    rain_wide.columns = [f"rain_at_{name}" for name in rain_wide.columns]

    # Regional means smooth out local noise and describe the large-scale moisture and
    # pressure pattern of the day.
    regional_columns = cfg["features"]["regional_mean_columns"]
    region_means = data.groupby("date")[regional_columns].mean()
    region_means.columns = [f"region_{column}" for column in regional_columns]
    rain_share = data.groupby("date")["rain_today"].mean().rename("region_rain_share")

    regional = pd.concat([rain_wide, region_means, rain_share], axis=1).reset_index()
    merged = data.merge(regional, on="date", how="left", validate="many_to_one")
    new_columns = [column for column in regional.columns if column != "date"]
    return merged, new_columns


@dataclass(frozen=True)
class FeatureSets:
    """Named feature lists shared by every experiment.

    Attributes:
        base: Numeric features computed from each location's own history (37 by default).
        regional: Base features plus same-day regional features (52 by default).
        categorical: Categorical columns, one-hot encoded or embedded by the models.
    """

    base: list[str]
    regional: list[str]
    categorical: list[str]

    def numeric(self, name: str) -> list[str]:
        """Return the numeric feature list for a feature-set name used in the YAML files."""
        if name not in ("base", "regional"):
            raise KeyError(f"Unknown feature set '{name}'; use 'base' or 'regional'.")
        return list(getattr(self, name))


def target_correlations(frame: pd.DataFrame, features: list[str], target: str,
                        top: int) -> pd.Series:
    """Return the features with the strongest linear correlation with the target.

    Args:
        frame: Rows to analyse (the training period, so the test years stay unseen).
        features: Numeric feature names.
        target: Target column name.
        top: Number of features to return, largest absolute correlation first.

    Returns:
        Signed Pearson correlations indexed by feature name.

    Correlation only captures linear, one-feature-at-a-time relationships, so it is used
    as a first look at the data rather than as a feature selection rule.
    """
    correlations = frame[features].corrwith(frame[target])
    order = correlations.abs().sort_values(ascending=False).index[:top]
    return correlations.loc[order]


def build_features(labelled: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, FeatureSets]:
    """Build the single feature table used by every model.

    Args:
        labelled: The labelled table from ``data_prep.add_target``.
        cfg: The main configuration dictionary.

    Returns:
        The feature table and the ``FeatureSets`` registry.

    Building all features once guarantees that every experiment, traditional or deep,
    sees exactly the same rows, which makes the comparison fair.
    """
    frame, base = add_base_features(labelled, cfg)
    frame, regional_extra = add_regional_features(frame, cfg)
    sets = FeatureSets(base=base, regional=base + regional_extra,
                       categorical=list(cfg["features"]["categorical"]))

    # A final guard makes the no-leakage rule explicit: neither the target nor tomorrow's
    # rainfall may appear in any feature list.
    forbidden = {cfg["target"]["name"], "tomorrow_mm"}
    for name in ("base", "regional"):
        leaked = forbidden.intersection(sets.numeric(name))
        if leaked:
            raise AssertionError(f"Feature set '{name}' contains forbidden columns {leaked}.")
    if frame[sets.regional].isna().any().any():
        raise AssertionError("Feature table contains missing values after construction.")
    return frame, sets
