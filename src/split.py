"""Chronological train, validation and test split.

This stage receives the feature table from ``features.build_features`` and cuts it by date
into three consecutive periods. Training, validation and test therefore represent the past,
the recent past and the "future" respectively, which mirrors how the model would be used.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Splits:
    """The three chronological partitions of the feature table.

    Attributes:
        train: Rows up to and including ``split.train_end``.
        val: Rows after ``train_end`` up to and including ``split.val_end``.
        test: Rows after ``val_end``; used once, in the final evaluation.
        target: Name of the target column, used by ``summary``.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target: str = "RainTomorrow"

    def part(self, name: str) -> pd.DataFrame:
        """Return one partition by name (``train``, ``val`` or ``test``)."""
        if name not in ("train", "val", "test"):
            raise KeyError(f"Unknown split '{name}'.")
        return getattr(self, name)

    def summary(self) -> pd.DataFrame:
        """Return rows, date range and rain share for each partition."""
        rows = []
        for name in ("train", "val", "test"):
            frame = self.part(name)
            rows.append({"split": name, "rows": len(frame),
                         "start": frame["date"].min().date(),
                         "end": frame["date"].max().date(),
                         "rain_share": round(frame[self.target].mean(), 3)})
        return pd.DataFrame(rows).set_index("split")


def split_by_year(data: pd.DataFrame, cfg: dict) -> Splits:
    """Split the feature table into consecutive time periods.

    Args:
        data: The feature table from ``features.build_features``.
        cfg: The main configuration dictionary (``split`` section is used).

    Returns:
        A ``Splits`` instance.

    The split is chronological rather than random because consecutive days are highly
    correlated and all locations share the same weather systems on a given day. A random
    split would place near-copies of test days (the day before, or the same day at another
    location) in the training set, inflating every score and hiding how the model would
    perform on genuinely unseen future weather.
    """
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])

    # Each partition is copied and re-indexed so that later stages can attach predictions
    # by position without worrying about the original index.
    train = data.loc[data["date"] <= train_end].reset_index(drop=True)
    val = data.loc[(data["date"] > train_end) & (data["date"] <= val_end)].reset_index(drop=True)
    test = data.loc[data["date"] > val_end].reset_index(drop=True)
    return Splits(train=train, val=val, test=test, target=cfg["target"]["name"])
