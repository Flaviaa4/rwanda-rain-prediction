"""Simple reference forecasts that every model must beat.

Baselines answer the question "is the model learning anything beyond the obvious?". They
receive the ``Splits`` object from ``split.split_by_year`` and return predictions for one
partition, which the runner scores with the same metrics as the trained models.
"""

from __future__ import annotations

import numpy as np

from src.split import Splits

BASELINE_KINDS = ("always_rain", "persistence", "climatology")


def predict_baseline(kind: str, splits: Splits, part: str = "val") -> np.ndarray:
    """Predict ``RainTomorrow`` with a rule that needs no training.

    Args:
        kind: ``always_rain``, ``persistence`` or ``climatology``.
        splits: The chronological splits.
        part: The partition to predict (``train``, ``val`` or ``test``).

    Returns:
        An integer array of 0/1 predictions, one per row of the partition.

    ``always_rain`` predicts the majority class and shows what accuracy alone can hide.
    ``persistence`` predicts that tomorrow will be like today, the classic hard-to-beat
    weather baseline. ``climatology`` predicts rain when the training-period rain rate of
    the current calendar month exceeds one half, which captures the seasonal cycle only.
    """
    frame = splits.part(part)
    if kind == "always_rain":
        return np.ones(len(frame), dtype=int)
    if kind == "persistence":
        return frame["rain_today"].to_numpy(dtype=int)
    if kind == "climatology":
        # The monthly rates are learnt from the training period only, so the baseline uses
        # no information from the period it is evaluated on.
        monthly_rate = splits.train.groupby(splits.train["date"].dt.month)[splits.target].mean()
        rate = frame["date"].dt.month.map(monthly_rate).to_numpy()
        return (rate >= 0.5).astype(int)
    raise ValueError(f"Unknown baseline '{kind}'; expected one of {BASELINE_KINDS}.")
