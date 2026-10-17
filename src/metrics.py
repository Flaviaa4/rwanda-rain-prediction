"""Metric computation and the shared results log.

Every baseline, scikit-learn run and Keras run is scored by ``compute_metrics`` and recorded
in one ``ResultsLog``, so the two paradigms are compared on identical terms. The log is
saved to ``results/experiments.csv`` after every run, which is also how the runner knows
which experiments are already finished after a Colab disconnect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

LOG_COLUMNS = ["run", "paradigm", "feature_set", "accuracy", "precision", "recall",
               "macro_f1", "roc_auc", "train_acc", "gap", "seconds", "epochs", "notes"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    scores: np.ndarray | None = None) -> dict[str, float]:
    """Compute the evaluation metrics used throughout the project.

    Args:
        y_true: True labels (0/1).
        y_pred: Predicted labels (0/1).
        scores: Optional continuous scores (probabilities or decision values) for ROC-AUC.

    Returns:
        A dictionary with accuracy, precision, recall, macro F1 and ROC-AUC.

    Precision and recall refer to the rain class, the one farmers act on. Macro F1 is the
    primary metric because about 59% of days are followed by rain: averaging the F1 of both
    classes penalises a model that neglects dry days, which accuracy would not.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": np.nan,
    }
    # ROC-AUC needs a ranking score; hard 0/1 rules such as persistence have none.
    if scores is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
    return {name: float(value) for name, value in metrics.items()}


class ResultsLog:
    """A table of validation results, one row per experiment.

    Re-adding a run replaces its previous row, so retraining never creates duplicates.
    """

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        """Create a log, optionally from an existing table."""
        if frame is None:
            frame = pd.DataFrame(columns=LOG_COLUMNS)
        self._frame = frame.reindex(columns=LOG_COLUMNS)

    def add(self, run: str, paradigm: str, metrics: dict[str, float], train_acc: float,
            seconds: float, notes: str = "", feature_set: str = "",
            epochs: int | None = None) -> None:
        """Record one experiment.

        Args:
            run: Unique run name from ``experiments.yaml``.
            paradigm: ``baseline``, ``ml`` or ``dl``.
            metrics: Validation metrics from ``compute_metrics``.
            train_acc: Accuracy on the training split, used to measure overfitting.
            seconds: Training time in seconds.
            notes: Short description of what changed in this run.
            feature_set: ``base`` or ``regional``.
            epochs: Number of epochs trained (deep learning only).

        The ``gap`` column (training minus validation accuracy) is a simple overfitting
        indicator: a large gap means the model memorised training days.
        """
        row = {"run": run, "paradigm": paradigm, "feature_set": feature_set, **metrics,
               "train_acc": train_acc, "gap": train_acc - metrics["accuracy"],
               "seconds": seconds, "epochs": epochs, "notes": notes}
        kept = self._frame.loc[self._frame["run"] != run]
        new_row = pd.DataFrame([row], columns=LOG_COLUMNS)
        self._frame = new_row if kept.empty else pd.concat([kept, new_row], ignore_index=True)

    def has(self, run: str) -> bool:
        """Return True when the run is already recorded."""
        return bool((self._frame["run"] == run).any())

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the full log."""
        return self._frame.copy().reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        """Write the log to CSV."""
        self._frame.to_csv(path, index=False)

    @classmethod
    def load(cls, path: str | Path) -> ResultsLog:
        """Read a log from CSV, or return an empty log if the file does not exist."""
        path = Path(path)
        if not path.exists():
            return cls()
        return cls(pd.read_csv(path))

    def report_table(self) -> pd.DataFrame:
        """Return the log formatted for the written report.

        Accuracy, precision and recall are shown as percentages; macro F1, ROC-AUC and the
        gap keep three decimals because differences between models are small.
        """
        frame = self.to_frame()
        table = pd.DataFrame({
            "Run": frame["run"],
            "Paradigm": frame["paradigm"],
            "Features": frame["feature_set"].fillna(""),
            "Accuracy (%)": (frame["accuracy"] * 100).round(1),
            "Precision (%)": (frame["precision"] * 100).round(1),
            "Recall (%)": (frame["recall"] * 100).round(1),
            "Macro F1": frame["macro_f1"].round(3),
            "ROC-AUC": frame["roc_auc"].round(3),
            "Gap": frame["gap"].round(3),
            "Time (s)": frame["seconds"].round(1),
            "Notes": frame["notes"].fillna(""),
        })
        return table.set_index("Run")
