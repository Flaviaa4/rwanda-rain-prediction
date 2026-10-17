"""Every figure used in the notebook and the report.

Plotting is kept separate from computation: each function receives tables or arrays that
other modules produced, draws one labelled figure (title, axes, legend), saves it at 200 dpi
in ``results/figures/`` and returns it for display. Colours are assigned by role and kept
fixed across figures (traditional ML is always blue, deep learning always orange, the
persistence baseline always aqua) so a reader can follow a model from chart to chart. The
palette was chosen to stay distinguishable for the common forms of colour blindness.
"""

from __future__ import annotations

import calendar
import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, roc_curve

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
NEUTRAL = "#a3a29c"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
PARADIGM_COLOURS = {"baseline": NEUTRAL, "ml": SERIES[0], "dl": SERIES[1]}
SEQUENTIAL = "Blues"
DPI = 200

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "font.size": 10,
}


def _styled(function):
    """Draw a figure inside the project style without changing global matplotlib settings."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return function(*args, **kwargs)
    return wrapper


def _finish(fig: plt.Figure, filename: str, figures_dir: Path | None) -> plt.Figure:
    """Tighten the layout and save the figure at the project resolution."""
    fig.tight_layout()
    if figures_dir is not None:
        fig.savefig(Path(figures_dir) / f"{filename}.png", dpi=DPI, bbox_inches="tight")
    return fig


@_styled
def plot_class_balance(frame: pd.DataFrame, target: str,
                       figures_dir: Path | None = None) -> plt.Figure:
    """Bar chart of the number of dry-tomorrow and rain-tomorrow days, with shares."""
    counts = frame[target].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["No rain tomorrow (0)", "Rain tomorrow (1)"], counts.values,
                  color=[NEUTRAL, SERIES[0]], width=0.55)
    for bar, count in zip(bars, counts.values):
        ax.annotate(f"{count:,}\n({count / counts.sum():.1%})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", color=INK, fontsize=9,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_title("Class balance of the target")
    ax.set_ylabel("Number of location-days")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="x", visible=False)
    return _finish(fig, "class_balance", figures_dir)


@_styled
def plot_monthly_rain_rate(frame: pd.DataFrame, target: str, rainy_months: list[int],
                           figures_dir: Path | None = None) -> plt.Figure:
    """Share of days followed by rain for each calendar month, rainy seasons highlighted."""
    rate = frame.groupby(frame["date"].dt.month)[target].mean()
    colours = [SERIES[0] if month in rainy_months else NEUTRAL for month in rate.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([calendar.month_abbr[m] for m in rate.index], rate.values, color=colours, width=0.7)
    ax.axhline(frame[target].mean(), color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.annotate("overall rate", (len(rate) - 0.5, frame[target].mean()), ha="right",
                va="bottom", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Rain-tomorrow rate by calendar month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of days followed by rain")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=SERIES[0], label="Rainy season months"),
                       Patch(color=NEUTRAL, label="Dry season months")], loc="upper right")
    return _finish(fig, "monthly_rain_rate", figures_dir)


@_styled
def plot_rain_rate_by_location(frame: pd.DataFrame, target: str,
                               figures_dir: Path | None = None) -> plt.Figure:
    """Horizontal bar chart of the rain-tomorrow rate at each location, sorted."""
    rate = frame.groupby("location")[target].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(rate.index, rate.values, color=SERIES[0], height=0.6)
    for y, value in enumerate(rate.values):
        ax.annotate(f"{value:.1%}", (value, y), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.set_title("Rain-tomorrow rate by location")
    ax.set_xlabel("Share of days followed by rain")
    ax.set_xlim(0, min(1, rate.max() * 1.15))
    ax.grid(axis="y", visible=False)
    return _finish(fig, "rain_rate_by_location", figures_dir)


@_styled
def plot_top_correlations(correlations: pd.Series,
                          figures_dir: Path | None = None) -> plt.Figure:
    """Horizontal bars of the features most correlated with the target, signed."""
    ordered = correlations.iloc[::-1]
    colours = [SERIES[0] if value >= 0 else SERIES[1] for value in ordered.values]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(ordered) + 1.2))
    ax.barh(ordered.index, ordered.values, color=colours, height=0.65)
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_title("Features most correlated with RainTomorrow (training period)")
    ax.set_xlabel("Pearson correlation with the target")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=SERIES[0], label="Positive"),
                       Patch(color=SERIES[1], label="Negative")], loc="lower right")
    return _finish(fig, "top_correlations", figures_dir)


@_styled
def plot_baselines_vs_models(log_frame: pd.DataFrame, metric: str = "macro_f1",
                             reference_run: str = "baseline_persistence",
                             figures_dir: Path | None = None) -> plt.Figure:
    """Dot plot of the validation score of every run, in the order runs were performed.

    A dot plot is used rather than bars because the models differ by fractions of a point;
    bars starting at zero would make those differences invisible.
    """
    frame = log_frame.iloc[::-1].reset_index(drop=True)
    metric_label = metric.replace("_", " ").replace("f1", "F1").replace("auc", "AUC")
    names = {"baseline": "Baseline", "ml": "Traditional ML", "dl": "Deep learning"}
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(frame) + 2))
    for paradigm, colour in PARADIGM_COLOURS.items():
        rows = frame.loc[frame["paradigm"] == paradigm]
        if rows.empty:
            continue
        ax.scatter(rows[metric], rows.index, color=colour, s=48, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5, label=names[paradigm])
    for y, value in zip(frame.index, frame[metric]):
        ax.annotate(f"{value:.3f}", (value, y), xytext=(7, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8)
    reference = log_frame.loc[log_frame["run"] == reference_run, metric]
    if not reference.empty:
        ax.axvline(reference.iloc[0], color=SERIES[2], linestyle="--", linewidth=1.5,
                   label="Persistence baseline")
    ax.set_yticks(frame.index, frame["run"])
    ax.set_xlim(frame[metric].min() - 0.02, frame[metric].max() + 0.04)
    ax.set_title(f"Validation {metric_label} of every run")
    ax.set_xlabel(f"Validation {metric_label}")
    ax.grid(axis="y", visible=False)
    # The legend sits below the axis so it can never hide a point.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    return _finish(fig, "baselines_vs_models", figures_dir)


@_styled
def plot_ml_learning_curves(curves: pd.DataFrame,
                            figures_dir: Path | None = None) -> plt.Figure:
    """Training and validation macro F1 against training size, one panel per run."""
    runs = list(pd.unique(curves["run"]))
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        rows = curves.loc[curves["run"] == run]
        ax.plot(rows["n_train"], rows["train_macro_f1"], marker="o", color=SERIES[0],
                label="Training")
        ax.plot(rows["n_train"], rows["val_macro_f1"], marker="o", color=SERIES[1],
                label="Validation")
        ax.set_xscale("log")
        ax.set_title(run)
        ax.set_xlabel("Training rows (log scale)")
        ax.set_ylabel("Macro F1")
        ax.legend(loc="lower right")
    fig.suptitle("Learning curves of the tuned scikit-learn models", color=INK,
                 fontweight="bold")
    return _finish(fig, "learning_curves_ml", figures_dir)


@_styled
def plot_dl_history(history: dict, run: str, figures_dir: Path | None = None) -> plt.Figure:
    """Loss, accuracy and AUC per epoch for training and validation, best epoch marked."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, key, label in zip(axes, ("loss", "accuracy", "auc"),
                              ("Binary cross-entropy", "Accuracy", "ROC-AUC")):
        ax.plot(epochs, history[key], color=SERIES[0], label="Training")
        ax.plot(epochs, history[f"val_{key}"], color=SERIES[1], label="Validation")
        ax.axvline(history["best_epoch"], color=INK_SECONDARY, linestyle="--", linewidth=1,
                   label=f"Best epoch ({history['best_epoch']})")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
    fig.suptitle(f"Training history of {run}", color=INK, fontweight="bold")
    return _finish(fig, f"dl_history_{run}", figures_dir)


@_styled
def plot_confusion_matrices(y_true: np.ndarray, predictions: dict[str, np.ndarray],
                            figures_dir: Path | None = None) -> plt.Figure:
    """Side-by-side confusion matrices, shaded by the share of each true class."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(4.8 * len(predictions), 4.2),
                             squeeze=False)
    labels = ["No rain", "Rain"]
    for ax, (name, pred) in zip(axes[0], predictions.items()):
        counts = confusion_matrix(y_true, pred, labels=[0, 1])
        shares = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{counts[i, j]:,}\n{shares[i, j]:.1%}", ha="center",
                        va="center", fontsize=10,
                        color=SURFACE if shares[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the test split (row shares)", color=INK,
                 fontweight="bold")
    return _finish(fig, "confusion_matrices", figures_dir)


@_styled
def plot_roc_curves(y_true: np.ndarray, scores: dict[str, np.ndarray],
                    points: dict[str, np.ndarray] | None = None,
                    figures_dir: Path | None = None) -> plt.Figure:
    """ROC curves of scored models plus single operating points for hard-rule baselines."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for colour, (name, model_scores) in zip(SERIES, scores.items()):
        fpr, tpr, _ = roc_curve(y_true, model_scores)
        ax.plot(fpr, tpr, color=colour, label=name)
    # A rule such as persistence has no score to vary, so it is a single point on the plot.
    for name, pred in (points or {}).items():
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        ax.scatter(fp / (fp + tn), tp / (tp + fn), color=SERIES[2], s=70, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5, label=name)
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle=":", linewidth=1, label="Chance")
    ax.set_title("ROC curves on the test split")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    return _finish(fig, "roc_curves", figures_dir)


@_styled
def plot_error_by_bucket(table: pd.DataFrame, models: list[str],
                         figures_dir: Path | None = None) -> plt.Figure:
    """Grouped bars of the error rate by tomorrow's rainfall amount."""
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(table))
    width = 0.8 / len(models)
    for k, (colour, model) in enumerate(zip(SERIES, models)):
        ax.bar(positions + (k - (len(models) - 1) / 2) * width, table[model], width * 0.92,
               color=colour, label=model)
    ax.set_xticks(positions, [f"{bucket}\n(n={days:,})"
                              for bucket, days in zip(table.index, table["days"])])
    ax.set_title("Test error rate by tomorrow's rainfall")
    ax.set_xlabel("Tomorrow's rainfall (mm)")
    ax.set_ylabel("Error rate")
    ax.grid(axis="x", visible=False)
    ax.legend()
    return _finish(fig, "error_by_bucket", figures_dir)


@_styled
def plot_error_by_month(table: pd.DataFrame, models: list[str],
                        figures_dir: Path | None = None) -> plt.Figure:
    """Error rate per calendar month for each model."""
    fig, ax = plt.subplots(figsize=(8, 4))
    months = [calendar.month_abbr[m] for m in table.index]
    for colour, model in zip(SERIES, models):
        ax.plot(months, table[model], marker="o", color=colour, label=model)
    ax.set_title("Test error rate by calendar month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Error rate")
    ax.legend()
    return _finish(fig, "error_by_month", figures_dir)


@_styled
def plot_feature_importance(table: pd.DataFrame, value_column: str, title: str,
                            filename: str, label_column: str = "feature",
                            xlabel: str | None = None,
                            figures_dir: Path | None = None) -> plt.Figure:
    """Horizontal bars of feature importance; signed values are coloured by direction."""
    ordered = table.iloc[::-1]
    values = ordered[value_column].to_numpy()
    signed = (values < 0).any()
    colours = [SERIES[0] if v >= 0 else SERIES[1] for v in values] if signed else SERIES[0]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(ordered) + 1.2))
    ax.barh(ordered[label_column], values, color=colours, height=0.65)
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel or value_column.replace("_", " "))
    ax.grid(axis="y", visible=False)
    if signed:
        ax.legend(handles=[Patch(color=SERIES[0], label="Raises rain probability"),
                           Patch(color=SERIES[1], label="Lowers rain probability")],
                  loc="lower right")
    return _finish(fig, filename, figures_dir)
