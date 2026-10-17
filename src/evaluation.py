"""Model selection, final test evaluation, learning curves and error analysis.

This is the last computational stage. It reads the results log produced by the runner,
selects the best model of each paradigm on validation macro F1, evaluates the two chosen
models and the persistence baseline on the test split exactly once, and provides the
analyses used to interpret them: learning curves, error breakdowns, logistic regression
coefficients, permutation importance for the network, and a side-by-side paradigm summary.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.baselines import predict_baseline
from src.config import ProjectPaths
from src.features import FeatureSets
from src.metrics import ResultsLog, compute_metrics
from src.ml import build_pipeline, decision_scores
from src.split import Splits

FINAL_ML_NAME = "best_ml_model.joblib"
FINAL_DL_NAME = "best_dl_model"


def _artifact(run: str, paths: ProjectPaths) -> Path | None:
    """Return the saved model file of a run, or None if the run saved no model."""
    for suffix in (".joblib", ".keras"):
        candidate = paths.experiment_models_dir / f"{run}{suffix}"
        if candidate.exists():
            return candidate
    return None


def rank_runs(log: ResultsLog, paradigm: str, metric: str = "macro_f1",
              decimals: int = 4) -> pd.DataFrame:
    """Order the runs of one paradigm by the selection rule.

    Args:
        log: The results log.
        paradigm: ``ml`` or ``dl``.
        metric: Validation metric to maximise.
        decimals: Scores are compared after rounding, so differences smaller than this are
            treated as ties.

    Returns:
        The runs sorted best first: highest rounded metric, then shortest training time.

    Rounding before comparing avoids preferring a model because of a difference far below
    run-to-run noise; among equally good models the faster one is preferred because it is
    cheaper to retrain and deploy.
    """
    frame = log.to_frame()
    frame = frame.loc[frame["paradigm"] == paradigm].copy()
    frame["score"] = frame[metric].astype(float).round(decimals)
    return frame.sort_values(["score", "seconds"], ascending=[False, True]).reset_index(drop=True)


def select_best(log: ResultsLog, paradigm: str, metric: str = "macro_f1",
                paths: ProjectPaths | None = None) -> str:
    """Return the name of the best run of a paradigm.

    Args:
        log: The results log.
        paradigm: ``ml`` or ``dl``.
        metric: Validation metric to maximise (macro F1 by default).
        paths: If given, runs without a saved model are skipped, since they cannot be
            deployed or evaluated on the test split.

    Returns:
        The selected run name.
    """
    ranked = rank_runs(log, paradigm, metric)
    for run in ranked["run"]:
        if paths is None or _artifact(run, paths) is not None:
            return run
        print(f"{run} skipped: no saved model")
    raise ValueError(f"No selectable {paradigm} run found in the log.")


def load_model(run: str, paths: ProjectPaths) -> Any:
    """Load the saved pipeline or Keras model of an experiment run."""
    artifact = _artifact(run, paths)
    if artifact is None:
        raise FileNotFoundError(f"No saved model for run '{run}'.")
    if artifact.suffix == ".joblib":
        return joblib.load(artifact)
    from tensorflow import keras
    return keras.models.load_model(artifact)


def save_final_models(best_ml: str, best_dl: str, paths: ProjectPaths) -> dict[str, Path]:
    """Store the two selected models under their final, submission-ready names.

    Args:
        best_ml: Name of the selected scikit-learn run.
        best_dl: Name of the selected Keras run.
        paths: Resolved project paths.

    Returns:
        The paths of the saved files.

    The Keras model is saved both in the native ``.keras`` format and in the legacy HDF5
    format (``.h5``) requested by the brief, so it can be opened with older tools.
    """
    from tensorflow import keras

    ml_target = paths.models_dir / FINAL_ML_NAME
    shutil.copyfile(_artifact(best_ml, paths), ml_target)
    dl_model = keras.models.load_model(_artifact(best_dl, paths))
    keras_target = paths.models_dir / f"{FINAL_DL_NAME}.keras"
    h5_target = paths.models_dir / f"{FINAL_DL_NAME}.h5"
    dl_model.save(keras_target)
    dl_model.save(h5_target)
    return {"ml": ml_target, "dl_keras": keras_target, "dl_h5": h5_target}


@dataclass
class TestResults:
    """Everything produced by the final test evaluation.

    Attributes:
        metrics: One row per model with the test metrics.
        y_true: True test labels.
        predictions: Predicted labels per model name.
        scores: Continuous scores per model name (None for the persistence rule).
    """

    metrics: pd.DataFrame
    y_true: np.ndarray
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    scores: dict[str, np.ndarray | None] = field(default_factory=dict)


def final_test_evaluation(best_ml: str, best_dl: str, splits: Splits, dl_data: Any,
                          paths: ProjectPaths, threshold: float,
                          force: bool = False) -> TestResults:
    """Evaluate the two final models and the persistence baseline on the test split.

    Args:
        best_ml: Name of the selected scikit-learn run.
        best_dl: Name of the selected Keras run.
        splits: The chronological splits.
        dl_data: The ``dl.DLData`` instance used for training.
        paths: Resolved project paths.
        threshold: Probability above which the network predicts rain.
        force: Overwrite an existing ``final_test_metrics.csv``.

    Returns:
        A ``TestResults`` with the metrics table, predictions and scores.

    This is the only place in the project where the test split is scored. The saved final
    models are used (not the in-memory ones), which also verifies the submitted files.
    If the metrics file already exists it is not overwritten unless ``force=True``; the
    predictions are still recomputed for the plots, and any difference from the stored
    metrics is reported, because it would mean a model changed after the test was run.
    """
    from tensorflow import keras

    from src.dl import layout_of, predict_scores

    test = splits.test
    y_true = test[splits.target].to_numpy(dtype=int)
    results = TestResults(metrics=pd.DataFrame(), y_true=y_true)

    # The scikit-learn pipeline selects its own input columns by name.
    ml_model = joblib.load(paths.models_dir / FINAL_ML_NAME)
    X_test = test[list(ml_model.feature_names_in_)]
    results.predictions["best_ml"] = ml_model.predict(X_test)
    results.scores["best_ml"] = decision_scores(ml_model, X_test)

    dl_model = keras.models.load_model(paths.models_dir / f"{FINAL_DL_NAME}.keras")
    dl_scores = predict_scores(dl_model, dl_data.inputs("test", layout_of(dl_model)))
    results.predictions["best_dl"] = (dl_scores >= threshold).astype(int)
    results.scores["best_dl"] = dl_scores

    results.predictions["persistence"] = predict_baseline("persistence", splits, "test")
    results.scores["persistence"] = None

    runs = {"best_ml": best_ml, "best_dl": best_dl, "persistence": "baseline_persistence"}
    rows = [{"model": name, "run": runs[name],
             **compute_metrics(y_true, results.predictions[name], results.scores[name])}
            for name in runs]
    results.metrics = pd.DataFrame(rows).set_index("model")

    # The stored test metrics are the official result; they are protected from being
    # silently replaced by a later, possibly retuned, model.
    target = paths.results_dir / "final_test_metrics.csv"
    if target.exists() and not force:
        stored = pd.read_csv(target, index_col="model")
        stored_f1 = stored["macro_f1"].reindex(results.metrics.index)
        changed = not np.allclose(stored_f1, results.metrics["macro_f1"], atol=1e-6)
        print(f"{target.name} already exists and was not overwritten"
              + (" (WARNING: current models give different test scores)" if changed else ""))
    else:
        results.metrics.to_csv(target)
    return results


def ml_learning_curve(spec: dict, splits: Splits, feature_sets: FeatureSets,
                      paths: ProjectPaths, fractions: tuple[float, ...] | list[float],
                      best_params: dict | None = None, seed: int = 0,
                      force: bool = False) -> pd.DataFrame:
    """Measure training and validation macro F1 as the training set grows.

    Args:
        spec: One ML entry from ``experiments.yaml``.
        splits: The chronological splits.
        feature_sets: The feature registry.
        paths: Resolved project paths (results are cached in ``results/``).
        fractions: Shares of the training split to train on.
        best_params: Tuned hyperparameters for this run, if any.
        seed: Seed for drawing the random training subsets.
        force: Recompute even if a cached curve exists for this run.

    Returns:
        One row per fraction with the training size and both scores.

    A large and persistent gap between the two curves indicates variance (overfitting);
    two curves that converge at a modest score indicate bias, in which case more data would
    help little and better features or a more flexible model would be needed.
    """
    cache = paths.results_dir / "learning_curves_ml.csv"
    cached = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    if not force and not cached.empty and (cached["run"] == spec["name"]).any():
        return cached.loc[cached["run"] == spec["name"]].reset_index(drop=True)

    target = splits.target
    rng = np.random.default_rng(seed)
    rows = []
    for fraction in fractions:
        # Random rows (not the earliest years) are drawn so that every subset covers the
        # whole training period and differs only in size.
        size = int(round(fraction * len(splits.train)))
        subset = splits.train.iloc[np.sort(rng.choice(len(splits.train), size, replace=False))]
        pipeline, columns = build_pipeline(spec, feature_sets, best_params)
        start = time.perf_counter()
        pipeline.fit(subset[columns], subset[target])
        rows.append({
            "run": spec["name"], "fraction": fraction, "n_train": size,
            "train_macro_f1": f1_score(subset[target], pipeline.predict(subset[columns]),
                                       average="macro"),
            "val_macro_f1": f1_score(splits.val[target], pipeline.predict(splits.val[columns]),
                                     average="macro"),
            "seconds": time.perf_counter() - start,
        })
    curve = pd.DataFrame(rows)

    # Only this run's rows are replaced in the cache, so curves of other runs survive.
    if not cached.empty:
        cached = cached.loc[cached["run"] != spec["name"]]
    pd.concat([cached, curve], ignore_index=True).to_csv(cache, index=False)
    return curve


def _bucket_labels(edges: list[float]) -> list[str]:
    """Human-readable labels for rainfall buckets, for example ``1 to 3`` and ``10+``."""
    fmt = lambda value: f"{value:g}"  # noqa: E731
    labels = [f"< {fmt(edges[0])}"]
    labels += [f"{fmt(low)} to {fmt(high)}" for low, high in zip(edges[:-1], edges[1:])]
    labels.append(f"{fmt(edges[-1])}+")
    return labels


def error_analysis(test_frame: pd.DataFrame, predictions: dict[str, np.ndarray],
                   bucket_edges: list[float], scores: dict[str, np.ndarray | None] | None = None,
                   top: int = 10, target: str = "RainTomorrow") -> dict[str, pd.DataFrame]:
    """Break test errors down by season, rainfall amount, regime change and location.

    Args:
        test_frame: The test split (needs ``date``, ``location``, ``rain_today``,
            ``tomorrow_mm`` and the target).
        predictions: Predicted labels per model name.
        bucket_edges: Boundaries of tomorrow's rainfall buckets in mm.
        scores: Continuous scores per model name, for the most confident mistakes.
        top: Number of most confident mistakes to list per model and error type.
        target: Name of the target column.

    Returns:
        A dictionary of tables: ``by_month``, ``by_bucket``, ``by_change``, ``by_location``
        and, when scores are given, ``false_positives_<model>`` and
        ``false_negatives_<model>``.

    The rainfall buckets show whether errors concentrate just around the 1 mm threshold,
    where the label is almost arbitrary, and the change-day table shows whether models fail
    mainly when tomorrow differs from today, which is the hard part of the problem.
    """
    frame = test_frame.copy()
    y_true = frame[target].to_numpy()
    for name, pred in predictions.items():
        frame[f"error_{name}"] = (np.asarray(pred) != y_true).astype(int)
    error_columns = [f"error_{name}" for name in predictions]
    rename = {f"error_{name}": name for name in predictions}

    frame["month"] = frame["date"].dt.month
    edges = [-np.inf, *bucket_edges, np.inf]
    frame["tomorrow_bucket"] = pd.cut(frame["tomorrow_mm"], bins=edges, right=False,
                                      labels=_bucket_labels(list(bucket_edges)))
    frame["day_type"] = np.where(frame["rain_today"] != frame[target],
                                 "change day", "persistent day")

    def rate_by(column: str) -> pd.DataFrame:
        """Error rate per model and the number of days for each value of ``column``."""
        grouped = frame.groupby(column, observed=True)
        table = grouped[error_columns].mean().rename(columns=rename)
        table["days"] = grouped.size()
        return table

    tables = {"by_month": rate_by("month"), "by_bucket": rate_by("tomorrow_bucket"),
              "by_change": rate_by("day_type"), "by_location": rate_by("location")}

    # The most confident mistakes are the most informative ones: they show situations in
    # which the model is systematically misled rather than merely uncertain.
    shown = ["date", "location", "PRECTOTCORR", "tomorrow_mm", "RH2M", "CLOUD_AMT",
             "rain_sum_7d", "days_since_rain"]
    shown = [column for column in shown if column in frame.columns]
    for name, model_scores in (scores or {}).items():
        if model_scores is None:
            continue
        pred = np.asarray(predictions[name])
        frame["score"] = model_scores
        tables[f"false_positives_{name}"] = (
            frame.loc[(pred == 1) & (y_true == 0), [*shown, "score"]]
            .sort_values("score", ascending=False).head(top).reset_index(drop=True))
        tables[f"false_negatives_{name}"] = (
            frame.loc[(pred == 0) & (y_true == 1), [*shown, "score"]]
            .sort_values("score", ascending=True).head(top).reset_index(drop=True))
    return tables


def logreg_coefficients(pipeline: Pipeline, top: int = 15) -> pd.DataFrame:
    """Return the largest standardised coefficients of a fitted linear pipeline.

    Args:
        pipeline: A fitted pipeline whose final step exposes ``coef_``.
        top: Number of coefficients to return, largest absolute value first.

    Returns:
        A table with the feature name, the coefficient and its absolute value.

    Because numeric inputs are standardised, each coefficient is the change in log-odds of
    rain tomorrow for a one standard deviation increase in that feature, so magnitudes are
    comparable across features.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise TypeError(f"{type(model).__name__} has no coefficients to report.")
    names = [name.split("__", 1)[-1]
             for name in pipeline.named_steps["prep"].get_feature_names_out()]
    table = pd.DataFrame({"feature": names, "coefficient": model.coef_.ravel()})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).head(top).reset_index(drop=True)


def dl_permutation_importance(model: Any, dl_data: Any, threshold: float,
                              feature_names: list[str] | None = None, split: str = "test",
                              seed: int = 0) -> pd.DataFrame:
    """Measure how much the network relies on each numeric feature.

    Args:
        model: A trained Keras model.
        dl_data: The ``dl.DLData`` instance.
        threshold: Probability above which a day is predicted rainy.
        feature_names: Numeric features to test (defaults to all of them).
        split: Split on which importance is measured.
        seed: Seed for the permutations.

    Returns:
        One row per feature with the drop in macro F1 when that feature is shuffled.

    Shuffling one column breaks its relationship with the target while keeping its
    distribution, so the resulting loss in macro F1 measures how much the trained network
    depends on it. Unlike coefficients, this works for any model, however non-linear.
    """
    from src.dl import layout_of, predict_scores

    layout = layout_of(model)
    y_true = dl_data.parts[split]["y"].astype(int)
    base_inputs = dl_data.inputs(split, layout)
    baseline = f1_score(y_true, predict_scores(model, base_inputs) >= threshold, average="macro")

    rng = np.random.default_rng(seed)
    rows = []
    for name in feature_names or dl_data.numeric_columns:
        column = dl_data.numeric_columns.index(name)
        # The numeric block comes first in both layouts, so the same column index applies.
        if layout == "flat":
            inputs = base_inputs.copy()
            inputs[:, column] = rng.permutation(inputs[:, column])
        else:
            numeric = base_inputs["numeric"].copy()
            numeric[:, column] = rng.permutation(numeric[:, column])
            inputs = {**base_inputs, "numeric": numeric}
        score = f1_score(y_true, predict_scores(model, inputs) >= threshold, average="macro")
        rows.append({"feature": name, "macro_f1_drop": baseline - score})
    return (pd.DataFrame(rows).sort_values("macro_f1_drop", ascending=False)
            .reset_index(drop=True))


def count_parameters(model: Any) -> int:
    """Count the learned parameters of a Keras model or a scikit-learn pipeline.

    Linear models count coefficients and intercepts; tree ensembles count tree nodes, which
    is the closest analogue of a parameter for a non-parametric model.
    """
    if hasattr(model, "count_params"):
        return int(model.count_params())
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "coef_"):
        return int(estimator.coef_.size + np.size(estimator.intercept_))
    if hasattr(estimator, "estimators_"):
        return int(sum(tree.tree_.node_count for tree in estimator.estimators_))
    if hasattr(estimator, "_predictors"):
        return int(sum(len(p.nodes) for stage in estimator._predictors for p in stage))
    return -1


INTERPRETABILITY = {
    "LogisticRegression": "High: one signed coefficient per feature",
    "LinearSVC": "High: one signed weight per feature",
    "RandomForestClassifier": "Medium: impurity or permutation importance only",
    "HistGradientBoostingClassifier": "Medium: permutation importance only",
    "keras": "Low: permutation importance only",
}


def paradigm_comparison(log: ResultsLog, final_metrics: pd.DataFrame,
                        models: dict[str, Any]) -> pd.DataFrame:
    """Summarise the best traditional and deep learning models side by side.

    Args:
        log: The results log (validation metrics and training times).
        final_metrics: ``TestResults.metrics`` from ``final_test_evaluation``.
        models: The loaded final models keyed by ``best_ml`` and ``best_dl``.

    Returns:
        One row per paradigm with validation and test scores, training time, parameter
        count and interpretability.
    """
    frame = log.to_frame().set_index("run")
    rows = []
    for key, label in (("best_ml", "Traditional ML"), ("best_dl", "Deep learning")):
        run = final_metrics.loc[key, "run"]
        model = models[key]
        if hasattr(model, "count_params"):
            kind = "keras"
        else:
            kind = type(model.named_steps["model"]).__name__
        rows.append({
            "paradigm": label, "run": run,
            "val_macro_f1": frame.loc[run, "macro_f1"],
            "test_macro_f1": final_metrics.loc[key, "macro_f1"],
            "val_roc_auc": frame.loc[run, "roc_auc"],
            "test_roc_auc": final_metrics.loc[key, "roc_auc"],
            "train_seconds": frame.loc[run, "seconds"],
            "parameters": count_parameters(model),
            "interpretability": INTERPRETABILITY.get(kind, "Unknown"),
        })
    return pd.DataFrame(rows).set_index("paradigm")
