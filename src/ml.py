"""Traditional machine learning with scikit-learn.

This stage receives the ``Splits`` and ``FeatureSets`` objects, builds a preprocessing and
model pipeline for each experiment described in ``experiments.yaml``, fits it on the
training split, scores it on the validation split and records the result in the shared
``ResultsLog``. Hyperparameter tuning uses a single time-aware validation fold rather than
shuffled cross-validation, so tuning respects the chronological nature of the data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.config import ProjectPaths
from src.features import FeatureSets
from src.metrics import ResultsLog, compute_metrics
from src.split import Splits

ESTIMATORS = {
    "logistic_regression": LogisticRegression,
    "linear_svm": LinearSVC,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Create the column transformer shared by every scikit-learn model.

    Args:
        numeric: Numeric feature names, standardised.
        categorical: Categorical feature names, one-hot encoded.

    Returns:
        An unfitted ``ColumnTransformer``.

    The transformer is always placed inside a ``Pipeline``, so its means, standard
    deviations and categories are learnt from the training split only. Unknown categories
    are ignored rather than raising an error, and the output is kept dense because
    histogram gradient boosting does not accept sparse input.
    """
    return ColumnTransformer(
        [("num", StandardScaler(), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        sparse_threshold=0.0,
    )


def build_pipeline(spec: dict, feature_sets: FeatureSets,
                   params: dict | None = None) -> tuple[Pipeline, list[str]]:
    """Assemble the preprocessing and estimator pipeline for one experiment.

    Args:
        spec: One ML entry from ``experiments.yaml``.
        feature_sets: The feature registry.
        params: Hyperparameters that override those in ``spec['params']``.

    Returns:
        The unfitted pipeline and the list of input columns it expects.
    """
    numeric = feature_sets.numeric(spec["feature_set"])
    columns = numeric + feature_sets.categorical
    model_params = {**spec.get("params", {}), **(params or {})}
    estimator = ESTIMATORS[spec["estimator"]](**model_params)
    pipeline = Pipeline([("prep", make_preprocessor(numeric, feature_sets.categorical)),
                         ("model", estimator)])
    return pipeline, columns


def decision_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return a continuous score for the rain class, for ROC-AUC and ROC curves.

    Args:
        pipeline: A fitted pipeline.
        X: Input rows.

    Returns:
        Probabilities when the model provides them, otherwise decision-function values
        (the case for ``LinearSVC``, which has no probabilistic output).
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def _to_python(value: Any) -> Any:
    """Convert NumPy scalars to plain Python types so they can be written to JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_best_params(path: Path) -> dict[str, dict]:
    """Read ``best_params.json``, returning an empty mapping if it does not exist yet."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_sklearn_experiment(spec: dict, splits: Splits, feature_sets: FeatureSets,
                           log: ResultsLog, paths: ProjectPaths,
                           best_params: dict | None = None) -> Pipeline:
    """Fit one scikit-learn pipeline on the training split and score it on validation.

    Args:
        spec: One ML entry from ``experiments.yaml``.
        splits: The chronological splits.
        feature_sets: The feature registry.
        log: The shared results log, updated in place.
        paths: Resolved project paths.
        best_params: Tuned hyperparameters to apply on top of ``spec['params']``.

    Returns:
        The fitted pipeline.

    The fitted pipeline is saved with joblib unless ``save_model: false``; random forests
    are not saved because their files run to hundreds of megabytes and they are not
    candidates for the final model when another run scores higher.
    """
    target = splits.target
    pipeline, columns = build_pipeline(spec, feature_sets, best_params)

    # Only the training split is used for fitting, so the scaler and encoder never see
    # validation or test statistics.
    start = time.perf_counter()
    pipeline.fit(splits.train[columns], splits.train[target])
    seconds = time.perf_counter() - start

    # Training accuracy is recorded alongside validation metrics to expose overfitting.
    train_acc = accuracy_score(splits.train[target], pipeline.predict(splits.train[columns]))
    X_val = splits.val[columns]
    metrics = compute_metrics(splits.val[target], pipeline.predict(X_val),
                              decision_scores(pipeline, X_val))
    log.add(spec["name"], "ml", metrics, train_acc, seconds,
            notes=spec.get("notes", ""), feature_set=spec["feature_set"])

    if spec.get("save_model", True):
        joblib.dump(pipeline, paths.experiment_models_dir / f"{spec['name']}.joblib")
    return pipeline


def _search_space(space: dict) -> dict[str, Any]:
    """Translate the YAML search space into scikit-learn parameter distributions.

    Args:
        space: Mapping of hyperparameter name to either ``{choices: [...]}`` or
            ``{dist: loguniform|randint|uniform, low, high}``.

    Returns:
        A ``param_distributions`` dictionary with the ``model__`` prefix required by the
        pipeline.

    Log-uniform distributions are used for scale parameters such as ``C`` or the learning
    rate, because their useful values span several orders of magnitude.
    """
    distributions = {}
    for name, definition in space.items():
        if "choices" in definition:
            value = list(definition["choices"])
        elif definition["dist"] == "loguniform":
            value = stats.loguniform(definition["low"], definition["high"])
        elif definition["dist"] == "randint":
            value = stats.randint(definition["low"], definition["high"])
        elif definition["dist"] == "uniform":
            value = stats.uniform(definition["low"], definition["high"] - definition["low"])
        else:
            raise ValueError(f"Unknown distribution for {name}: {definition}")
        distributions[f"model__{name}"] = value
    return distributions


def tune_sklearn_experiment(spec: dict, splits: Splits, feature_sets: FeatureSets,
                            log: ResultsLog, paths: ProjectPaths,
                            seed: int) -> tuple[Pipeline, dict]:
    """Tune hyperparameters on the validation period, then refit on the training split.

    Args:
        spec: One ML entry from ``experiments.yaml`` with a ``search`` block.
        splits: The chronological splits.
        feature_sets: The feature registry.
        log: The shared results log, updated in place.
        paths: Resolved project paths.
        seed: Random seed for the search.

    Returns:
        The refitted pipeline and the best hyperparameters.

    ``RandomizedSearchCV`` normally shuffles rows into folds, which would let the search
    train on days that come after the days it is scored on. A ``PredefinedSplit`` marks
    training rows with -1 (always used for fitting) and validation rows with 0 (the single
    scoring fold), so every candidate is trained on the past and scored on the future.
    Because the same validation period is used both to choose and to report, validation
    scores of tuned runs are slightly optimistic; this is exactly why the test split is held
    back for the final evaluation.
    """
    target = splits.target
    search_cfg = spec["search"]
    pipeline, columns = build_pipeline(spec, feature_sets)

    # Training and validation rows are stacked so the predefined fold can address both.
    X_all = pd.concat([splits.train[columns], splits.val[columns]], ignore_index=True)
    y_all = pd.concat([splits.train[target], splits.val[target]], ignore_index=True)
    fold = np.r_[np.full(len(splits.train), -1), np.zeros(len(splits.val), dtype=int)]

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=_search_space(search_cfg["space"]),
        n_iter=search_cfg["n_iter"],
        scoring="f1_macro",
        cv=PredefinedSplit(fold),
        refit=False,
        random_state=seed,
        n_jobs=search_cfg.get("search_jobs", -1),
    )
    start = time.perf_counter()
    search.fit(X_all, y_all)
    print(f"{spec['name']}: search of {search_cfg['n_iter']} candidates took "
          f"{time.perf_counter() - start:.0f} s, best validation macro F1 "
          f"{search.best_score_:.4f}")

    # The pipeline prefix is stripped and NumPy types converted so the parameters can be
    # stored as readable JSON and reused by later runs through ``params_from``.
    best = {name.removeprefix("model__"): _to_python(value)
            for name, value in search.best_params_.items()}
    store_path = paths.results_dir / "best_params.json"
    store = load_best_params(store_path)
    store[spec["name"]] = best
    with open(store_path, "w", encoding="utf-8") as handle:
        json.dump(store, handle, indent=2)

    # The final model of the run is refitted on the training split alone, so its recorded
    # validation score is computed exactly like the untuned runs.
    fitted = run_sklearn_experiment(spec, splits, feature_sets, log, paths, best_params=best)
    return fitted, best
