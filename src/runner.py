"""Run every experiment listed in ``experiments.yaml``, resuming after interruptions.

The runner is the glue between the experiment registry and the training modules. It walks
the baselines, scikit-learn runs and Keras runs in the order they were performed, resolves
inherited hyperparameters, and saves the results log after every run. In ``auto`` mode a
run whose result and artifact already exist on Drive is reported as "loaded" instead of
being retrained, so a Colab disconnect never forces the full set of experiments to rerun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.baselines import predict_baseline
from src.config import ProjectPaths
from src.features import FeatureSets
from src.metrics import ResultsLog, compute_metrics
from src.ml import load_best_params, run_sklearn_experiment, tune_sklearn_experiment
from src.split import Splits


def results_path(paths: ProjectPaths) -> Path:
    """Location of the shared results log."""
    return paths.results_dir / "experiments.csv"


def merge_dl_spec(defaults: dict, spec: dict) -> dict:
    """Combine ``dl_defaults`` with one DL entry; the entry's values take precedence."""
    return {**defaults, **spec}


def _ml_finished(spec: dict, log: ResultsLog, paths: ProjectPaths) -> bool:
    """Decide whether an ML run can be reused instead of retrained.

    A run counts as finished when it is in the log, its model file exists (unless the run
    deliberately does not save one), and, for tuned runs, its best parameters were stored.
    """
    if not log.has(spec["name"]):
        return False
    if spec.get("save_model", True):
        if not (paths.experiment_models_dir / f"{spec['name']}.joblib").exists():
            return False
    if "search" in spec:
        return spec["name"] in load_best_params(paths.results_dir / "best_params.json")
    return True


def _dl_finished(spec: dict, log: ResultsLog, paths: ProjectPaths) -> bool:
    """Decide whether a DL run can be reused: logged, model saved and history saved."""
    return (log.has(spec["name"])
            and (paths.experiment_models_dir / f"{spec['name']}.keras").exists()
            and (paths.histories_dir / f"{spec['name']}.json").exists())


def run_experiments(experiments: dict, splits: Splits, feature_sets: FeatureSets,
                    dl_data: Any, log: ResultsLog, paths: ProjectPaths,
                    cfg: dict) -> ResultsLog:
    """Run baselines, ML runs and DL runs in the order listed in the registry.

    Args:
        experiments: The dictionary from ``config.load_experiments``.
        splits: The chronological splits.
        feature_sets: The feature registry.
        dl_data: A ``dl.DLData`` instance, or None to skip the deep learning runs.
        log: The shared results log (typically loaded from ``experiments.csv``).
        paths: Resolved project paths.
        cfg: The main configuration dictionary.

    Returns:
        The updated results log, also saved to ``results/experiments.csv``.
    """
    mode = cfg["run"]["mode"]
    if mode not in ("auto", "train"):
        raise ValueError("run.mode must be 'auto' or 'train'.")
    reuse = mode == "auto"
    seed = cfg["project"]["seed"]
    target = splits.target

    # Baselines take a fraction of a second, so they are always recomputed; this also keeps
    # them in step with any change to the target definition.
    for spec in experiments.get("baselines", []):
        val_pred = predict_baseline(spec["kind"], splits, "val")
        train_pred = predict_baseline(spec["kind"], splits, "train")
        metrics = compute_metrics(splits.val[target], val_pred)
        train_acc = compute_metrics(splits.train[target], train_pred)["accuracy"]
        log.add(spec["name"], "baseline", metrics, train_acc, 0.0,
                notes=spec.get("notes", spec["kind"].replace("_", " ")))
        print(f"[computed] {spec['name']}")
    log.save(results_path(paths))

    for spec in experiments.get("ml", []):
        if reuse and _ml_finished(spec, log, paths):
            print(f"[loaded]   {spec['name']}")
            continue

        # Inherited hyperparameters come from an earlier tuning run, so the regional runs
        # test the effect of the new features rather than a different configuration.
        inherited = None
        if "params_from" in spec:
            store = load_best_params(paths.results_dir / "best_params.json")
            if spec["params_from"] not in store:
                raise KeyError(f"{spec['name']} needs tuned parameters from "
                               f"{spec['params_from']}, which has not been run yet.")
            inherited = store[spec["params_from"]]

        if "search" in spec:
            tune_sklearn_experiment(spec, splits, feature_sets, log, paths, seed)
        else:
            run_sklearn_experiment(spec, splits, feature_sets, log, paths, inherited)
        log.save(results_path(paths))
        print(f"[trained]  {spec['name']}")

    dl_specs = experiments.get("dl", [])
    if dl_specs and dl_data is None:
        print("Deep learning runs skipped: no DLData provided.")
        return log

    for raw_spec in dl_specs:
        spec = merge_dl_spec(experiments["dl_defaults"], raw_spec)
        if reuse and _dl_finished(spec, log, paths):
            print(f"[loaded]   {spec['name']}")
            continue
        if spec["feature_set"] != dl_data.feature_set:
            raise ValueError(f"{spec['name']} uses feature set '{spec['feature_set']}' but "
                             f"DLData was built for '{dl_data.feature_set}'.")

        # TensorFlow is imported only here so the rest of the pipeline works without it.
        from src.dl import run_dl_experiment
        run_dl_experiment(spec, dl_data, log, paths,
                          threshold=cfg["evaluation"]["threshold"],
                          verbose=cfg["run"]["dl_verbose"])
        log.save(results_path(paths))
        print(f"[trained]  {spec['name']}")
    return log
