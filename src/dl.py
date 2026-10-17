"""Deep learning with TensorFlow/Keras and the ``tf.data`` API.

This stage receives the same ``Splits`` and ``FeatureSets`` as the scikit-learn stage, so
both paradigms learn from identical rows and features. ``DLData`` performs the
preprocessing once (scaler fitted on the training split only), ``make_dataset`` streams the
arrays through ``tf.data`` pipelines, and ``run_dl_experiment`` builds, trains, scores and
saves one network per entry in ``experiments.yaml``. Both the Sequential API (a plain
multilayer perceptron) and the Functional API (a two-input network with a learned location
embedding) are supported.
"""

from __future__ import annotations

import json
import re
import time

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src.config import ProjectPaths, set_global_seed
from src.features import FeatureSets
from src.metrics import ResultsLog, compute_metrics
from src.split import Splits

PARTS = ("train", "val", "test")


class DLData:
    """Scaled arrays for every split, in the two layouts the networks need.

    For each split, ``parts[split]`` holds:

    * ``flat``: scaled numeric features followed by a one-hot location block (Sequential).
    * ``numeric``: scaled numeric features only (Functional, first input).
    * ``location``: integer location index (Functional, embedding input).
    * ``y``: the binary target.

    Preparing the arrays once and reusing them for every run keeps the deep learning
    experiments fast and guarantees they all see exactly the same inputs.
    """

    def __init__(self, splits: Splits, feature_sets: FeatureSets, feature_set: str,
                 paths: ProjectPaths, seed: int) -> None:
        """Fit the scaler on the training split and transform every split.

        Args:
            splits: The chronological splits.
            feature_sets: The feature registry.
            feature_set: ``base`` or ``regional``; all DL runs must use this set.
            paths: Resolved project paths (the scaler is saved to ``models/``).
            seed: Global seed, used to shuffle the training dataset reproducibly.
        """
        self.feature_set = feature_set
        self.numeric_columns = feature_sets.numeric(feature_set)
        self.seed = seed

        # The scaler is fitted on training rows only, so validation and test statistics
        # never influence the inputs the network learns from.
        self.scaler = StandardScaler().fit(splits.train[self.numeric_columns])
        joblib.dump(self.scaler, paths.models_dir / "dl_scaler.joblib")

        # Location indices follow the sorted training locations, giving the embedding and
        # the one-hot block a stable, documented order.
        self.locations = sorted(splits.train["location"].unique())
        index_of = {name: index for index, name in enumerate(self.locations)}

        self.parts: dict[str, dict[str, np.ndarray]] = {}
        for name in PARTS:
            frame = splits.part(name)
            numeric = self.scaler.transform(frame[self.numeric_columns]).astype("float32")
            location = frame["location"].map(index_of)
            if location.isna().any():
                raise ValueError(f"Split '{name}' contains locations unseen in training.")
            location = location.to_numpy(dtype="int32")
            one_hot = np.eye(len(self.locations), dtype="float32")[location]
            self.parts[name] = {
                "flat": np.hstack([numeric, one_hot]),
                "numeric": numeric,
                "location": location,
                "y": frame[splits.target].to_numpy(dtype="float32"),
            }

    @property
    def n_numeric(self) -> int:
        """Number of numeric input features."""
        return len(self.numeric_columns)

    @property
    def flat_dim(self) -> int:
        """Width of the flat input used by Sequential models."""
        return self.n_numeric + len(self.locations)

    def inputs(self, split: str, layout: str) -> np.ndarray | dict[str, np.ndarray]:
        """Return the model inputs of one split in the requested layout.

        Args:
            split: ``train``, ``val`` or ``test``.
            layout: ``flat`` for Sequential models, ``functional`` for the two-input model.
        """
        part = self.parts[split]
        if layout == "flat":
            return part["flat"]
        return {"numeric": part["numeric"], "location": part["location"].reshape(-1, 1)}


def make_dataset(inputs: np.ndarray | dict[str, np.ndarray], y: np.ndarray | None,
                 batch_size: int, shuffle: bool = False,
                 seed: int | None = None) -> tf.data.Dataset:
    """Wrap arrays in a batched, prefetched ``tf.data`` pipeline.

    Args:
        inputs: A feature array, or a dictionary of arrays keyed by input name.
        y: Targets, or None to build a prediction-only dataset.
        batch_size: Number of rows per batch.
        shuffle: Whether to shuffle rows each epoch (training split only).
        seed: Shuffle seed, so the order of batches is reproducible.

    Returns:
        A ``tf.data.Dataset`` yielding ``(inputs, y)`` or ``inputs`` batches.

    Only the training split is shuffled, so that batches mix seasons and locations;
    validation and test keep their order so predictions can be matched to rows. Prefetching
    lets the input pipeline prepare the next batch while the current one trains.
    """
    tensors = inputs if y is None else (inputs, y)
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    if shuffle:
        # A buffer as large as the split gives a full shuffle; the split fits easily in memory.
        dataset = dataset.shuffle(buffer_size=len(y), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_dataset(dl_data: DLData, split: str, layout: str, batch_size: int,
                  shuffle: bool = False) -> tf.data.Dataset:
    """Build the ``tf.data`` pipeline for one split of ``DLData``."""
    return make_dataset(dl_data.inputs(split, layout), dl_data.parts[split]["y"],
                        batch_size, shuffle=shuffle, seed=dl_data.seed)


def _hidden_layers(spec: dict) -> list[keras.layers.Layer]:
    """Create the hidden layers shared by both architectures.

    Args:
        spec: A merged DL spec (``units``, ``batch_norm``, ``dropout`` and ``l2`` are used).

    Returns:
        The layers in order, ready to be stacked by either API.

    Each block is Dense, optional BatchNormalization, ReLU and optional Dropout. The Dense
    layer has no bias when batch normalisation follows, because the normalisation's own
    shift parameter makes the bias redundant. L2 regularisation, when set, penalises large
    weights to reduce overfitting.
    """
    regulariser = keras.regularizers.l2(spec["l2"]) if spec["l2"] > 0 else None
    layers: list[keras.layers.Layer] = []
    for units in spec["units"]:
        layers.append(keras.layers.Dense(units, use_bias=not spec["batch_norm"],
                                         kernel_regularizer=regulariser))
        if spec["batch_norm"]:
            layers.append(keras.layers.BatchNormalization())
        layers.append(keras.layers.ReLU())
        if spec["dropout"] > 0:
            layers.append(keras.layers.Dropout(spec["dropout"]))
    return layers


def _model_name(run: str) -> str:
    """Make a run name safe for use as a Keras model name."""
    return re.sub(r"[^A-Za-z0-9_]", "_", run)


def build_sequential(spec: dict, input_dim: int) -> keras.Model:
    """Build a multilayer perceptron with the Sequential API.

    Args:
        spec: A merged DL spec (``dl_defaults`` plus the run's overrides).
        input_dim: Width of the flat input (numeric features plus one-hot location).

    Returns:
        An uncompiled ``keras.Sequential`` model with a sigmoid output.
    """
    return keras.Sequential([keras.Input(shape=(input_dim,)), *_hidden_layers(spec),
                             keras.layers.Dense(1, activation="sigmoid")],
                            name=_model_name(spec["name"]))


def build_functional(spec: dict, n_numeric: int, n_locations: int) -> keras.Model:
    """Build a two-input network with a learned location embedding (Functional API).

    Args:
        spec: A merged DL spec; ``embed_dim`` sets the embedding size.
        n_numeric: Number of numeric features.
        n_locations: Number of distinct locations.

    Returns:
        An uncompiled ``keras.Model``.

    A one-hot location treats every district as unrelated to every other. An embedding
    instead learns a small dense vector per location, so districts with similar rainfall
    behaviour can end up close together, which the Sequential API cannot express because
    it needs two separate inputs.
    """
    numeric_in = keras.Input(shape=(n_numeric,), name="numeric")
    location_in = keras.Input(shape=(1,), dtype="int32", name="location")
    embedded = keras.layers.Embedding(n_locations, spec["embed_dim"],
                                      name="location_embedding")(location_in)
    embedded = keras.layers.Flatten()(embedded)
    x = keras.layers.Concatenate()([numeric_in, embedded])
    for layer in _hidden_layers(spec):
        x = layer(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model([numeric_in, location_in], output, name=_model_name(spec["name"]))


def make_optimizer(name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    """Create an optimiser by name.

    Args:
        name: ``adam``, ``sgd`` (with momentum 0.9) or ``rmsprop``.
        learning_rate: Step size.

    Returns:
        A Keras optimiser instance.
    """
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimiser '{name}'.")


def layout_of(model: keras.Model) -> str:
    """Return ``functional`` for the two-input model and ``flat`` otherwise."""
    return "functional" if len(model.inputs) > 1 else "flat"


def predict_scores(model: keras.Model, inputs: np.ndarray | dict[str, np.ndarray],
                   batch_size: int = 4096) -> np.ndarray:
    """Predict rain probabilities, feeding the inputs through ``tf.data``.

    Args:
        model: A trained Keras model.
        inputs: Inputs in the layout the model expects.
        batch_size: Prediction batch size (large, since no gradients are computed).

    Returns:
        A one-dimensional array of probabilities.
    """
    dataset = make_dataset(inputs, None, batch_size)
    return model.predict(dataset, verbose=0).ravel()


def run_dl_experiment(spec: dict, dl_data: DLData, log: ResultsLog, paths: ProjectPaths,
                      threshold: float, verbose: int = 0) -> tuple[keras.Model, dict]:
    """Build, train, score and save one network.

    Args:
        spec: A merged DL spec (``dl_defaults`` plus the run's overrides).
        dl_data: Prepared arrays.
        log: The shared results log, updated in place.
        paths: Resolved project paths.
        threshold: Probability above which a day is predicted rainy.
        verbose: Keras ``fit`` verbosity.

    Returns:
        The trained model (with its best weights) and its training history.

    Early stopping monitors validation loss and restores the best weights, so the stored
    model is the one from the best epoch, not the last one. The seed is reset before every
    run so that differences between runs come from the configuration, not from chance.
    """
    set_global_seed(dl_data.seed)

    if spec["architecture"] == "functional":
        model = build_functional(spec, dl_data.n_numeric, len(dl_data.locations))
        layout = "functional"
    else:
        model = build_sequential(spec, dl_data.flat_dim)
        layout = "flat"

    model.compile(optimizer=make_optimizer(spec["optimizer"], spec["learning_rate"]),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])

    train_ds = split_dataset(dl_data, "train", layout, spec["batch_size"], shuffle=True)
    val_ds = split_dataset(dl_data, "val", layout, spec["batch_size"])
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss",
                                            patience=spec["early_stopping_patience"],
                                            restore_best_weights=True)

    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=spec["max_epochs"],
                        callbacks=[stopper], verbose=verbose)
    seconds = time.perf_counter() - start

    # Metrics are recomputed from predictions with the restored best weights, using the
    # same functions as the scikit-learn runs so that both paradigms are scored identically.
    val_scores = predict_scores(model, dl_data.inputs("val", layout))
    val_pred = (val_scores >= threshold).astype(int)
    metrics = compute_metrics(dl_data.parts["val"]["y"].astype(int), val_pred, val_scores)
    train_scores = predict_scores(model, dl_data.inputs("train", layout))
    train_acc = accuracy_score(dl_data.parts["train"]["y"].astype(int),
                               (train_scores >= threshold).astype(int))

    record = {key: [float(v) for v in values] for key, values in history.history.items()}
    epochs_run = len(record["loss"])
    record["best_epoch"] = int(np.argmin(record["val_loss"])) + 1
    record["epochs_run"] = epochs_run

    log.add(spec["name"], "dl", metrics, train_acc, seconds, notes=spec.get("notes", ""),
            feature_set=spec["feature_set"], epochs=epochs_run)

    model.save(paths.experiment_models_dir / f"{spec['name']}.keras")
    with open(paths.histories_dir / f"{spec['name']}.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle)
    return model, record
