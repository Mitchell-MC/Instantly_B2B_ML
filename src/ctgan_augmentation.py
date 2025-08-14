"""
CTGAN-based synthetic data augmentation for tabular training data.

This module augments the training split only (never the test split) to avoid
data leakage. It supports per-class balancing by training a small CTGAN per
class label and sampling up to the majority class size or a configured cap.

If the `ctgan` package is not installed, the module will no-op gracefully.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # Lazy import; will be checked at runtime
    from ctgan import CTGAN
    _CTGAN_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _CTGAN_AVAILABLE = False

from sklearn.impute import SimpleImputer


def _get_ctgan_params(config: dict) -> dict:
    augmentation_cfg = (
        config.get("augmentation", {})
        .get("ctgan", {})
    )
    # Sensible, relatively light defaults if not provided
    return {
        "epochs": augmentation_cfg.get("epochs", 100),
        "batch_size": augmentation_cfg.get("batch_size", 256),
        "generator_dim": augmentation_cfg.get("generator_dim", (128, 128)),
        "discriminator_dim": augmentation_cfg.get("discriminator_dim", (128, 128)),
        "log_frequency": augmentation_cfg.get("log_frequency", True),
        "verbose": augmentation_cfg.get("verbose", True),
        "random_state": augmentation_cfg.get("random_state", 42),
    }


def _detect_discrete_columns(df: pd.DataFrame, max_unique: int = 20) -> List[str]:
    """
    Heuristic discrete column detection for numeric-only dataframes.
    Any integer column with limited unique values is treated as discrete.
    """
    discrete_cols: List[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
            n_unique = series.nunique(dropna=True)
            if n_unique <= max_unique:
                discrete_cols.append(col)
    return discrete_cols


def _impute_for_ctgan(X: pd.DataFrame) -> Tuple[pd.DataFrame, SimpleImputer]:
    """Median-impute missing values for CTGAN training/sampling."""
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    return X_imputed, imputer


def _train_ctgan_single_class(
    X_class: pd.DataFrame,
    ctgan_params: dict,
    discrete_columns: Optional[List[str]]
) -> Optional[CTGAN]:
    if len(X_class) < 50:
        # Too few rows to train a meaningful CTGAN
        return None

    model = CTGAN(
        epochs=ctgan_params["epochs"],
        batch_size=ctgan_params["batch_size"],
        generator_dim=ctgan_params["generator_dim"],
        discriminator_dim=ctgan_params["discriminator_dim"],
        log_frequency=ctgan_params["log_frequency"],
        verbose=ctgan_params["verbose"],
        random_state=ctgan_params["random_state"],
    )

    model.fit(X_class, discrete_columns=discrete_columns or [])
    return model


def augment_training_data_with_ctgan(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Optionally augment the training data using CTGAN.

    - If augmentation is disabled or ctgan is unavailable, returns inputs unchanged
    - Per-class balancing: sample up to the majority class count or a configured cap
    """
    aug_cfg = config.get("augmentation", {}).get("ctgan", {})
    enabled = aug_cfg.get("enabled", False)
    if not enabled:
        return X_train, y_train

    if not _CTGAN_AVAILABLE:
        print("⚠️ CTGAN is not installed. Skipping CTGAN augmentation.")
        print("   Install with: pip install ctgan")
        return X_train, y_train

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    # Impute for CTGAN training
    X_imputed, imputer = _impute_for_ctgan(X_train)

    # Detect discrete columns for CTGAN
    discrete_columns = _detect_discrete_columns(X_imputed)

    # Determine per-class targets
    value_counts = y_train.value_counts()
    classes = list(value_counts.index)
    max_count = int(value_counts.max())

    # Optional cap per class
    max_per_class_cap = aug_cfg.get("max_samples_per_class")
    if isinstance(max_per_class_cap, int) and max_per_class_cap > 0:
        target_per_class = min(max_count, max_per_class_cap)
    else:
        target_per_class = max_count

    ctgan_params = _get_ctgan_params(config)

    augmented_rows: List[pd.DataFrame] = []
    augmented_labels: List[pd.Series] = []

    for cls in classes:
        cls_mask = y_train == cls
        X_cls = X_imputed.loc[cls_mask]
        current_n = len(X_cls)

        # Nothing to add if already at or above target
        if current_n >= target_per_class:
            continue

        n_to_sample = target_per_class - current_n

        try:
            model = _train_ctgan_single_class(X_cls, ctgan_params, discrete_columns)
        except Exception as e:  # pragma: no cover - safety net
            print(f"⚠️ CTGAN training failed for class {cls}: {e}")
            model = None

        if model is None:
            continue

        try:
            X_synth = model.sample(n_to_sample)
            # Map back any imputation effects are acceptable; we keep synthetic as-is
            # Ensure column order
            X_synth = X_synth[X_imputed.columns]
            y_synth = pd.Series([cls] * len(X_synth), index=X_synth.index)

            augmented_rows.append(X_synth)
            augmented_labels.append(y_synth)
        except Exception as e:  # pragma: no cover - safety net
            print(f"⚠️ CTGAN sampling failed for class {cls}: {e}")
            continue

    if not augmented_rows:
        # No augmentation generated
        return X_train, y_train

    X_aug = pd.concat([X_imputed] + augmented_rows, axis=0, ignore_index=True)
    y_aug = pd.concat([y_train.reset_index(drop=True)] + augmented_labels, axis=0, ignore_index=True)

    # Shuffle after augmentation
    rng = np.random.RandomState(_get_ctgan_params(config)["random_state"])
    perm = rng.permutation(len(X_aug))
    X_aug = X_aug.iloc[perm].reset_index(drop=True)
    y_aug = y_aug.iloc[perm].reset_index(drop=True)

    # Return with original column names/dtypes as DataFrame/Series
    return X_aug.astype(float), y_aug


