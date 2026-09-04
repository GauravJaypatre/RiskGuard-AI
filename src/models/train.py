"""
RiskGuard AI — Model Training

Trains and benchmarks Logistic Regression, Random Forest, and XGBoost.
Picks the best on held-out PR-AUC. Reports the comparison honestly.

All stochastic steps use config.RANDOM_SEED for reproducibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.config import RANDOM_SEED, MODELS_DIR, THRESHOLD_SWEEP_POINTS

logger = logging.getLogger(__name__)


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Train LR, RF, and XGBoost. Use validation set for hyperparameter selection.

    Parameters
    ----------
    X_train, y_train : Training features and labels.
    X_val, y_val : Validation features and labels.
    feature_names : List of feature column names.

    Returns
    -------
    Dict with trained models, scaler, and validation metrics.
    """
    results = {}

    # Ensure feature order consistency
    X_train_arr = X_train[feature_names].values
    X_val_arr = X_val[feature_names].values

    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_arr)
    X_val_scaled = scaler.transform(X_val_arr)

    # ── Logistic Regression ───────────────────────────────────────────
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
    )
    lr.fit(X_train_scaled, y_train)
    lr_val_metrics = _evaluate(lr, X_val_scaled, y_val, "LogisticRegression")
    results["LogisticRegression"] = {
        "model": lr,
        "needs_scaling": True,
        "metrics": lr_val_metrics,
    }

    # ── Random Forest ─────────────────────────────────────────────────
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_arr, y_train)
    rf_val_metrics = _evaluate(rf, X_val_arr, y_val, "RandomForest")
    results["RandomForest"] = {
        "model": rf,
        "needs_scaling": False,
        "metrics": rf_val_metrics,
    }

    # ── XGBoost ───────────────────────────────────────────────────────
    logger.info("Training XGBoost...")
    # Calculate scale_pos_weight for class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="aucpr",
        early_stopping_rounds=20,
    )
    xgb_model.fit(
        X_train_arr, y_train,
        eval_set=[(X_val_arr, y_val)],
        verbose=False,
    )
    xgb_val_metrics = _evaluate(xgb_model, X_val_arr, y_val, "XGBoost")
    results["XGBoost"] = {
        "model": xgb_model,
        "needs_scaling": False,
        "metrics": xgb_val_metrics,
    }

    # ── Pick the best by F1 (with PR-AUC as tiebreaker) ─────────────
    # F1 rather than PR-AUC because a model with high PR-AUC but 0
    # precision/recall at the operating threshold (0.5) is unusable.
    best_name = max(
        results,
        key=lambda k: (results[k]["metrics"]["f1"], results[k]["metrics"]["pr_auc"]),
    )
    results["best_model_name"] = best_name
    results["scaler"] = scaler
    results["feature_names"] = feature_names

    logger.info(f"\nModel Comparison (Validation Set):")
    logger.info(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'PR-AUC':>10}")
    logger.info("-" * 60)
    for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        m = results[name]["metrics"]
        marker = " ← BEST" if name == best_name else ""
        logger.info(
            f"{name:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m['pr_auc']:>10.4f}{marker}"
        )

    # Save the best model and scaler
    _save_models(results)

    return results


def _evaluate(model, X, y, name: str) -> Dict[str, float]:
    """Evaluate a model and return metrics."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    pr_auc = average_precision_score(y, y_prob)

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
    }


def evaluate_on_test(
    results: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models on held-out test set. Report honest comparison.
    """
    test_metrics = {}

    for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        model = results[name]["model"]
        needs_scaling = results[name]["needs_scaling"]

        X_arr = X_test[feature_names].values
        if needs_scaling:
            X_arr = results["scaler"].transform(X_arr)

        metrics = _evaluate(model, X_arr, y_test, name)
        test_metrics[name] = metrics

    logger.info(f"\nTest Set Results:")
    logger.info(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'PR-AUC':>10}")
    logger.info("-" * 60)
    for name, m in test_metrics.items():
        logger.info(
            f"{name:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m['pr_auc']:>10.4f}"
        )

    return test_metrics


def _save_models(results: Dict[str, Any]):
    """Persist best model, all models, and scaler to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_name = results["best_model_name"]
    best_model = results[best_name]["model"]

    # Save best model
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    # Save scaler
    joblib.dump(results["scaler"], MODELS_DIR / "scaler.joblib")

    # Save feature names
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(results["feature_names"], f)

    # Save all models
    for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        joblib.dump(results[name]["model"], MODELS_DIR / f"{name}.joblib")

    # Save model comparison
    comparison = {}
    for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        comparison[name] = results[name]["metrics"]
    comparison["best"] = best_name

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Models saved to {MODELS_DIR}")


def load_best_model() -> Tuple[Any, Any, List[str]]:
    """Load the best model, scaler, and feature names from disk."""
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names
