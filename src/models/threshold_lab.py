"""
RiskGuard AI — Interactive Threshold Lab

Sweeps threshold 0.01–0.99, computes precision/recall/F1/FP/FN/expected_loss
at each point. Produces a JSON array the dashboard slider can consume live.

Does NOT hardcode two example thresholds — computes the full curve.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.config import (
    THRESHOLD_SWEEP_POINTS,
    FP_COST_PER_TRANSACTION,
    FN_COST_MULTIPLIER,
    MODELS_DIR,
)
from src.schemas import ThresholdPoint

logger = logging.getLogger(__name__)


def compute_threshold_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    num_points: int = THRESHOLD_SWEEP_POINTS,
) -> List[ThresholdPoint]:
    """
    Sweep threshold from 0.01 to 0.99 and compute metrics at each point.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_prob : np.ndarray
        Model-predicted probabilities.
    amounts : np.ndarray
        Transaction amounts (for expected loss calculation).
    num_points : int
        Number of points in the sweep.

    Returns
    -------
    List[ThresholdPoint]
        Full threshold curve data.
    """
    thresholds = np.linspace(0.01, 0.99, num_points)
    curve: List[ThresholdPoint] = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Expected loss: FP cost + FN cost
        fp_cost = fp * FP_COST_PER_TRANSACTION
        fn_cost = float(amounts[(y_pred == 0) & (y_true == 1)].sum()) * FN_COST_MULTIPLIER
        expected_loss = fp_cost + fn_cost

        curve.append(ThresholdPoint(
            threshold=round(float(threshold), 4),
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            f1=round(float(f1), 4),
            fp_count=int(fp),
            fn_count=int(fn),
            expected_loss=round(float(expected_loss), 2),
        ))

    return curve


def save_threshold_curve(curve: List[ThresholdPoint]) -> None:
    """Save threshold curve to JSON for dashboard consumption."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "threshold": p.threshold,
            "precision": p.precision,
            "recall": p.recall,
            "f1": p.f1,
            "fp_count": p.fp_count,
            "fn_count": p.fn_count,
            "expected_loss": p.expected_loss,
        }
        for p in curve
    ]

    with open(MODELS_DIR / "threshold_curve.json", "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Threshold curve ({len(data)} points) saved to {MODELS_DIR / 'threshold_curve.json'}")


def load_threshold_curve() -> List[Dict]:
    """Load threshold curve from disk."""
    with open(MODELS_DIR / "threshold_curve.json") as f:
        return json.load(f)


def find_optimal_threshold(curve: List[ThresholdPoint], metric: str = "f1") -> ThresholdPoint:
    """Find the threshold that maximizes a given metric."""
    return max(curve, key=lambda p: getattr(p, metric))
