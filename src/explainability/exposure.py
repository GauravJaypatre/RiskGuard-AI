"""
RiskGuard AI — Financial Exposure Engine

Estimates potential exposure (sum of flagged transaction amounts) and
high-confidence exposure (subset with higher-confidence risk scores).

IMPORTANT: This engine NEVER claims "fraud prevented" — only
"potential exposure identified."
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import DEFAULT_RISK_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD
from src.schemas import ExposureResult


def calculate_exposure(
    merchant_id: str,
    scored_df: pd.DataFrame,
    threshold: float = DEFAULT_RISK_THRESHOLD,
    high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> ExposureResult:
    """
    Compute potential and high-confidence financial exposure for a merchant.

    Potential exposure = sum of amounts for all transactions scoring above threshold.
    High-confidence exposure = sum of amounts for transactions scoring above high_threshold.

    The 'scored_df' must have a 'risk_score' column (model output probability)
    and an 'amount' column.

    Parameters
    ----------
    merchant_id : str
        Merchant to compute exposure for.
    scored_df : pd.DataFrame
        Transactions with columns: transaction_id, merchant_id, amount, risk_score.
    threshold : float
        Risk score threshold for potential exposure. Default 0.5.
    high_threshold : float
        Higher threshold for high-confidence exposure. Default 0.8.

    Returns
    -------
    ExposureResult
        Exposure estimate — never labeled as "fraud prevented."
    """
    # Filter to this merchant
    mdf = scored_df[scored_df["merchant_id"] == merchant_id]

    if mdf.empty:
        return ExposureResult(
            merchant_id=merchant_id,
            potential_exposure=0.0,
            high_confidence_exposure=0.0,
            flagged_count=0,
            high_confidence_count=0,
        )

    # Potential exposure: transactions scoring above threshold
    flagged = mdf[mdf["risk_score"] >= threshold]
    potential_exposure = float(flagged["amount"].sum())
    flagged_count = len(flagged)

    # High-confidence exposure: transactions scoring above high threshold
    high_conf = mdf[mdf["risk_score"] >= high_threshold]
    high_confidence_exposure = float(high_conf["amount"].sum())
    high_confidence_count = len(high_conf)

    return ExposureResult(
        merchant_id=merchant_id,
        potential_exposure=round(potential_exposure, 2),
        high_confidence_exposure=round(high_confidence_exposure, 2),
        flagged_count=flagged_count,
        high_confidence_count=high_confidence_count,
        computed_at=datetime.now(),
    )


def calculate_total_exposure(
    scored_df: pd.DataFrame,
    threshold: float = DEFAULT_RISK_THRESHOLD,
    high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> ExposureResult:
    """
    Compute total exposure across ALL merchants.

    Parameters
    ----------
    scored_df : pd.DataFrame
        All transactions with risk_score column.
    threshold : float
        Risk score threshold for flagging.
    high_threshold : float
        Higher threshold for high-confidence subset.

    Returns
    -------
    ExposureResult
        Total exposure with merchant_id set to "ALL".
    """
    if scored_df.empty or "risk_score" not in scored_df.columns:
        return ExposureResult(
            merchant_id="ALL",
            potential_exposure=0.0,
            high_confidence_exposure=0.0,
            flagged_count=0,
            high_confidence_count=0,
        )

    flagged = scored_df[scored_df["risk_score"] >= threshold]
    high_conf = scored_df[scored_df["risk_score"] >= high_threshold]

    return ExposureResult(
        merchant_id="ALL",
        potential_exposure=round(float(flagged["amount"].sum()), 2),
        high_confidence_exposure=round(float(high_conf["amount"].sum()), 2),
        flagged_count=len(flagged),
        high_confidence_count=len(high_conf),
        computed_at=datetime.now(),
    )


def calculate_per_merchant_exposure(
    scored_df: pd.DataFrame,
    threshold: float = DEFAULT_RISK_THRESHOLD,
    high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> Dict[str, ExposureResult]:
    """
    Compute exposure for each merchant separately.

    Returns
    -------
    Dict[str, ExposureResult]
        Keyed by merchant_id.
    """
    results: Dict[str, ExposureResult] = {}

    for merchant_id in scored_df["merchant_id"].unique():
        results[merchant_id] = calculate_exposure(
            merchant_id=merchant_id,
            scored_df=scored_df,
            threshold=threshold,
            high_threshold=high_threshold,
        )

    return results


def format_exposure(result: ExposureResult) -> str:
    """Format exposure result as a human-readable summary."""
    lines = [
        f"Financial Exposure Estimate — Merchant: {result.merchant_id}",
        "-" * 50,
        f"  Potential exposure identified:      ₹{result.potential_exposure:>12,.2f}  ({result.flagged_count} transactions)",
        f"  High-confidence exposure identified: ₹{result.high_confidence_exposure:>12,.2f}  ({result.high_confidence_count} transactions)",
        "",
        "  Note: These figures represent potential exposure identified by the",
        "  system. They are NOT claims of fraud prevented.",
    ]
    return "\n".join(lines)
