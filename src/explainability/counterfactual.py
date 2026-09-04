"""
RiskGuard AI — Counterfactual Explanation Engine

For a flagged transaction, perturbs meaningful features one at a time,
re-scores with the model, and shows how risk score changes.

All explanations are explicitly framed as model sensitivity, NOT causal claims:
  "According to the model, if [feature] were [value], the risk score
   would change from X to Y."

The most impactful counterfactual is tied to a concrete investigation
recommendation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.schemas import CounterfactualResult, Perturbation


# ═══════════════════════════════════════════════════════════════════════════
# Feature Perturbation Strategies
# ═══════════════════════════════════════════════════════════════════════════

# Maps feature names to human-readable descriptions and perturbation targets.
# "safe_value" is what we perturb TO — a value that looks normal/benign.
# "investigation_hint" is the recommendation if this is the top factor.
PERTURBATION_CONFIG: Dict[str, Dict[str, Any]] = {
    "shared_device_count": {
        "display_name": "Shared device count",
        "safe_value": 1,
        "investigation_hint": (
            "The strongest factor is device sharing — investigate whether "
            "the {original:.0f} customers on this device are distinct individuals."
        ),
    },
    "customers_per_device": {
        "display_name": "Customers per device",
        "safe_value": 1,
        "investigation_hint": (
            "Multiple customers ({original:.0f}) share a single device. "
            "Verify whether these are separate people or synthetic identities."
        ),
    },
    "velocity_1h": {
        "display_name": "Transaction velocity (1h)",
        "safe_value": 2,
        "investigation_hint": (
            "High short-term velocity ({original:.0f} tx/hour) is a key factor. "
            "Check if this burst correlates with a known event or is anomalous."
        ),
    },
    "velocity_24h": {
        "display_name": "Transaction velocity (24h)",
        "safe_value": 5,
        "investigation_hint": (
            "Elevated daily transaction count ({original:.0f}) drove the risk score. "
            "Compare with merchant's typical daily volume."
        ),
    },
    "cluster_size": {
        "display_name": "Relationship cluster size",
        "safe_value": 1,
        "investigation_hint": (
            "This transaction is part of a cluster of {original:.0f} related entities. "
            "Investigate the shared devices and IPs connecting these customers."
        ),
    },
    "failed_attempt_count": {
        "display_name": "Failed payment attempts",
        "safe_value": 0,
        "investigation_hint": (
            "High number of failed attempts ({original:.0f}) preceding this transaction "
            "suggests possible card testing. Review the sequence of failed payments."
        ),
    },
    "refund_count": {
        "display_name": "Customer refund count",
        "safe_value": 0,
        "investigation_hint": (
            "Customer has {original:.0f} refunds in their history. "
            "Check for refund-abuse patterns or friendly fraud indicators."
        ),
    },
    "amount": {
        "display_name": "Transaction amount",
        "safe_value": None,  # Will be set to merchant's median amount
        "investigation_hint": (
            "Transaction amount (₹{original:,.2f}) is unusual for this merchant. "
            "Verify if this is consistent with the merchant's product pricing."
        ),
    },
    "hour_sin": {
        "display_name": "Transaction hour (cyclical)",
        "safe_value": 0.5,  # ~midday
        "investigation_hint": (
            "Transaction occurred at an unusual time. Review if the merchant "
            "typically sees activity at this hour."
        ),
    },
    "device_velocity_1h": {
        "display_name": "Device velocity (1h)",
        "safe_value": 1,
        "investigation_hint": (
            "This device processed {original:.0f} transactions in 1 hour. "
            "Investigate whether the device is being used for automated/scripted payments."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Counterfactual Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_counterfactuals(
    transaction_id: str,
    features: pd.Series,
    model: Any,
    feature_names: List[str],
    feature_ranges: Optional[Dict[str, Dict]] = None,
    median_amount: Optional[float] = None,
) -> CounterfactualResult:
    """
    Generate counterfactual explanations for a flagged transaction.

    For each feature in PERTURBATION_CONFIG that exists in the feature vector,
    perturb it to the "safe" value, re-score, and record the score delta.

    Parameters
    ----------
    transaction_id : str
        ID of the transaction being explained.
    features : pd.Series
        Feature vector for this transaction (must match model input).
    model : Any
        Trained sklearn-compatible model with .predict_proba().
    feature_names : List[str]
        Ordered list of feature names matching the model's expected input.
    feature_ranges : Optional[Dict[str, Dict]]
        Per-feature {min, max, median} from training data. Used for
        setting "safe" values when not hardcoded.
    median_amount : Optional[float]
        Merchant's median transaction amount, used as "safe" value for amount.

    Returns
    -------
    CounterfactualResult
        Complete counterfactual explanation with perturbations, most impactful
        feature, recommendation, and disclaimer.
    """
    # Get original score
    feature_array = features[feature_names].values.reshape(1, -1)
    original_score = float(model.predict_proba(feature_array)[0, 1])

    perturbations: List[Perturbation] = []

    for feat_name, config in PERTURBATION_CONFIG.items():
        if feat_name not in feature_names:
            continue

        original_value = float(features[feat_name])
        safe_value = config["safe_value"]

        # Handle dynamic safe values
        if safe_value is None:
            if feat_name == "amount" and median_amount is not None:
                safe_value = median_amount
            elif feature_ranges and feat_name in feature_ranges:
                safe_value = feature_ranges[feat_name].get("median", original_value)
            else:
                continue  # Can't perturb without a target

        # Skip if already at the safe value
        if abs(original_value - safe_value) < 1e-6:
            continue

        # Create perturbed feature vector
        perturbed = features[feature_names].copy()
        perturbed[feat_name] = safe_value
        perturbed_array = perturbed.values.reshape(1, -1)

        new_score = float(model.predict_proba(perturbed_array)[0, 1])
        delta = new_score - original_score

        perturbations.append(Perturbation(
            feature=feat_name,
            original_value=original_value,
            perturbed_value=float(safe_value),
            original_score=original_score,
            new_score=new_score,
            delta=delta,
            direction="decrease" if delta < 0 else "increase",
        ))

    # Sort by absolute delta descending — most impactful first
    perturbations.sort(key=lambda p: abs(p.delta), reverse=True)

    # Build recommendation from most impactful
    most_impactful = perturbations[0] if perturbations else None
    recommendation = ""
    if most_impactful and most_impactful.feature in PERTURBATION_CONFIG:
        hint_template = PERTURBATION_CONFIG[most_impactful.feature]["investigation_hint"]
        recommendation = hint_template.format(original=most_impactful.original_value)

    return CounterfactualResult(
        transaction_id=transaction_id,
        original_score=original_score,
        perturbations=perturbations,
        most_impactful=most_impactful,
        recommendation=recommendation,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Batch Counterfactuals
# ═══════════════════════════════════════════════════════════════════════════

def generate_batch_counterfactuals(
    transaction_ids: List[str],
    feature_df: pd.DataFrame,
    model: Any,
    feature_names: List[str],
    feature_ranges: Optional[Dict[str, Dict]] = None,
    median_amounts: Optional[Dict[str, float]] = None,
) -> Dict[str, CounterfactualResult]:
    """
    Generate counterfactuals for a batch of flagged transactions.

    Parameters
    ----------
    transaction_ids : List[str]
        IDs of transactions to explain.
    feature_df : pd.DataFrame
        Full feature matrix, indexed by transaction_id.
    model : Any
        Trained model.
    feature_names : List[str]
        Feature column names.
    feature_ranges : Optional[Dict[str, Dict]]
        Feature range stats.
    median_amounts : Optional[Dict[str, float]]
        Per-merchant median amounts.

    Returns
    -------
    Dict[str, CounterfactualResult]
        Keyed by transaction_id.
    """
    results: Dict[str, CounterfactualResult] = {}

    for tid in transaction_ids:
        if tid not in feature_df.index:
            continue

        features = feature_df.loc[tid]

        # Look up merchant-specific median amount if available
        median_amt = None
        if median_amounts and "merchant_id" in features.index:
            merchant_id = features.get("merchant_id", "")
            median_amt = median_amounts.get(str(merchant_id))

        results[tid] = generate_counterfactuals(
            transaction_id=tid,
            features=features,
            model=model,
            feature_names=feature_names,
            feature_ranges=feature_ranges,
            median_amount=median_amt,
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════════════

def format_counterfactual(result: CounterfactualResult) -> str:
    """Format a counterfactual result as a human-readable explanation."""
    lines = [
        f"Transaction {result.transaction_id} — Risk Score: {result.original_score:.3f}",
        "",
        "Counterfactual Analysis (what-if the input were different):",
        "-" * 60,
    ]

    for p in result.perturbations[:5]:  # Top 5
        display_name = PERTURBATION_CONFIG.get(p.feature, {}).get("display_name", p.feature)
        lines.append(
            f"  If '{display_name}' were {p.perturbed_value:.1f} instead of "
            f"{p.original_value:.1f}: score {p.original_score:.3f} → {p.new_score:.3f} "
            f"(Δ{p.delta:+.3f})"
        )

    if result.recommendation:
        lines.extend(["", "Investigation Recommendation:", f"  {result.recommendation}"])

    lines.extend(["", f"Disclaimer: {result.disclaimer}"])

    return "\n".join(lines)
