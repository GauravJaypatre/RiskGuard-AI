"""
RiskGuard AI — Agent Tool Functions

These 10 functions are the tools the investigation agent can call.
The agent does NOT compute fraud scores itself — it reasons over evidence
that the ML/rules layers already produced.

Each tool takes simple arguments and returns JSON-serializable dicts,
suitable for the Gemini function-calling protocol.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src.schemas import (
    MerchantBaseline,
    ExposureResult,
    CounterfactualResult,
    Incident,
    RelationshipCluster,
)


class AgentToolkit:
    """
    Encapsulates all 10 investigation tools.

    Initialized with pre-computed data so the agent can query it.
    The agent never sees raw DataFrames — only JSON-safe dicts.
    """

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        baselines: Dict[str, MerchantBaseline],
        risk_scores: Dict[str, float],
        feature_importances: Dict[str, Dict[str, float]],
        clusters: List[RelationshipCluster],
        incidents: List[Incident],
        exposures: Dict[str, ExposureResult],
        counterfactuals: Dict[str, CounterfactualResult],
        cases: Optional[Dict[str, Dict]] = None,
    ):
        self.transactions_df = transactions_df
        self.baselines = baselines
        self.risk_scores = risk_scores
        self.feature_importances = feature_importances
        self.clusters = clusters
        self.incidents = {inc.incident_id: inc for inc in incidents}
        self.exposures = exposures
        self.counterfactuals = counterfactuals
        self.cases = cases or {}

        # Build lookup indices
        self._tx_index = transactions_df.set_index("transaction_id")

    # ───────────────────────────────────────────────────────────────────
    # Tool 1: get_transaction
    # ───────────────────────────────────────────────────────────────────
    def get_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Get full transaction details + risk score."""
        if transaction_id not in self._tx_index.index:
            return {"error": f"Transaction {transaction_id} not found"}

        row = self._tx_index.loc[transaction_id]
        return {
            "transaction_id": transaction_id,
            "merchant_id": str(row.get("merchant_id", "")),
            "customer_id": str(row.get("customer_id", "")),
            "device_id": str(row.get("device_id", "")),
            "amount": float(row.get("amount", 0)),
            "currency": "INR",
            "payment_method": str(row.get("payment_method", "")),
            "status": str(row.get("status", "")),
            "timestamp": str(row.get("timestamp", "")),
            "ip_address": str(row.get("ip_address", "")),
            "risk_score": self.risk_scores.get(transaction_id, 0.0),
        }

    # ───────────────────────────────────────────────────────────────────
    # Tool 2: get_customer_history
    # ───────────────────────────────────────────────────────────────────
    def get_customer_history(self, customer_id: str) -> Dict[str, Any]:
        """Get customer's recent transaction history, refund/dispute summary."""
        cdf = self.transactions_df[self.transactions_df["customer_id"] == customer_id]

        if cdf.empty:
            return {"error": f"Customer {customer_id} not found"}

        total_tx = len(cdf)
        total_amount = float(cdf["amount"].sum())
        avg_amount = float(cdf["amount"].mean())
        failed = int((cdf["status"] == "failed").sum())
        refunds = int(cdf["refund_id"].notna().sum())
        disputes = int(cdf["dispute_id"].notna().sum())
        fraud_count = int(cdf["is_fraudulent"].sum()) if "is_fraudulent" in cdf.columns else 0

        # Recent transactions (last 10)
        recent = cdf.sort_values("timestamp", ascending=False).head(10)[
            ["transaction_id", "amount", "status", "timestamp", "device_id"]
        ].to_dict("records")

        for r in recent:
            r["timestamp"] = str(r["timestamp"])
            r["risk_score"] = self.risk_scores.get(r["transaction_id"], 0.0)

        return {
            "customer_id": customer_id,
            "total_transactions": total_tx,
            "total_amount": total_amount,
            "avg_amount": avg_amount,
            "failed_attempts": failed,
            "refund_count": refunds,
            "dispute_count": disputes,
            "flagged_transactions": fraud_count,
            "recent_transactions": recent,
        }

    # ───────────────────────────────────────────────────────────────────
    # Tool 3: get_device_history
    # ───────────────────────────────────────────────────────────────────
    def get_device_history(self, device_id: str) -> Dict[str, Any]:
        """Get device's transaction history and unique customer count."""
        ddf = self.transactions_df[self.transactions_df["device_id"] == device_id]

        if ddf.empty:
            return {"error": f"Device {device_id} not found"}

        unique_customers = ddf["customer_id"].nunique()
        unique_merchants = ddf["merchant_id"].nunique()
        total_tx = len(ddf)

        # Recent transactions
        recent = ddf.sort_values("timestamp", ascending=False).head(10)[
            ["transaction_id", "customer_id", "amount", "status", "timestamp"]
        ].to_dict("records")

        for r in recent:
            r["timestamp"] = str(r["timestamp"])

        return {
            "device_id": device_id,
            "unique_customers": unique_customers,
            "unique_merchants": unique_merchants,
            "total_transactions": total_tx,
            "suspicious": unique_customers > 3,
            "recent_transactions": recent,
        }

    # ───────────────────────────────────────────────────────────────────
    # Tool 4: get_merchant_baseline
    # ───────────────────────────────────────────────────────────────────
    def get_merchant_baseline(self, merchant_id: str) -> Dict[str, Any]:
        """Get merchant's baseline stats and current deviation."""
        baseline = self.baselines.get(merchant_id)
        if not baseline:
            return {"error": f"No baseline for merchant {merchant_id}"}

        return {
            "merchant_id": merchant_id,
            "baseline_period": f"{baseline.period_start} to {baseline.period_end}",
            "normal_daily_volume": baseline.normal_volume,
            "normal_risk_rate": round(baseline.normal_risk_rate, 4),
            "normal_failure_rate": round(baseline.normal_failure_rate, 4),
            "normal_refund_rate": round(baseline.normal_refund_rate, 4),
            "volume_std": round(baseline.volume_std, 2),
            "risk_rate_std": round(baseline.risk_rate_std, 4),
        }

    # ───────────────────────────────────────────────────────────────────
    # Tool 5: find_related_activity
    # ───────────────────────────────────────────────────────────────────
    def find_related_activity(self, entity_id: str) -> Dict[str, Any]:
        """Find related entities (shared devices, customers, IPs) via cluster membership."""
        related_clusters = []
        for cluster in self.clusters:
            if (entity_id in cluster.customer_ids
                    or entity_id in cluster.device_ids):
                related_clusters.append({
                    "cluster_id": cluster.cluster_id,
                    "cluster_size": cluster.cluster_size,
                    "sample_customers": cluster.customer_ids[:10],
                    "total_customers": len(cluster.customer_ids),
                    "sample_devices": cluster.device_ids[:10],
                    "total_devices": len(cluster.device_ids),
                    "max_customers_per_device": cluster.max_customers_per_device,
                    "cross_merchant": cluster.cross_merchant,
                    "risk_score": cluster.risk_score,
                    "transaction_count": len(cluster.transaction_ids),
                })

        return {
            "entity_id": entity_id,
            "related_clusters": related_clusters,
            "total_clusters": len(related_clusters),
        }

    # ───────────────────────────────────────────────────────────────────
    # Tool 6: calculate_velocity
    # ───────────────────────────────────────────────────────────────────
    def calculate_velocity(self, entity_id: str, window: str = "1h") -> Dict[str, Any]:
        """Calculate transaction velocity for an entity over a time window."""
        window_map = {"5m": 5, "30m": 30, "1h": 60, "24h": 1440}
        minutes = window_map.get(window, 60)

        # Try as customer first, then device
        mask = (self.transactions_df["customer_id"] == entity_id)
        entity_type = "customer"
        if not mask.any():
            mask = (self.transactions_df["device_id"] == entity_id)
            entity_type = "device"

        edf = self.transactions_df[mask].copy()
        if edf.empty:
            return {"error": f"Entity {entity_id} not found"}

        edf["timestamp"] = pd.to_datetime(edf["timestamp"])
        latest = edf["timestamp"].max()
        window_start = latest - timedelta(minutes=minutes)
        window_df = edf[edf["timestamp"] >= window_start]

        return {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "window": window,
            "transaction_count": len(window_df),
            "total_amount": float(window_df["amount"].sum()),
            "window_start": str(window_start),
            "window_end": str(latest),
        }

    # ───────────────────────────────────────────────────────────────────
    # Tool 7: calculate_exposure
    # ───────────────────────────────────────────────────────────────────
    def calculate_exposure(self, merchant_id: str) -> Dict[str, Any]:
        """Get pre-computed exposure for a merchant."""
        exposure = self.exposures.get(merchant_id)
        if not exposure:
            return {"error": f"No exposure data for merchant {merchant_id}"}

        return {
            "merchant_id": merchant_id,
            "potential_exposure": exposure.potential_exposure,
            "high_confidence_exposure": exposure.high_confidence_exposure,
            "flagged_count": exposure.flagged_count,
            "high_confidence_count": exposure.high_confidence_count,
            "currency": exposure.currency,
            "note": "These figures represent potential exposure identified, not fraud prevented.",
        }

    # ───────────────────────────────────────────────────────────────────
    # Tool 8: get_risk_score
    # ───────────────────────────────────────────────────────────────────
    def get_risk_score(self, transaction_id: str) -> Dict[str, Any]:
        """Get model risk score and top feature importances for a transaction."""
        score = self.risk_scores.get(transaction_id)
        if score is None:
            return {"error": f"No risk score for transaction {transaction_id}"}

        importances = self.feature_importances.get(transaction_id, {})
        top_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        return {
            "transaction_id": transaction_id,
            "risk_score": score,
            "risk_level": _score_to_level(score),
            "top_contributing_features": [
                {"feature": f, "importance": round(v, 4)} for f, v in top_features
            ],
        }

    # ───────────────────────────────────────────────────────────────────
    # Tool 9: get_policy
    # ───────────────────────────────────────────────────────────────────
    def get_policy(self, risk_level: str) -> Dict[str, Any]:
        """Get allowed actions and recommendation for a risk level."""
        from src.agent.policy import evaluate_policy
        result = evaluate_policy(risk_level)
        return {
            "risk_level": result.risk_level,
            "allowed_actions": result.allowed_actions,
            "recommendation": result.recommendation,
            "requires_human_authorization": result.requires_human_authorization,
        }

    # ───────────────────────────────────────────────────────────────────
    # Tool 10: create_case
    # ───────────────────────────────────────────────────────────────────
    def create_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Persist an audit record and return the case ID."""
        from src.agent.audit import create_case_from_dict
        case = create_case_from_dict(case_data)
        self.cases[case.case_id] = case
        return {
            "case_id": case.case_id,
            "status": "created",
            "timestamp": str(case.timestamp),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Tool Schema Definitions (Gemini function-calling format)
# ═══════════════════════════════════════════════════════════════════════════

def _build_gemini_tool_declarations():
    """
    Build Gemini FunctionDeclaration objects for the 10 agent tools.

    Uses google.genai.types.FunctionDeclaration with parameters_json_schema
    (standard JSON Schema format).

    Lazy-imported so the module loads even if google-genai isn't installed
    (e.g. during pipeline-only runs that never invoke the agent).
    """
    from google.genai import types

    return [
        types.FunctionDeclaration(
            name="get_transaction",
            description="Get full details and risk score for a specific transaction.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string", "description": "The transaction ID to look up."}
                },
                "required": ["transaction_id"],
            },
        ),
        types.FunctionDeclaration(
            name="get_customer_history",
            description="Get a customer's transaction history, including refund and dispute counts, failed attempts, and recent transactions with risk scores.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The customer ID to look up."}
                },
                "required": ["customer_id"],
            },
        ),
        types.FunctionDeclaration(
            name="get_device_history",
            description="Get a device's transaction history and the number of unique customers who have used it.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "The device ID to look up."}
                },
                "required": ["device_id"],
            },
        ),
        types.FunctionDeclaration(
            name="get_merchant_baseline",
            description="Get a merchant's learned normal behavior (baseline daily volume, risk rate, failure rate, refund rate) and standard deviations.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "merchant_id": {"type": "string", "description": "The merchant ID to look up."}
                },
                "required": ["merchant_id"],
            },
        ),
        types.FunctionDeclaration(
            name="find_related_activity",
            description="Find clusters of related entities (shared devices, customers, IPs) that an entity belongs to. Useful for investigating potential fraud rings.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "A customer ID or device ID to search for."}
                },
                "required": ["entity_id"],
            },
        ),
        types.FunctionDeclaration(
            name="calculate_velocity",
            description="Calculate how many transactions an entity (customer or device) made within a given time window.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "Customer or device ID."},
                    "window": {"type": "string", "enum": ["5m", "30m", "1h", "24h"], "description": "Time window."},
                },
                "required": ["entity_id", "window"],
            },
        ),
        types.FunctionDeclaration(
            name="calculate_exposure",
            description="Get the estimated financial exposure for a merchant — both potential and high-confidence exposure amounts.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "merchant_id": {"type": "string", "description": "The merchant ID."}
                },
                "required": ["merchant_id"],
            },
        ),
        types.FunctionDeclaration(
            name="get_risk_score",
            description="Get the ML model's risk score for a transaction, along with the top contributing features.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string", "description": "The transaction ID."}
                },
                "required": ["transaction_id"],
            },
        ),
        types.FunctionDeclaration(
            name="get_policy",
            description="Get the allowed actions and recommendation for a given risk level (LOW, MEDIUM, HIGH, CRITICAL).",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"], "description": "The risk level."}
                },
                "required": ["risk_level"],
            },
        ),
        types.FunctionDeclaration(
            name="create_case",
            description="Create an audit case record for an investigated incident.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "case_data": {
                        "type": "object",
                        "description": "Case data including merchant_id, incident_id, evidence_summary, recommendation.",
                    }
                },
                "required": ["case_data"],
            },
        ),
    ]


def _score_to_level(score: float) -> str:
    """Convert risk score to risk level."""
    from src.config import RISK_LEVEL_THRESHOLDS
    for level, (low, high) in RISK_LEVEL_THRESHOLDS.items():
        if low <= score < high:
            return level
    return "CRITICAL"
