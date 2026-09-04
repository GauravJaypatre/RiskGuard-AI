"""
RiskGuard AI — Shared Data Schemas

All dataclasses used across tracks. These are the canonical interface contracts.
Every component reads/writes these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Entity Schemas (used by data generator, features, API)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Merchant:
    """A merchant on the Razorpay platform."""
    merchant_id: str
    name: str
    category: str                   # ecommerce, subscription, digital_services, marketplace
    registration_date: datetime
    avg_monthly_volume: float       # Expected monthly transaction count


@dataclass
class Customer:
    """An end-customer making payments through a merchant."""
    customer_id: str
    merchant_id: str
    email: str
    account_created: datetime
    is_synthetic_fraud: bool        # Ground-truth label for synthetic data


@dataclass
class Device:
    """A device fingerprint associated with transactions."""
    device_id: str
    device_type: str                # mobile, desktop, tablet
    os: str
    browser: str
    fingerprint: str


@dataclass
class Transaction:
    """A payment transaction with full relational context."""
    transaction_id: str
    merchant_id: str
    customer_id: str
    device_id: str
    amount: float
    currency: str
    payment_method: str             # card, upi, netbanking, wallet
    status: str                     # success, failed, disputed
    is_fraudulent: bool             # Ground-truth label
    scenario_type: str              # normal, fraud_spike, slow_ring, device_cluster,
                                    # amount_manipulation, flash_sale, post_incident
    timestamp: datetime
    ip_address: str
    order_id: Optional[str] = None
    refund_id: Optional[str] = None
    dispute_id: Optional[str] = None


@dataclass
class Order:
    """An order linked to a payment transaction."""
    order_id: str
    transaction_id: str
    merchant_id: str
    customer_id: str
    amount: float
    status: str                     # created, paid, fulfilled, cancelled
    created_at: datetime


@dataclass
class Refund:
    """A refund issued against a transaction."""
    refund_id: str
    transaction_id: str
    merchant_id: str
    customer_id: str
    amount: float
    reason: str
    status: str                     # initiated, processed, failed
    created_at: datetime


@dataclass
class Dispute:
    """A dispute / chargeback raised on a transaction."""
    dispute_id: str
    transaction_id: str
    merchant_id: str
    customer_id: str
    amount: float
    reason: str
    status: str                     # open, won, lost
    created_at: datetime


# ═══════════════════════════════════════════════════════════════════════════
# Detection Output Schemas
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MerchantBaseline:
    """A merchant's learned normal behavior over a baseline period."""
    merchant_id: str
    period_start: datetime
    period_end: datetime
    normal_volume: float            # Average daily transaction count
    normal_risk_rate: float         # fraud_tx / total_tx
    normal_failure_rate: float      # failed_tx / total_tx
    normal_refund_rate: float       # refund_tx / total_tx
    volume_std: float               # Std deviation of daily volume
    risk_rate_std: float
    failure_rate_std: float
    refund_rate_std: float


@dataclass
class Incident:
    """A detected risk incident for a merchant."""
    incident_id: str
    merchant_id: str
    trigger_type: str               # spike, ring, device_cluster, amount_pattern
    risk_increase_pct: float        # Percentage increase over baseline
    affected_payment_count: int
    estimated_exposure: float       # Sum of flagged transaction amounts
    high_confidence_exposure: float
    confidence: float               # 0–1 confidence in the detection
    risk_level: str                 # LOW, MEDIUM, HIGH, CRITICAL
    flagged_transaction_ids: List[str] = field(default_factory=list)
    related_entity_ids: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)
    status: str = "active"          # active, investigating, resolved


@dataclass
class RelationshipCluster:
    """A suspicious cluster of related entities found by the relationship engine."""
    cluster_id: str
    device_ids: List[str] = field(default_factory=list)
    customer_ids: List[str] = field(default_factory=list)
    transaction_ids: List[str] = field(default_factory=list)
    cluster_size: int = 0
    max_customers_per_device: int = 0
    cross_merchant: bool = False
    risk_score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Explainability Schemas
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Perturbation:
    """A single feature perturbation and its effect on model score."""
    feature: str
    original_value: float
    perturbed_value: float
    original_score: float
    new_score: float
    delta: float                    # new_score - original_score
    direction: str                  # "decrease" or "increase"


@dataclass
class CounterfactualResult:
    """Counterfactual explanation for a flagged transaction."""
    transaction_id: str
    original_score: float
    perturbations: List[Perturbation] = field(default_factory=list)
    most_impactful: Optional[Perturbation] = None
    recommendation: str = ""        # Investigation recommendation tied to top counterfactual
    disclaimer: str = (
        "These counterfactual explanations describe how the model's output would change "
        "if specific inputs were different. They reflect model sensitivity, not causal claims "
        "about real-world fraud."
    )


@dataclass
class ExposureResult:
    """Financial exposure estimate for a merchant."""
    merchant_id: str
    potential_exposure: float        # Sum of all flagged transaction amounts
    high_confidence_exposure: float  # Sum of tx scoring > high threshold
    flagged_count: int
    high_confidence_count: int
    currency: str = "INR"
    computed_at: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════
# Agent / Audit Schemas
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyResult:
    """Policy evaluation result for a risk level."""
    risk_level: str
    allowed_actions: List[str] = field(default_factory=list)
    recommendation: str = ""
    requires_human_authorization: bool = False


@dataclass
class InvestigationReport:
    """Structured report produced by the investigation agent."""
    incident_id: str
    agent_mode: str                  # "live" | "mock" — ALWAYS SURFACED, never hidden
    mock_reason: Optional[str] = None # None | "rate_limited" | "no_api_key" | "invalid_key" | "high_demand" | "api_error"
    summary: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    related_entities: List[Dict[str, Any]] = field(default_factory=list)
    exposure: Optional[ExposureResult] = None
    counterfactuals: Optional[CounterfactualResult] = None
    risk_assessment: str = ""
    recommendation: str = ""
    policy_result: Optional[PolicyResult] = None
    confidence_notes: str = ""


@dataclass
class CaseRecord:
    """Full audit trail for an investigated incident."""
    case_id: str
    timestamp: datetime
    merchant_id: str
    incident_id: str
    trigger_type: str
    risk_score: float
    model_version: str
    evidence_summary: str
    related_entities: List[str] = field(default_factory=list)
    exposure_estimate: Optional[ExposureResult] = None
    counterfactual_summary: Optional[CounterfactualResult] = None
    agent_recommendation: str = ""
    agent_mode: str = ""             # "live" | "mock" — audit trail always records this
    policy_result: str = ""
    human_decision: Optional[str] = None     # None until human acts
    final_outcome: Optional[str] = None      # None until resolved


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Schemas
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DriftDay:
    """Single day in a drift / re-baselining simulation."""
    day: int
    date: datetime
    status: str                      # CRITICAL, MONITOR, NEW_NORMAL
    baseline_risk_rate: float
    current_risk_rate: float
    validation_passed: Optional[bool] = None


@dataclass
class AdversarialResult:
    """Comparison of naive rules vs RiskGuard on adversarial scenarios."""
    naive_rule_detection_rate: Dict[str, float] = field(default_factory=dict)
    riskguard_detection_rate: Dict[str, float] = field(default_factory=dict)
    naive_rule_fp_rate: float = 0.0
    riskguard_fp_rate: float = 0.0
    exposure_missed_naive: float = 0.0
    exposure_missed_riskguard: float = 0.0


@dataclass
class ThresholdPoint:
    """Single point on the threshold sweep curve."""
    threshold: float
    precision: float
    recall: float
    f1: float
    fp_count: int
    fn_count: int
    expected_loss: float


@dataclass
class CostCondition:
    """A single condition in the cost-of-inaction benchmark."""
    name: str                        # "no_system", "static_rules", "riskguard"
    total_loss: float
    fp_cost: float
    fn_cost: float
    detected_fraud_pct: float


@dataclass
class LatencyResult:
    """Measured latency and throughput numbers."""
    ml_inference_ms: float
    agent_investigation_ms: float
    end_to_end_ms: float
    events_per_sec: float
    batch_size: int


@dataclass
class EvaluationMatrix:
    """Full evaluation matrix — the final deliverable."""
    # ML metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    pr_auc: float = 0.0

    # Financial metrics
    fp_cost: float = 0.0
    fn_cost: float = 0.0
    expected_loss: float = 0.0
    potential_exposure: float = 0.0

    # Robustness
    adversarial: Optional[AdversarialResult] = None

    # Operational
    detection_latency_ms: float = 0.0
    events_per_sec: float = 0.0
    investigation_time_ms: float = 0.0

    # Explainability
    counterfactual_coverage: float = 0.0
    avg_perturbations: float = 0.0
