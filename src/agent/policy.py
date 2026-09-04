"""
RiskGuard AI — Policy / Guardrail Layer

Bounds the agent's output to recommendations only. The agent never takes
autonomous account/payment actions.

    LOW      → Continue monitoring
    MEDIUM   → Enhanced verification recommended
    HIGH     → Merchant review recommended
    CRITICAL → Restrictive action recommendation (REQUIRES human authorization)
"""

from __future__ import annotations

from src.config import POLICY_ACTIONS, RISK_LEVEL_THRESHOLDS
from src.schemas import PolicyResult


def evaluate_policy(risk_level: str) -> PolicyResult:
    """
    Map a risk level to its policy-bounded response.

    Parameters
    ----------
    risk_level : str
        One of "LOW", "MEDIUM", "HIGH", "CRITICAL".

    Returns
    -------
    PolicyResult
        Allowed actions, recommendation text, and whether human auth is required.
    """
    risk_level = risk_level.upper()

    if risk_level not in POLICY_ACTIONS:
        # Default to most restrictive if unknown level
        risk_level = "CRITICAL"

    config = POLICY_ACTIONS[risk_level]

    return PolicyResult(
        risk_level=risk_level,
        allowed_actions=config["actions"],
        recommendation=config["recommendation"],
        requires_human_authorization=config["requires_human_authorization"],
    )


def score_to_risk_level(score: float) -> str:
    """
    Convert a numeric risk score (0–1) to a risk level string.

    Parameters
    ----------
    score : float
        Risk score from the ML model.

    Returns
    -------
    str
        Risk level: LOW, MEDIUM, HIGH, or CRITICAL.
    """
    for level, (low, high) in RISK_LEVEL_THRESHOLDS.items():
        if low <= score < high:
            return level
    return "CRITICAL"


def is_action_allowed(risk_level: str, action: str) -> bool:
    """
    Check if a specific action is allowed at a given risk level.

    Parameters
    ----------
    risk_level : str
        The risk level.
    action : str
        The action to check.

    Returns
    -------
    bool
        True if the action is in the allowed list for this level.
    """
    risk_level = risk_level.upper()
    config = POLICY_ACTIONS.get(risk_level, POLICY_ACTIONS["CRITICAL"])
    return action in config["actions"]


def requires_human_auth(risk_level: str) -> bool:
    """Check if a risk level requires human authorization."""
    risk_level = risk_level.upper()
    config = POLICY_ACTIONS.get(risk_level, POLICY_ACTIONS["CRITICAL"])
    return config["requires_human_authorization"]
