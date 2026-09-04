"""
RiskGuard AI — End-to-End Pipeline Runner

Generates data → builds features → trains models → runs detection →
produces all benchmarks → saves everything.

Run: python scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Execute the complete RiskGuard AI pipeline."""

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Generate Synthetic Dataset
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 1: Generating synthetic dataset...")
    logger.info("=" * 70)

    from src.data_generator.generate import generate_full_dataset
    dataset = generate_full_dataset()

    train_df = dataset["train"]
    val_df = dataset["validation"]
    test_df = dataset["test"]
    customers_df = dataset["customers"]

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Feature Engineering
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 2: Building feature matrix...")
    logger.info("=" * 70)

    from src.features.pipeline import build_feature_matrix

    train_features, feature_names = build_feature_matrix(train_df, customers_df)
    val_features, _ = build_feature_matrix(val_df, customers_df)
    test_features, _ = build_feature_matrix(test_df, customers_df)

    # Align feature columns
    for col in feature_names:
        if col not in val_features.columns:
            val_features[col] = 0
        if col not in test_features.columns:
            test_features[col] = 0

    # Extract labels
    y_train = train_df.set_index("transaction_id")["is_fraudulent"].astype(int).reindex(train_features.index).fillna(0)
    y_val = val_df.set_index("transaction_id")["is_fraudulent"].astype(int).reindex(val_features.index).fillna(0)
    y_test = test_df.set_index("transaction_id")["is_fraudulent"].astype(int).reindex(test_features.index).fillna(0)

    logger.info(f"Train: {len(y_train)} samples, {y_train.sum()} positive ({y_train.mean():.1%})")
    logger.info(f"Val:   {len(y_val)} samples, {y_val.sum()} positive ({y_val.mean():.1%})")
    logger.info(f"Test:  {len(y_test)} samples, {y_test.sum()} positive ({y_test.mean():.1%})")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Train Models
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 3: Training models...")
    logger.info("=" * 70)

    from src.models.train import train_all_models, evaluate_on_test

    model_results = train_all_models(
        train_features, y_train, val_features, y_val, feature_names
    )

    # Evaluate on held-out test
    test_metrics = evaluate_on_test(model_results, test_features, y_test, feature_names)

    # Get the best model
    best_name = model_results["best_model_name"]
    best_model = model_results[best_name]["model"]
    needs_scaling = model_results[best_name]["needs_scaling"]
    scaler = model_results["scaler"]

    logger.info(f"Best model: {best_name}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Threshold Lab
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 4: Computing threshold curve...")
    logger.info("=" * 70)

    from src.models.threshold_lab import compute_threshold_curve, save_threshold_curve, find_optimal_threshold

    X_test_arr = test_features[feature_names].values
    if needs_scaling:
        X_test_arr = scaler.transform(X_test_arr)
    y_prob_test = best_model.predict_proba(X_test_arr)[:, 1]

    threshold_curve = compute_threshold_curve(
        y_test.values, y_prob_test, test_df.set_index("transaction_id")["amount"].reindex(test_features.index).fillna(0).values
    )
    save_threshold_curve(threshold_curve)

    optimal = find_optimal_threshold(threshold_curve, "f1")
    logger.info(f"Optimal threshold (F1): {optimal.threshold:.3f} → F1={optimal.f1:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Detection Layer
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 5: Running detection layer...")
    logger.info("=" * 70)

    from src.detection.merchant_baseline import compute_baselines
    from src.detection.spike_detector import detect_spikes
    from src.detection.relationship_engine import detect_abuse_rings
    from src.detection.adversarial_demo import (
        run_naive_rule, run_riskguard_detection,
        run_adversarial_comparison, format_adversarial_report,
    )

    # Compute baselines from training data
    baselines = compute_baselines(train_df)
    logger.info(f"Computed baselines for {len(baselines)} merchants")

    # Run spike detection on test data
    risk_scores_series = pd.Series(y_prob_test, index=test_features.index)
    incidents = detect_spikes(test_df, baselines, risk_scores=risk_scores_series)
    logger.info(f"Detected {len(incidents)} incidents")

    # Run relationship engine
    clusters, graph = detect_abuse_rings(test_df)
    logger.info(f"Found {len(clusters)} suspicious clusters")

    # Adversarial comparison (pass .values to match training format)
    adversarial_result = run_adversarial_comparison(test_df, baselines, best_model, test_features[feature_names].values)
    logger.info("\n" + format_adversarial_report(adversarial_result))

    # Save adversarial results
    import dataclasses
    with open(MODELS_DIR / "adversarial_results.json", "w") as f:
        json.dump(dataclasses.asdict(adversarial_result), f, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Explainability
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 6: Generating explanations...")
    logger.info("=" * 70)

    from src.explainability.exposure import calculate_per_merchant_exposure, calculate_total_exposure

    # Score test data — map risk scores from feature matrix index to test_df transaction_id
    scored_test = test_df.copy()
    risk_score_map = dict(zip(test_features.index, y_prob_test))
    scored_test["risk_score"] = scored_test["transaction_id"].map(risk_score_map).fillna(0.0)

    # Calculate exposure
    total_exposure = calculate_total_exposure(scored_test)
    per_merchant_exposure = calculate_per_merchant_exposure(scored_test)
    logger.info(f"Total potential exposure: ₹{total_exposure.potential_exposure:,.2f}")
    logger.info(f"High-confidence exposure: ₹{total_exposure.high_confidence_exposure:,.2f}")

    # Counterfactuals for top incidents
    from src.explainability.counterfactual import generate_counterfactuals, format_counterfactual

    counterfactuals = {}
    for inc in incidents[:3]:  # Top 3 incidents
        if inc.flagged_transaction_ids:
            tid = inc.flagged_transaction_ids[0]
            if tid in test_features.index:
                cf = generate_counterfactuals(
                    tid, test_features.loc[tid][feature_names], best_model, feature_names
                )
                counterfactuals[tid] = cf
                logger.info("\n" + format_counterfactual(cf))

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Cost-of-Inaction Benchmark
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 7: Cost-of-inaction benchmark...")
    logger.info("=" * 70)

    from src.benchmarks.cost_of_inaction import compute_cost_of_inaction, save_cost_of_inaction

    naive_flags = run_naive_rule(test_df)
    riskguard_flags = run_riskguard_detection(test_df, best_model, test_features[feature_names].values)

    cost_conditions = compute_cost_of_inaction(test_df, naive_flags, riskguard_flags)
    save_cost_of_inaction(cost_conditions)

    # ══════════════════════════════════════════════════════════════════
    # STEP 8: Drift / Re-baselining
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 8: Drift re-baselining simulation...")
    logger.info("=" * 70)

    from src.benchmarks.drift_rebaseline import simulate_drift_rebaseline, format_drift_timeline

    # Find a post-incident merchant
    post_incident_merchants = dataset["merchants"][
        dataset["merchants"]["scenario_type"] == "post_incident"
    ]
    if not post_incident_merchants.empty:
        drift_merchant = post_incident_merchants.iloc[0]["merchant_id"]
        drift_baseline = baselines.get(drift_merchant)
        if drift_baseline:
            incident_tx = train_df[
                (train_df["merchant_id"] == drift_merchant) &
                (train_df["is_fraudulent"])
            ]
            post_tx = test_df[test_df["merchant_id"] == drift_merchant]
            drift_days = simulate_drift_rebaseline(
                drift_merchant, incident_tx, post_tx, drift_baseline
            )
            logger.info("\n" + format_drift_timeline(drift_days))

            # Save drift results
            drift_data = [
                {
                    "day": d.day,
                    "date": str(d.date.date()),
                    "status": d.status,
                    "baseline_risk_rate": d.baseline_risk_rate,
                    "current_risk_rate": d.current_risk_rate,
                    "validation_passed": d.validation_passed,
                }
                for d in drift_days
            ]
            with open(MODELS_DIR / "drift_results.json", "w") as f:
                json.dump(drift_data, f, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # STEP 9: Hard Failure Case (Flash Sale)
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 9: Flash sale analysis...")
    logger.info("=" * 70)

    from src.benchmarks.hard_failure import analyze_flash_sale

    flash_sale_df = test_df[test_df["scenario_type"] == "flash_sale"]
    if not flash_sale_df.empty:
        flash_features = test_features.reindex(flash_sale_df["transaction_id"]).dropna()
        if not flash_features.empty:
            flash_analysis = analyze_flash_sale(
                flash_sale_df, best_model, flash_features, feature_names, baselines
            )
            logger.info("\n" + flash_analysis["write_up"])
            with open(MODELS_DIR / "flash_sale_analysis.json", "w") as f:
                # Filter serializable parts
                serializable = {k: v for k, v in flash_analysis.items() if k != "write_up"}
                serializable["write_up"] = flash_analysis["write_up"]
                json.dump(serializable, f, indent=2, default=str)

    # ══════════════════════════════════════════════════════════════════
    # STEP 10: Latency Benchmark
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 10: Latency benchmark...")
    logger.info("=" * 70)

    from src.benchmarks.latency_throughput import measure_latency

    latency = measure_latency(
        model=best_model,
        feature_matrix=test_features,
        feature_names=feature_names,
        scaler=scaler if needs_scaling else None,
        needs_scaling=needs_scaling,
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 11: Save Summary State for API
    # ══════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 11: Saving pipeline state...")
    logger.info("=" * 70)

    # Save incidents
    incidents_data = []
    for inc in incidents:
        incidents_data.append({
            "incident_id": inc.incident_id,
            "merchant_id": inc.merchant_id,
            "trigger_type": inc.trigger_type,
            "risk_level": inc.risk_level,
            "risk_increase_pct": inc.risk_increase_pct,
            "affected_payment_count": inc.affected_payment_count,
            "estimated_exposure": inc.estimated_exposure,
            "high_confidence_exposure": inc.high_confidence_exposure,
            "confidence": inc.confidence,
            "detected_at": str(inc.detected_at),
            "status": inc.status,
            "flagged_transaction_ids": inc.flagged_transaction_ids[:20],  # Limit for JSON
            "related_entity_ids": inc.related_entity_ids[:20],
        })
    with open(MODELS_DIR / "incidents.json", "w") as f:
        json.dump(incidents_data, f, indent=2)

    # Save exposure
    exposure_data = {
        "total": {
            "potential_exposure": total_exposure.potential_exposure,
            "high_confidence_exposure": total_exposure.high_confidence_exposure,
            "flagged_count": total_exposure.flagged_count,
            "high_confidence_count": total_exposure.high_confidence_count,
        },
        "per_merchant": {
            mid: {
                "potential_exposure": exp.potential_exposure,
                "high_confidence_exposure": exp.high_confidence_exposure,
                "flagged_count": exp.flagged_count,
            }
            for mid, exp in per_merchant_exposure.items()
        },
    }
    with open(MODELS_DIR / "exposure.json", "w") as f:
        json.dump(exposure_data, f, indent=2)

    # Save dashboard metrics
    dashboard_metrics = {
        "total_transactions": len(test_df),
        "active_alerts": len(incidents),
        "total_exposure": total_exposure.potential_exposure,
        "high_confidence_exposure": total_exposure.high_confidence_exposure,
        "current_risk_rate": round(float(test_df["is_fraudulent"].mean()), 4),
        "flagged_count": total_exposure.flagged_count,
        "best_model": best_name,
        "model_pr_auc": test_metrics[best_name]["pr_auc"],
    }
    with open(MODELS_DIR / "dashboard_metrics.json", "w") as f:
        json.dump(dashboard_metrics, f, indent=2)

    # Save risk trend (daily aggregation)
    test_df_copy = test_df.copy()
    test_df_copy["date"] = pd.to_datetime(test_df_copy["timestamp"]).dt.date
    risk_trend = test_df_copy.groupby("date").agg(
        risk_rate=("is_fraudulent", "mean"),
        volume=("transaction_id", "count"),
        total_amount=("amount", "sum"),
    ).reset_index()
    risk_trend["date"] = risk_trend["date"].astype(str)
    risk_trend_data = risk_trend.to_dict("records")
    with open(MODELS_DIR / "risk_trend.json", "w") as f:
        json.dump(risk_trend_data, f, indent=2)

    # Save evaluation matrix
    eval_matrix = {
        "ml": test_metrics[best_name],
        "financial": {
            "fp_cost": cost_conditions[2].fp_cost,  # RiskGuard
            "fn_cost": cost_conditions[2].fn_cost,
            "expected_loss": cost_conditions[2].total_loss,
            "potential_exposure": total_exposure.potential_exposure,
        },
        "robustness": dataclasses.asdict(adversarial_result),
        "operational": {
            "detection_latency_ms": latency.ml_inference_ms,
            "events_per_sec": latency.events_per_sec,
            "investigation_time_ms": latency.agent_investigation_ms,
        },
        "explainability": {
            "counterfactual_coverage": len(counterfactuals) / max(len(incidents), 1),
            "avg_perturbations": np.mean([len(cf.perturbations) for cf in counterfactuals.values()]) if counterfactuals else 0,
        },
    }
    with open(MODELS_DIR / "evaluation_matrix.json", "w") as f:
        json.dump(eval_matrix, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best model: {best_name} (PR-AUC: {test_metrics[best_name]['pr_auc']:.4f})")
    logger.info(f"Incidents detected: {len(incidents)}")
    logger.info(f"Suspicious clusters: {len(clusters)}")
    logger.info(f"Total exposure: ₹{total_exposure.potential_exposure:,.2f}")
    logger.info(f"Latency: {latency.ml_inference_ms:.1f}ms ML, {latency.events_per_sec:.0f} events/sec")


if __name__ == "__main__":
    run_full_pipeline()
