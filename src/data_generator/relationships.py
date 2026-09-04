"""
RiskGuard AI — Relationship Builder

Links refunds, disputes, and orders back to transactions for
relational integrity in the synthetic dataset.
"""

from __future__ import annotations

import pandas as pd


def link_relationships(
    transactions_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    refunds_df: pd.DataFrame,
    disputes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Link order/refund/dispute IDs back to the transactions DataFrame.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Base transactions (may already have some nullable FK columns).
    orders_df : pd.DataFrame
        Orders with transaction_id → order_id.
    refunds_df : pd.DataFrame
        Refunds with transaction_id → refund_id.
    disputes_df : pd.DataFrame
        Disputes with transaction_id → dispute_id.

    Returns
    -------
    pd.DataFrame
        Transactions with order_id, refund_id, dispute_id columns populated.
    """
    df = transactions_df.copy()

    # Link orders
    if not orders_df.empty:
        order_map = orders_df.set_index("transaction_id")["order_id"]
        df["order_id"] = df["transaction_id"].map(order_map)

    # Link refunds
    if not refunds_df.empty:
        refund_map = refunds_df.set_index("transaction_id")["refund_id"]
        df["refund_id"] = df["transaction_id"].map(refund_map)

    # Link disputes
    if not disputes_df.empty:
        dispute_map = disputes_df.set_index("transaction_id")["dispute_id"]
        df["dispute_id"] = df["transaction_id"].map(dispute_map)

    return df


def validate_relationships(
    transactions_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    refunds_df: pd.DataFrame,
    disputes_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    merchants_df: pd.DataFrame,
) -> dict:
    """
    Validate referential integrity across all entity tables.

    Returns a summary dict with pass/fail for each check.
    """
    results = {}

    # Check: all customer_ids in transactions exist in customers table
    tx_custs = set(transactions_df["customer_id"].unique())
    known_custs = set(customers_df["customer_id"].unique())
    results["customers_valid"] = tx_custs.issubset(known_custs)
    results["orphan_customers"] = len(tx_custs - known_custs)

    # Check: all merchant_ids in transactions exist in merchants table
    tx_mers = set(transactions_df["merchant_id"].unique())
    known_mers = set(merchants_df["merchant_id"].unique())
    results["merchants_valid"] = tx_mers.issubset(known_mers)
    results["orphan_merchants"] = len(tx_mers - known_mers)

    # Check: all transaction_ids in orders exist in transactions
    if not orders_df.empty:
        order_txs = set(orders_df["transaction_id"].unique())
        known_txs = set(transactions_df["transaction_id"].unique())
        results["orders_valid"] = order_txs.issubset(known_txs)
        results["orphan_orders"] = len(order_txs - known_txs)
    else:
        results["orders_valid"] = True
        results["orphan_orders"] = 0

    # Check: all transaction_ids in refunds exist in transactions
    if not refunds_df.empty:
        refund_txs = set(refunds_df["transaction_id"].unique())
        known_txs = set(transactions_df["transaction_id"].unique())
        results["refunds_valid"] = refund_txs.issubset(known_txs)
        results["orphan_refunds"] = len(refund_txs - known_txs)
    else:
        results["refunds_valid"] = True
        results["orphan_refunds"] = 0

    # Check: all transaction_ids in disputes exist in transactions
    if not disputes_df.empty:
        dispute_txs = set(disputes_df["transaction_id"].unique())
        known_txs = set(transactions_df["transaction_id"].unique())
        results["disputes_valid"] = dispute_txs.issubset(known_txs)
        results["orphan_disputes"] = len(dispute_txs - known_txs)
    else:
        results["disputes_valid"] = True
        results["orphan_disputes"] = 0

    results["all_valid"] = all(
        results[k] for k in results if k.endswith("_valid")
    )

    return results
