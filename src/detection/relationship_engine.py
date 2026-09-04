"""
RiskGuard AI — Relationship / Abuse-Ring Engine

Graph-based detector over shared devices / IPs / customers that catches
threshold-aware attackers who stay under naive velocity limits individually
but form a suspicious structure collectively.

Example: one device used by 4 customers who each individually look fine,
but collectively represent an organized ring.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from src.config import (
    MIN_CLUSTER_SIZE,
    MAX_CUSTOMERS_PER_DEVICE,
    CROSS_MERCHANT_SHARING_FLAG,
)
from src.schemas import RelationshipCluster


# ═══════════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════════

def build_relationship_graph(transactions_df: pd.DataFrame) -> nx.Graph:
    """
    Build a bipartite graph from transactions where:
      - Customer nodes are prefixed with "C:"
      - Device nodes are prefixed with "D:"
      - IP nodes are prefixed with "IP:" (if ip_address column exists)
      - Edges connect customers to the devices/IPs they've transacted from

    Edge weight = number of transactions on that customer-device pair.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Must contain columns: customer_id, device_id.
        Optionally: ip_address, merchant_id, transaction_id.

    Returns
    -------
    nx.Graph
        Bipartite graph with typed nodes and weighted edges.
    """
    G = nx.Graph()

    # Aggregate edges: customer → device
    customer_device = (
        transactions_df.groupby(["customer_id", "device_id"])
        .agg(
            tx_count=("transaction_id", "count"),
            merchants=("merchant_id", lambda x: set(x)),
            transaction_ids=("transaction_id", list),
        )
        .reset_index()
    )

    for _, row in customer_device.iterrows():
        cnode = f"C:{row['customer_id']}"
        dnode = f"D:{row['device_id']}"

        G.add_node(cnode, node_type="customer", entity_id=row["customer_id"])
        G.add_node(dnode, node_type="device", entity_id=row["device_id"])
        G.add_edge(
            cnode, dnode,
            weight=row["tx_count"],
            merchants=row["merchants"],
            transaction_ids=row["transaction_ids"],
        )

    return G


# ═══════════════════════════════════════════════════════════════════════════
# Cluster Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_clusters(
    graph: nx.Graph,
    transactions_df: pd.DataFrame,
    min_size: int = MIN_CLUSTER_SIZE,
    max_customers_per_device: int = MAX_CUSTOMERS_PER_DEVICE,
) -> List[RelationshipCluster]:
    """
    Find suspicious localized clusters in the relationship graph.

    A cluster is flagged as suspicious if:
      1. Any device in the cluster is shared by > max_customers_per_device customers, OR
      2. The cluster exhibits cross-merchant sharing across multiple customers.

    To prevent giant percolation components from masking local fraud rings,
    large connected components are decomposed into localized communities
    via Louvain modularity optimization.

    Parameters
    ----------
    graph : nx.Graph
        The relationship graph from build_relationship_graph().
    transactions_df : pd.DataFrame
        Original transactions for enriching cluster details.
    min_size : int
        Minimum cluster size (customer + device nodes) to flag.
    max_customers_per_device : int
        Flag if any device has more unique customers than this.

    Returns
    -------
    List[RelationshipCluster]
        Suspicious localized clusters, sorted by risk_score descending.
    """
    from networkx.algorithms.community import louvain_communities

    clusters: List[RelationshipCluster] = []
    raw_components: List[Set[str]] = []

    for component_nodes in nx.connected_components(graph):
        if len(component_nodes) > 50:
            subgraph = graph.subgraph(component_nodes)
            try:
                comms = louvain_communities(subgraph, seed=42)
                for comm in comms:
                    if len(comm) >= min_size:
                        raw_components.append(comm)
            except Exception:
                raw_components.append(component_nodes)
        elif len(component_nodes) >= min_size:
            raw_components.append(component_nodes)

    for component_nodes in raw_components:
        subgraph = graph.subgraph(component_nodes)

        # Separate node types
        customer_ids = [
            graph.nodes[n]["entity_id"]
            for n in component_nodes
            if graph.nodes[n].get("node_type") == "customer"
        ]
        device_ids = [
            graph.nodes[n]["entity_id"]
            for n in component_nodes
            if graph.nodes[n].get("node_type") == "device"
        ]

        total_entities = len(customer_ids) + len(device_ids)

        # Check device sharing within this localized cluster
        max_cust_per_dev = _max_customers_per_device(subgraph)

        # Check cross-merchant activity
        all_merchants: Set[str] = set()
        all_transaction_ids: List[str] = []
        for _, _, edata in subgraph.edges(data=True):
            merchants = edata.get("merchants", set())
            all_merchants.update(merchants)
            tx_ids = edata.get("transaction_ids", [])
            all_transaction_ids.extend(tx_ids)

        is_cross_merchant = len(all_merchants) > 1

        # Determine if this localized cluster is genuinely suspicious:
        # Requires either exceeding max_customers_per_device OR cross-merchant multi-customer sharing
        is_suspicious = (
            max_cust_per_dev > max_customers_per_device
            or (CROSS_MERCHANT_SHARING_FLAG and is_cross_merchant and len(customer_ids) > 1 and max_cust_per_dev > 1)
        )

        if not is_suspicious:
            continue

        # Compute risk score for the cluster
        risk_score = _compute_cluster_risk(
            num_customers=len(customer_ids),
            num_devices=len(device_ids),
            max_cust_per_dev=max_cust_per_dev,
            is_cross_merchant=is_cross_merchant,
            total_entities=total_entities,
        )

        clusters.append(RelationshipCluster(
            cluster_id=f"CLU-{uuid.uuid4().hex[:8].upper()}",
            device_ids=device_ids,
            customer_ids=customer_ids,
            transaction_ids=list(set(all_transaction_ids)),  # Deduplicate
            cluster_size=total_entities,
            max_customers_per_device=max_cust_per_dev,
            cross_merchant=is_cross_merchant,
            risk_score=risk_score,
        ))

    # Sort by risk score descending
    clusters.sort(key=lambda c: c.risk_score, reverse=True)
    return clusters


def _max_customers_per_device(subgraph: nx.Graph) -> int:
    """Find the maximum number of unique customers sharing a single device."""
    max_count = 0
    for node in subgraph.nodes():
        if subgraph.nodes[node].get("node_type") == "device":
            # Count customer neighbors
            customer_neighbors = [
                n for n in subgraph.neighbors(node)
                if subgraph.nodes[n].get("node_type") == "customer"
            ]
            max_count = max(max_count, len(customer_neighbors))
    return max_count


def _compute_cluster_risk(
    num_customers: int,
    num_devices: int,
    max_cust_per_dev: int,
    is_cross_merchant: bool,
    total_entities: int,
) -> float:
    """
    Heuristic risk score for a cluster.

    Factors:
      - High customer-to-device ratio (many customers sharing few devices)
      - Large cluster size
      - Cross-merchant sharing
      - Extreme device sharing (one device → many customers)
    """
    # Customer-device ratio: if many customers share few devices → suspicious
    if num_devices > 0:
        ratio_score = min(1.0, (num_customers / num_devices - 1.0) / 3.0)
    else:
        ratio_score = 0.0

    # Cluster size factor: larger clusters → more suspicious
    size_score = min(1.0, (total_entities - 2) / 8.0)

    # Device sharing factor
    sharing_score = min(1.0, (max_cust_per_dev - 1) / 5.0)

    # Cross-merchant bonus
    cross_merchant_score = 0.3 if is_cross_merchant else 0.0

    # Weighted combination
    risk = (
        0.30 * ratio_score
        + 0.25 * size_score
        + 0.30 * sharing_score
        + 0.15 * cross_merchant_score
    )

    return round(min(1.0, max(0.0, risk)), 3)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Detect Rings from Raw Transactions
# ═══════════════════════════════════════════════════════════════════════════

def detect_abuse_rings(
    transactions_df: pd.DataFrame,
    min_size: int = MIN_CLUSTER_SIZE,
    max_customers_per_device: int = MAX_CUSTOMERS_PER_DEVICE,
) -> Tuple[List[RelationshipCluster], nx.Graph]:
    """
    End-to-end ring detection: build graph → find suspicious clusters.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        All transactions to analyze.
    min_size : int
        Minimum cluster size to flag.
    max_customers_per_device : int
        Maximum customers per device before flagging.

    Returns
    -------
    Tuple[List[RelationshipCluster], nx.Graph]
        (suspicious_clusters, full_graph)
    """
    graph = build_relationship_graph(transactions_df)
    clusters = detect_clusters(graph, transactions_df, min_size, max_customers_per_device)
    return clusters, graph
