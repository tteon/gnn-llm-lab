"""
Link prediction evaluation metrics.

- compute_link_prediction_metrics: MRR and Hits@K for GNN link prediction
- format_metrics: pretty-print metric dict
"""

from typing import Dict

import torch

from src.utils import get_logger

logger = get_logger("evaluation")


def compute_link_prediction_metrics(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    k_values: tuple = (1, 3, 10),
) -> Dict[str, float]:
    """Compute MRR and Hits@K for link prediction.

    For each positive edge, rank it against all negative edges.

    Args:
        pos_scores: [num_pos] logit scores for positive edges
        neg_scores: [num_neg] logit scores for negative edges
        k_values: tuple of K values for Hits@K

    Returns:
        dict with 'mrr', 'hits@1', 'hits@3', 'hits@10', etc.
    """
    # For each positive score, count how many negative scores are >= it
    # pos_scores: [P], neg_scores: [N]
    # ranks[i] = 1 + number of negatives scoring >= pos_scores[i]
    pos_scores = pos_scores.detach()
    neg_scores = neg_scores.detach()

    # Expand for comparison: [P, 1] vs [1, N] → [P, N]
    # rank = 1 + count(neg >= pos)
    ranks = (neg_scores.unsqueeze(0) >= pos_scores.unsqueeze(1)).sum(dim=1) + 1
    ranks = ranks.float()

    mrr = (1.0 / ranks).mean().item()
    metrics = {"mrr": mrr}
    for k in k_values:
        metrics[f"hits@{k}"] = (ranks <= k).float().mean().item()

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dict as a single-line string."""
    parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
    return " | ".join(parts)
