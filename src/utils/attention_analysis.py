"""
Attention distribution analysis metrics for LPG vs RDF context comparison.

Computes entropy, entity coverage, prefix waste ratio, semantic density,
and per-head statistics from AttentionResult data.
"""

from typing import Dict, List, Optional

import numpy as np

from .logging_config import get_logger

logger = get_logger("attention_analysis")

# Common FIBO/RDF prefix fragments that carry no semantic meaning
DEFAULT_PREFIX_FRAGMENTS = [
    "fibo", "fnd", "fbc", "fct", "rel", "fi", "acc", "agr", "aap",
    "agt", "arr", "cur", "aeq", "std", "utl", "pty", "pas", "org",
    "dt", "le", "lp", "fm", "oc", "fd", "ip", "breg", "fse", "fpas",
    "ex:",
]

# Structural tokens that are not semantic
STRUCTURAL_TOKENS = {":", "-", "(", ")", ",", ".", "->", "--[", "]-->", "===", ""}


class AttentionAnalyzer:
    """Computes attention distribution metrics for context format comparison."""

    def __init__(
        self,
        prefix_fragments: Optional[List[str]] = None,
    ):
        self.prefix_fragments = prefix_fragments or DEFAULT_PREFIX_FRAGMENTS

    def entropy(self, scores: np.ndarray) -> float:
        """Compute Shannon entropy of attention distribution in bits.

        Lower entropy = more concentrated attention = model "knows where to look".
        Higher entropy = diffuse attention = model is uncertain.

        Args:
            scores: Attention weights over context tokens [num_ctx_tokens]

        Returns:
            Entropy in bits. Range: [0, log2(num_tokens)]
        """
        total = scores.sum()
        if total < 1e-10 or len(scores) == 0:
            return 0.0
        p = scores / total
        p = p[p > 1e-10]
        return float(-np.sum(p * np.log2(p)))

    def entity_coverage_at_k(
        self,
        token_scores: np.ndarray,
        token_strings: List[str],
        ground_truth_entities: List[str],
        k: int = 20,
    ) -> float:
        """Fraction of ground truth entities found in top-K attended tokens.

        Measures whether the model attends to the right entities.

        Args:
            token_scores: Attention weights [num_ctx_tokens]
            token_strings: Decoded token strings
            ground_truth_entities: Entity names from the answer/graph
            k: Number of top tokens to consider

        Returns:
            Coverage ratio 0.0 ~ 1.0
        """
        if len(token_scores) == 0 or not ground_truth_entities:
            return 0.0

        top_indices = np.argsort(token_scores)[-k:][::-1]
        top_tokens_lower = set()
        for i in top_indices:
            if i < len(token_strings):
                top_tokens_lower.add(token_strings[i].strip().lower())

        covered = 0
        for entity in ground_truth_entities:
            entity_words = entity.lower().split()
            # Entity is covered if any of its words appear in top tokens
            if any(
                ew in tok or tok in ew
                for ew in entity_words
                for tok in top_tokens_lower
                if len(ew) > 1 and len(tok) > 1
            ):
                covered += 1

        return covered / len(ground_truth_entities)

    def prefix_waste_ratio(
        self,
        token_scores: np.ndarray,
        token_strings: List[str],
    ) -> float:
        """Fraction of attention allocated to ontology prefix tokens.

        Higher = more attention wasted on non-semantic FIBO prefixes.

        Args:
            token_scores: Attention weights [num_ctx_tokens]
            token_strings: Decoded token strings

        Returns:
            Waste ratio 0.0 ~ 1.0
        """
        total = float(token_scores.sum())
        if total < 1e-10:
            return 0.0

        prefix_attn = 0.0
        for i, tok in enumerate(token_strings):
            if self._is_prefix_token(tok):
                prefix_attn += float(token_scores[i])

        return prefix_attn / total

    def semantic_density(
        self,
        token_strings: List[str],
    ) -> float:
        """Ratio of semantically meaningful tokens to total tokens.

        Higher = more efficient use of context window.

        Args:
            token_strings: Decoded token strings

        Returns:
            Density ratio 0.0 ~ 1.0
        """
        if not token_strings:
            return 0.0

        semantic_count = sum(
            1 for tok in token_strings
            if not self._is_prefix_token(tok) and not self._is_structural_token(tok)
        )
        return semantic_count / len(token_strings)

    def per_head_entropy_stats(
        self,
        per_head_scores: Dict[int, np.ndarray],
    ) -> Dict[str, float]:
        """Statistics of entropy across attention heads.

        High std = some heads are specialized (sharp) while others are diffuse.
        Low std = all heads behave similarly.

        Args:
            per_head_scores: {head_idx: attention_weights}

        Returns:
            {"mean", "std", "min", "max", "num_sharp", "num_diffuse"}
        """
        if not per_head_scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                    "num_sharp": 0, "num_diffuse": 0}

        entropies = [self.entropy(scores) for scores in per_head_scores.values()]
        arr = np.array(entropies)
        mean_h = float(arr.mean())
        std_h = float(arr.std())

        return {
            "mean": mean_h,
            "std": std_h,
            "min": float(arr.min()),
            "max": float(arr.max()),
            # Heads with entropy < mean - 1*std are "sharp" (specialized)
            "num_sharp": int(np.sum(arr < mean_h - std_h)),
            # Heads with entropy > mean + 1*std are "diffuse"
            "num_diffuse": int(np.sum(arr > mean_h + std_h)),
        }

    def compute_all_metrics(
        self,
        token_scores: np.ndarray,
        token_strings: List[str],
        ground_truth_entities: Optional[List[str]] = None,
        per_head_scores: Optional[Dict[int, np.ndarray]] = None,
        k: int = 20,
    ) -> Dict[str, float]:
        """Compute all attention metrics in one call.

        Args:
            token_scores: Aggregated attention weights [num_ctx_tokens]
            token_strings: Decoded token strings
            ground_truth_entities: Entity names for coverage
            per_head_scores: Per-head attention for head analysis
            k: Top-K for entity coverage

        Returns:
            Dict with all metric values
        """
        metrics: Dict[str, float] = {
            "entropy": self.entropy(token_scores),
            "prefix_waste_ratio": self.prefix_waste_ratio(token_scores, token_strings),
            "semantic_density": self.semantic_density(token_strings),
            "num_context_tokens": float(len(token_strings)),
        }

        if ground_truth_entities:
            metrics["entity_coverage_at_k"] = self.entity_coverage_at_k(
                token_scores, token_strings, ground_truth_entities, k=k,
            )

        if per_head_scores:
            head_stats = self.per_head_entropy_stats(per_head_scores)
            for key, val in head_stats.items():
                metrics[f"head_entropy_{key}"] = float(val)

        return metrics

    def _is_prefix_token(self, tok: str) -> bool:
        """Check if a token is an ontology prefix fragment."""
        tok_lower = tok.strip().lower()
        if not tok_lower:
            return False
        return any(frag in tok_lower for frag in self.prefix_fragments)

    def _is_structural_token(self, tok: str) -> bool:
        """Check if a token is structural (punctuation, formatting)."""
        return tok.strip() in STRUCTURAL_TOKENS
