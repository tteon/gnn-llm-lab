"""
Evaluation metrics for GraphRAG experiments.

Computes exact match, substring match, token F1, ROUGE, and BERTScore
for comparing generated answers against ground truth.
"""

import re
import string
from dataclasses import dataclass
from typing import Dict, List, Optional

from .logging_config import get_logger

logger = get_logger("evaluation")


@dataclass
class EvaluationResult:
    """Evaluation scores for a single prediction."""
    exact_match: float
    substring_match: float
    token_f1: float
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bert_score_f1: float = 0.0


class Evaluator:
    """Computes evaluation metrics for generated responses."""

    def __init__(self, use_bertscore: bool = True, use_rouge: bool = True):
        self.use_bertscore = use_bertscore
        self.use_rouge = use_rouge
        self._rouge_scorer = None
        self._bert_scorer = None

    def evaluate_single(self, prediction: str, reference: str) -> EvaluationResult:
        """Evaluate a single prediction against reference."""
        em = self.compute_exact_match(prediction, reference)
        sm = self.compute_substring_match(prediction, reference)
        f1 = self.compute_token_f1(prediction, reference)

        rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        if self.use_rouge:
            rouge_scores = self.compute_rouge(prediction, reference)

        return EvaluationResult(
            exact_match=em,
            substring_match=sm,
            token_f1=f1,
            rouge_1=rouge_scores.get("rouge1", 0.0),
            rouge_2=rouge_scores.get("rouge2", 0.0),
            rouge_l=rouge_scores.get("rougeL", 0.0),
        )

    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate a batch and return aggregated metrics.

        Returns:
            Dict with mean scores for each metric
        """
        results = [
            self.evaluate_single(p, r) for p, r in zip(predictions, references)
        ]

        n = len(results)
        if n == 0:
            return {}

        metrics = {
            "exact_match": sum(r.exact_match for r in results) / n,
            "substring_match": sum(r.substring_match for r in results) / n,
            "token_f1": sum(r.token_f1 for r in results) / n,
        }

        if self.use_rouge:
            metrics["rouge_1"] = sum(r.rouge_1 for r in results) / n
            metrics["rouge_2"] = sum(r.rouge_2 for r in results) / n
            metrics["rouge_l"] = sum(r.rouge_l for r in results) / n

        # BERTScore: batch computation is more efficient
        if self.use_bertscore and predictions:
            try:
                bert_scores = self._compute_bertscore_batch(predictions, references)
                metrics["bert_score_f1"] = bert_scores
                # Update individual results
                for r in results:
                    r.bert_score_f1 = bert_scores
            except Exception as e:
                logger.warning(f"BERTScore computation failed: {e}")
                metrics["bert_score_f1"] = 0.0

        return metrics

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization after normalization."""
        return Evaluator._normalize(text).split()

    @staticmethod
    def compute_exact_match(prediction: str, reference: str) -> float:
        """Normalized exact match."""
        return float(Evaluator._normalize(prediction) == Evaluator._normalize(reference))

    @staticmethod
    def compute_substring_match(prediction: str, reference: str) -> float:
        """Check if normalized reference is a substring of normalized prediction."""
        ref_norm = Evaluator._normalize(reference)
        pred_norm = Evaluator._normalize(prediction)
        if not ref_norm:
            return 1.0
        return float(ref_norm in pred_norm)

    @staticmethod
    def compute_token_f1(prediction: str, reference: str) -> float:
        """Token-level F1 score."""
        pred_tokens = Evaluator._tokenize(prediction)
        ref_tokens = Evaluator._tokenize(reference)

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
            except ImportError:
                logger.warning("rouge-score not installed, skipping ROUGE metrics")
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        scores = self._rouge_scorer.score(reference, prediction)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    def _compute_bertscore_batch(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Compute mean BERTScore F1 for a batch."""
        if self._bert_scorer is None:
            try:
                from bert_score import BERTScorer
                self._bert_scorer = BERTScorer(
                    model_type="microsoft/deberta-xlarge-mnli",
                    lang="en",
                    rescale_with_baseline=True,
                )
            except ImportError:
                logger.warning("bert-score not installed, skipping BERTScore")
                return 0.0

        P, R, F1 = self._bert_scorer.score(predictions, references)
        return float(F1.mean())
