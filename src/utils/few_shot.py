"""
Few-shot example selection for category-based representative sampling.

Selects per-category representative examples using embedding centroids
and caches them for reproducibility.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FewShotConfig
from .logging_config import get_logger

logger = get_logger("few_shot")


class FewShotSelector:
    """Selects representative few-shot examples per category."""

    def __init__(self, config: FewShotConfig):
        self.config = config
        self.examples: Dict[str, List[dict]] = {}
        self._loaded = False

    def build_examples(self, df: pd.DataFrame, embedder=None) -> None:
        """
        Build few-shot examples from dataset.

        Args:
            df: DataFrame with columns: _id, text, answer, category
            embedder: SentenceTransformer for representative selection
        """
        if "category" not in df.columns:
            logger.warning("No 'category' column found, using all data as one group")
            df = df.copy()
            df["category"] = "general"

        categories = df["category"].dropna().unique()
        n_per_cat = self.config.num_examples_per_category

        logger.info(
            f"Building few-shot examples: {len(categories)} categories, "
            f"{n_per_cat} per category, strategy={self.config.selection_strategy}"
        )

        for cat in categories:
            cat_df = df[df["category"] == cat].copy()

            if len(cat_df) == 0:
                continue

            if self.config.selection_strategy == "representative" and embedder is not None:
                selected = self._select_representative(cat_df, embedder, n_per_cat)
            else:
                selected = self._select_random(cat_df, n_per_cat)

            self.examples[cat] = selected
            logger.debug(f"  {cat}: {len(selected)} examples selected")

        self._loaded = True
        logger.info(f"Few-shot examples built: {sum(len(v) for v in self.examples.values())} total")

    def _select_representative(
        self, cat_df: pd.DataFrame, embedder, n: int
    ) -> List[dict]:
        """Select centroid-nearest examples."""
        questions = cat_df["text"].tolist()
        embeddings = embedder.encode(questions, show_progress_bar=False)
        centroid = np.mean(embeddings, axis=0)

        # Compute distances to centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        nearest_indices = np.argsort(distances)[:n]

        selected = []
        for idx in nearest_indices:
            row = cat_df.iloc[idx]
            selected.append({
                "question_id": str(row["_id"]),
                "question": row["text"],
                "answer": str(row["answer"]),
            })
        return selected

    def _select_random(self, cat_df: pd.DataFrame, n: int) -> List[dict]:
        """Select random examples."""
        sampled = cat_df.sample(n=min(n, len(cat_df)), random_state=42)
        selected = []
        for _, row in sampled.iterrows():
            selected.append({
                "question_id": str(row["_id"]),
                "question": row["text"],
                "answer": str(row["answer"]),
            })
        return selected

    def get_examples(
        self, category: str, exclude_question_id: Optional[str] = None
    ) -> List[dict]:
        """
        Get few-shot examples for a category.

        Args:
            category: Question category
            exclude_question_id: Question ID to exclude (to avoid data leakage)

        Returns:
            List of example dicts with question/answer
        """
        if not self._loaded:
            logger.warning("Few-shot examples not loaded, returning empty")
            return []

        examples = self.examples.get(category, [])

        if exclude_question_id:
            examples = [e for e in examples if e["question_id"] != exclude_question_id]

        return examples

    def format_for_prompt(self, examples: List[dict]) -> List[Dict[str, str]]:
        """
        Format examples as chat message pairs for the prompt.

        Returns:
            List of {"role": "user"/"assistant", "content": ...} dicts
        """
        messages = []
        for ex in examples:
            messages.append({
                "role": "user",
                "content": f"Question: {ex['question']}",
            })
            messages.append({
                "role": "assistant",
                "content": ex["answer"],
            })
        return messages

    def save(self, path: Optional[str] = None) -> None:
        """Save examples to JSON cache."""
        save_path = Path(path or self.config.cache_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "strategy": self.config.selection_strategy,
                "n_per_cat": self.config.num_examples_per_category,
            },
            "examples": self.examples,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Few-shot examples saved: {save_path}")

    def load(self, path: Optional[str] = None) -> bool:
        """
        Load examples from JSON cache.

        Returns:
            True if loaded successfully
        """
        load_path = Path(path or self.config.cache_path)
        if not load_path.exists():
            logger.info(f"No cached few-shot examples at {load_path}")
            return False

        with open(load_path) as f:
            data = json.load(f)

        self.examples = data.get("examples", {})
        self._loaded = bool(self.examples)

        meta = data.get("metadata", {})
        logger.info(
            f"Few-shot examples loaded: {sum(len(v) for v in self.examples.values())} total "
            f"(strategy={meta.get('strategy', '?')}, n_per_cat={meta.get('n_per_cat', '?')})"
        )
        return self._loaded
