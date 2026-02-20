"""Vocabulary: bidirectional str↔idx mapping for graph element types."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class Vocabulary:
    """Bidirectional string ↔ index mapping."""

    name: str
    _str_to_idx: Dict[str, int] = field(default_factory=dict)
    _idx_to_str: Dict[int, str] = field(default_factory=dict)

    def add(self, token: str) -> int:
        """Add token, return its index. Idempotent."""
        if token in self._str_to_idx:
            return self._str_to_idx[token]
        idx = len(self._str_to_idx)
        self._str_to_idx[token] = idx
        self._idx_to_str[idx] = token
        return idx

    def __getitem__(self, token: str) -> int:
        return self._str_to_idx[token]

    def get(self, token: str, default: Optional[int] = None) -> Optional[int]:
        return self._str_to_idx.get(token, default)

    def idx_to_str(self, idx: int) -> str:
        return self._idx_to_str[idx]

    def __len__(self) -> int:
        return len(self._str_to_idx)

    def __contains__(self, token: str) -> bool:
        return token in self._str_to_idx

    def to_dict(self) -> Dict:
        return {"name": self.name, "str_to_idx": self._str_to_idx}

    @classmethod
    def from_dict(cls, d: Dict) -> Vocabulary:
        vocab = cls(name=d["name"])
        vocab._str_to_idx = d["str_to_idx"]
        vocab._idx_to_str = {v: k for k, v in d["str_to_idx"].items()}
        return vocab


class VocabularyBuilder:
    """Scan a parquet DataFrame to build global vocabularies."""

    @staticmethod
    def build(df: pd.DataFrame) -> Dict[str, Vocabulary]:
        """Build 3 vocabularies from the filtered DataFrame.

        Args:
            df: DataFrame with lpg_edges (JSON str) and rdf_triples (JSON str) columns.

        Returns:
            Dict with keys: lpg_edge_types, rdf_entities, rdf_relations.
        """
        lpg_edge_types = Vocabulary("lpg_edge_types")
        rdf_entities = Vocabulary("rdf_entities")
        rdf_relations = Vocabulary("rdf_relations")

        for _, row in df.iterrows():
            # LPG edge types
            edges = json.loads(row["lpg_edges"])
            for e in edges:
                lpg_edge_types.add(e["type"])

            # RDF entities and relations
            triples = json.loads(row["rdf_triples"])
            for t in triples:
                rdf_entities.add(t["subject"])
                rdf_entities.add(t["object"])
                rdf_relations.add(t["predicate"])

        logger.info(
            f"Vocabularies built: lpg_edge_types={len(lpg_edge_types)}, "
            f"rdf_entities={len(rdf_entities)}, rdf_relations={len(rdf_relations)}"
        )
        return {
            "lpg_edge_types": lpg_edge_types,
            "rdf_entities": rdf_entities,
            "rdf_relations": rdf_relations,
        }

    @staticmethod
    def save(vocabs: Dict[str, Vocabulary], path) -> None:
        torch.save({k: v.to_dict() for k, v in vocabs.items()}, path)
        logger.info(f"Saved vocabularies to {path}")

    @staticmethod
    def load(path) -> Dict[str, Vocabulary]:
        raw = torch.load(path, weights_only=False)
        return {k: Vocabulary.from_dict(v) for k, v in raw.items()}
