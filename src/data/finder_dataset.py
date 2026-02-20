"""FinDERGraphQADataset: PyG InMemoryDataset with dual LPG+RDF subgraphs per question."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from .graph_builders import build_lpg_subgraph, build_rdf_subgraph
from .vocabulary import Vocabulary, VocabularyBuilder

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "data/processed/finder_pyg"
DEFAULT_PARQUET = "data/raw/FinDER_KG_Merged.parquet"
DEFAULT_LPG_CACHE = "data/processed/lpg_full_graph.pt"


class FinDERGraphQADataset(InMemoryDataset):
    """Per-question dual-graph dataset: LPG (GAT-ready) + RDF (TransE-ready).

    Each sample is a PyG Data object with namespaced fields:
        Text:  question, answer, question_id, category
        LPG:   lpg_x, lpg_edge_index, lpg_edge_type, lpg_num_nodes, lpg_global_node_idx
        RDF:   rdf_edge_index, rdf_edge_type, rdf_num_nodes, rdf_global_node_idx
    """

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        split: str = "train",
        parquet_path: str = DEFAULT_PARQUET,
        lpg_cache_path: str = DEFAULT_LPG_CACHE,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        force_reload: bool = False,
    ):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got '{split}'")

        self.split = split
        self.parquet_path = str(parquet_path)
        self.lpg_cache_path = str(lpg_cache_path)
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # PyG InMemoryDataset calls process() if needed
        super().__init__(root, force_reload=force_reload)

        self.load(self.processed_paths[self.SPLITS.index(split)])

    @property
    def processed_file_names(self) -> List[str]:
        return ["train.pt", "val.pt", "test.pt"]

    def process(self) -> None:
        logger.info("Processing FinDERGraphQADataset...")

        # 1. Load and filter parquet
        df = pd.read_parquet(self.parquet_path)
        mask = df["lpg_nodes"].apply(_json_nonempty) & df["rdf_triples"].apply(
            _json_nonempty
        )
        df = df[mask].drop_duplicates(subset="_id").reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} samples (both LPG+RDF, deduplicated by _id)")

        # 2. Build vocabularies from all data (before split)
        vocabs = VocabularyBuilder.build(df)
        vocab_path = Path(self.processed_dir) / "vocab.pt"
        VocabularyBuilder.save(vocabs, vocab_path)

        # 3. Load global LPG graph
        cached = torch.load(self.lpg_cache_path, weights_only=False, map_location="cpu")
        global_node_features = cached["data"].x  # [13920, 384]
        global_node_to_idx = cached["metadata"]["node_to_idx"]
        logger.info(
            f"Loaded global LPG graph: {global_node_features.shape[0]} nodes, "
            f"{global_node_features.shape[1]}d features"
        )

        # 4. Stratified split
        splits = _stratified_split(
            df["category"].values,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )

        # 5. Build Data objects per split
        for split_name, indices in zip(self.SPLITS, splits):
            split_df = df.iloc[indices].reset_index(drop=True)
            data_list = []
            for _, row in split_df.iterrows():
                data = _build_data_object(
                    row,
                    global_node_to_idx,
                    global_node_features,
                    vocabs["lpg_edge_types"],
                    vocabs["rdf_entities"],
                    vocabs["rdf_relations"],
                )
                data_list.append(data)

            path = Path(self.processed_dir) / f"{split_name}.pt"
            self.save(data_list, str(path))
            logger.info(f"  {split_name}: {len(data_list)} samples → {path}")

        # 6. Save metadata
        metadata = {
            "created": datetime.now().isoformat(),
            "parquet_path": self.parquet_path,
            "total_samples": len(df),
            "splits": {
                name: len(idx) for name, idx in zip(self.SPLITS, splits)
            },
            "vocab_sizes": {k: len(v) for k, v in vocabs.items()},
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "lpg_feature_dim": int(global_node_features.shape[1]),
        }
        metadata_path = Path(self.processed_dir) / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

    @classmethod
    def get_vocab(cls, root: str = DEFAULT_ROOT) -> Dict[str, Vocabulary]:
        """Load vocabularies without instantiating the full dataset."""
        vocab_path = Path(root) / "processed" / "vocab.pt"
        return VocabularyBuilder.load(vocab_path)


def _json_nonempty(val) -> bool:
    """Check if a JSON string field contains a non-empty list."""
    if pd.isna(val) or not val:
        return False
    parsed = json.loads(val)
    return isinstance(parsed, list) and len(parsed) > 0


def _stratified_split(
    categories: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple:
    """Category-stratified train/val/test split using numpy only."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    for cat in np.unique(categories):
        cat_indices = np.where(categories == cat)[0]
        rng.shuffle(cat_indices)

        n = len(cat_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx.extend(cat_indices[:n_train])
        val_idx.extend(cat_indices[n_train : n_train + n_val])
        test_idx.extend(cat_indices[n_train + n_val :])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _build_data_object(
    row: pd.Series,
    global_node_to_idx: Dict[str, int],
    global_node_features: torch.Tensor,
    lpg_edge_type_vocab: Vocabulary,
    rdf_entity_vocab: Vocabulary,
    rdf_relation_vocab: Vocabulary,
) -> Data:
    """Build a single PyG Data object from a parquet row."""
    # LPG subgraph
    lpg = build_lpg_subgraph(
        row["lpg_nodes"],
        row["lpg_edges"],
        global_node_to_idx,
        global_node_features,
        lpg_edge_type_vocab,
    )

    # RDF subgraph
    rdf = build_rdf_subgraph(
        row["rdf_triples"],
        rdf_entity_vocab,
        rdf_relation_vocab,
    )

    data = Data(
        # Text fields
        question=row["text"],
        answer=row["answer"],
        question_id=row["_id"],
        category=row["category"],
        # LPG fields
        **lpg,
        # RDF fields
        **rdf,
    )
    return data
