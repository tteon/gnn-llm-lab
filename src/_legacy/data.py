"""
Data pipeline for GNN/KGE link prediction training.

- LPGGraphBuilder: Neo4j finderlpg → PyG Data (nodes + edges + features)
- RDFTripleBuilder: Neo4j finderrdf → triple tensors
- split_edges_gnn: RandomLinkSplit for GNN training
- split_triples_kge: random index split for KGE training
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from src.utils import (
    Neo4jClient,
    Neo4jConfig,
    TrainingConfig,
    get_logger,
)

logger = get_logger("data")

CACHE_DIR = Path("data/processed")


class LPGGraphBuilder:
    """Build PyG Data from Neo4j LPG (finderlpg) database."""

    def __init__(self, neo4j_config: Neo4jConfig = None):
        self.config = neo4j_config or Neo4jConfig(database="finderlpg")

    def build(self, cache_path: str = None) -> Tuple[Data, Dict]:
        """Build full LPG graph as PyG Data.

        Returns:
            (data, metadata) where data has x, edge_index, edge_type
            and metadata has node_to_idx, idx_to_node, rel_to_idx, idx_to_rel
        """
        cache_path = Path(cache_path) if cache_path else CACHE_DIR / "lpg_full_graph.pt"
        if cache_path.exists():
            logger.info(f"Loading cached LPG graph from {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            return cached["data"], cached["metadata"]

        logger.info("Building LPG graph from Neo4j (finderlpg)...")
        cfg = Neo4jConfig(
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password,
            database="finderlpg",
        )
        with Neo4jClient(cfg) as client:
            client.connect()

            # Get all Entity nodes
            nodes = client.query(
                "MATCH (e:Entity) "
                "RETURN e.id AS id, coalesce(e.name, e.id) AS name, e.label AS label "
                "ORDER BY e.id"
            )
            logger.info(f"Fetched {len(nodes)} nodes")

            # Get all edges
            edges = client.query(
                "MATCH (s:Entity)-[r]->(t:Entity) "
                "RETURN s.id AS source, t.id AS target, type(r) AS rel_type"
            )
            logger.info(f"Fetched {len(edges)} edges")

        # Build node mapping
        node_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
        idx_to_node = {i: n["id"] for i, n in enumerate(nodes)}

        # Build node text for embedding
        node_texts = []
        for n in nodes:
            text = n["name"] or n["id"]
            if n.get("label"):
                text = f"{n['label']}: {text}"
            node_texts.append(text)

        # Encode node features with sentence-transformers
        logger.info("Encoding node features with sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        x = encoder.encode(node_texts, show_progress_bar=True, convert_to_tensor=True)
        x = x.float().clone()  # clone to detach from inference mode
        logger.info(f"Node features: {x.shape}")

        # Build edge_index and edge_type
        rel_types = sorted(set(e["rel_type"] for e in edges))
        rel_to_idx = {r: i for i, r in enumerate(rel_types)}
        idx_to_rel = {i: r for i, r in enumerate(rel_types)}

        src_list, dst_list, etype_list = [], [], []
        skipped = 0
        for e in edges:
            if e["source"] not in node_to_idx or e["target"] not in node_to_idx:
                skipped += 1
                continue
            src_list.append(node_to_idx[e["source"]])
            dst_list.append(node_to_idx[e["target"]])
            etype_list.append(rel_to_idx[e["rel_type"]])

        if skipped:
            logger.warning(f"Skipped {skipped} edges with missing nodes")

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(etype_list, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        data.num_nodes = len(nodes)

        metadata = {
            "node_to_idx": node_to_idx,
            "idx_to_node": idx_to_node,
            "rel_to_idx": rel_to_idx,
            "idx_to_rel": idx_to_rel,
            "node_texts": node_texts,
        }

        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"data": data, "metadata": metadata}, cache_path)
        logger.info(f"Cached LPG graph to {cache_path}")

        return data, metadata


class RDFTripleBuilder:
    """Build triple tensors from Neo4j RDF (finderrdf) database."""

    def __init__(self, neo4j_config: Neo4jConfig = None):
        self.config = neo4j_config or Neo4jConfig(database="finderrdf")

    def build(self, cache_path: str = None) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Build RDF triple tensors.

        Returns:
            (triples, metadata) where triples has head_index, rel_type, tail_index
            and metadata has node_to_idx, idx_to_node, rel_to_idx, idx_to_rel
        """
        cache_path = Path(cache_path) if cache_path else CACHE_DIR / "rdf_triples.pt"
        if cache_path.exists():
            logger.info(f"Loading cached RDF triples from {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            return cached["triples"], cached["metadata"]

        logger.info("Building RDF triples from Neo4j (finderrdf)...")
        cfg = Neo4jConfig(
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password,
            database="finderrdf",
        )
        with Neo4jClient(cfg) as client:
            client.connect()

            # Get all Resource nodes
            resources = client.query(
                "MATCH (r:Resource) RETURN r.uri AS uri ORDER BY r.uri"
            )
            logger.info(f"Fetched {len(resources)} resources")

            # Get all relationships (triples)
            # finderrdf uses TRIPLE rels with predicate property, or dynamic types via APOC
            raw_triples = client.query(
                "MATCH (s:Resource)-[r]->(o:Resource) "
                "RETURN s.uri AS subject, o.uri AS object, "
                "       coalesce(r.predicate, type(r)) AS predicate"
            )
            logger.info(f"Fetched {len(raw_triples)} triples")

        # Build node mapping
        node_to_idx = {r["uri"]: i for i, r in enumerate(resources)}
        idx_to_node = {i: r["uri"] for i, r in enumerate(resources)}

        # Build relation mapping
        predicates = sorted(set(t["predicate"] for t in raw_triples))
        rel_to_idx = {p: i for i, p in enumerate(predicates)}
        idx_to_rel = {i: p for i, p in enumerate(predicates)}

        head_list, rel_list, tail_list = [], [], []
        skipped = 0
        for t in raw_triples:
            if t["subject"] not in node_to_idx or t["object"] not in node_to_idx:
                skipped += 1
                continue
            head_list.append(node_to_idx[t["subject"]])
            rel_list.append(rel_to_idx[t["predicate"]])
            tail_list.append(node_to_idx[t["object"]])

        if skipped:
            logger.warning(f"Skipped {skipped} triples with missing nodes")

        triples = {
            "head_index": torch.tensor(head_list, dtype=torch.long),
            "rel_type": torch.tensor(rel_list, dtype=torch.long),
            "tail_index": torch.tensor(tail_list, dtype=torch.long),
            "num_nodes": len(resources),
            "num_relations": len(predicates),
        }

        metadata = {
            "node_to_idx": node_to_idx,
            "idx_to_node": idx_to_node,
            "rel_to_idx": rel_to_idx,
            "idx_to_rel": idx_to_rel,
        }

        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"triples": triples, "metadata": metadata}, cache_path)
        logger.info(f"Cached RDF triples to {cache_path}")

        return triples, metadata


def split_edges_gnn(
    data: Data,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Data, Data, Data]:
    """Split edges for GNN link prediction using RandomLinkSplit.

    Returns:
        (train_data, val_data, test_data) with pos/neg edge labels
    """
    logger.info(f"Splitting edges: val={val_ratio}, test={test_ratio}")
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=False,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        split_labels=True,
    )
    # Set seed for reproducibility
    torch.manual_seed(seed)
    train_data, val_data, test_data = transform(data)
    logger.info(
        f"Split: train={train_data.pos_edge_label_index.size(1)} pos edges, "
        f"val={val_data.pos_edge_label_index.size(1)}, "
        f"test={test_data.pos_edge_label_index.size(1)}"
    )
    return train_data, val_data, test_data


def split_triples_kge(
    triples: Dict[str, torch.Tensor],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    """Split triples for KGE training by random index permutation.

    Returns:
        (train, val, test) triple dicts each with head_index, rel_type, tail_index
    """
    n = triples["head_index"].size(0)
    torch.manual_seed(seed)
    perm = torch.randperm(n)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    splits = {}
    for name, start, end in [
        ("train", 0, n_train),
        ("val", n_train, n_train + n_val),
        ("test", n_train + n_val, n),
    ]:
        idx = perm[start:end]
        splits[name] = {
            "head_index": triples["head_index"][idx],
            "rel_type": triples["rel_type"][idx],
            "tail_index": triples["tail_index"][idx],
        }

    logger.info(
        f"Triple split: train={splits['train']['head_index'].size(0)}, "
        f"val={splits['val']['head_index'].size(0)}, "
        f"test={splits['test']['head_index'].size(0)}"
    )
    return splits["train"], splits["val"], splits["test"]
