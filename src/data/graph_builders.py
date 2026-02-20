"""Per-question subgraph builders: LPG → GAT-ready, RDF → TransE-ready tensors."""

from __future__ import annotations

import json
import logging
from typing import Dict, List

import torch

from .vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def build_lpg_subgraph(
    lpg_nodes_json: str,
    lpg_edges_json: str,
    global_node_to_idx: Dict[str, int],
    global_node_features: torch.Tensor,
    edge_type_vocab: Vocabulary,
) -> Dict[str, torch.Tensor]:
    """Build LPG subgraph tensors for a single question.

    Args:
        lpg_nodes_json: JSON string of node list.
        lpg_edges_json: JSON string of edge list.
        global_node_to_idx: Global node ID → index mapping (from lpg_full_graph.pt).
        global_node_features: [num_global_nodes, 384] features tensor.
        edge_type_vocab: LPG edge type vocabulary.

    Returns:
        Dict with keys: lpg_x, lpg_edge_index, lpg_edge_type,
                        lpg_num_nodes, lpg_global_node_idx
    """
    nodes: List[Dict] = json.loads(lpg_nodes_json)
    edges: List[Dict] = json.loads(lpg_edges_json)

    if not nodes:
        return _empty_lpg()

    # Local node mapping: node_id → local 0..N-1
    local_ids = []
    global_indices = []
    for n in nodes:
        nid = n["id"]
        if nid in global_node_to_idx:
            local_ids.append(nid)
            global_indices.append(global_node_to_idx[nid])

    if not local_ids:
        return _empty_lpg()

    local_node_to_idx = {nid: i for i, nid in enumerate(local_ids)}
    num_nodes = len(local_ids)

    # Slice features from global graph
    global_idx_tensor = torch.tensor(global_indices, dtype=torch.long)
    x = global_node_features[global_idx_tensor]  # [N, 384]

    # Build COO edge_index + edge_type
    src_list, dst_list, etype_list = [], [], []
    for e in edges:
        s, t = e["source"], e["target"]
        if s in local_node_to_idx and t in local_node_to_idx:
            et = edge_type_vocab.get(e["type"])
            if et is not None:
                src_list.append(local_node_to_idx[s])
                dst_list.append(local_node_to_idx[t])
                etype_list.append(et)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(etype_list, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

    return {
        "lpg_x": x,
        "lpg_edge_index": edge_index,
        "lpg_edge_type": edge_type,
        "lpg_num_nodes": torch.tensor(num_nodes, dtype=torch.long),
        "lpg_global_node_idx": global_idx_tensor,
    }


def _empty_lpg() -> Dict[str, torch.Tensor]:
    return {
        "lpg_x": torch.zeros(0, 384),
        "lpg_edge_index": torch.zeros(2, 0, dtype=torch.long),
        "lpg_edge_type": torch.zeros(0, dtype=torch.long),
        "lpg_num_nodes": torch.tensor(0, dtype=torch.long),
        "lpg_global_node_idx": torch.zeros(0, dtype=torch.long),
    }


def build_rdf_subgraph(
    rdf_triples_json: str,
    entity_vocab: Vocabulary,
    relation_vocab: Vocabulary,
) -> Dict[str, torch.Tensor]:
    """Build RDF subgraph tensors for a single question.

    Args:
        rdf_triples_json: JSON string of triple list.
        entity_vocab: Global RDF entity vocabulary.
        relation_vocab: Global RDF relation vocabulary.

    Returns:
        Dict with keys: rdf_edge_index, rdf_edge_type,
                        rdf_num_nodes, rdf_global_node_idx
    """
    triples: List[Dict] = json.loads(rdf_triples_json)

    if not triples:
        return _empty_rdf()

    # Collect local entities and build local→global mapping
    local_entities: Dict[str, int] = {}  # entity_str → local idx
    global_indices: List[int] = []

    def _get_local_idx(entity_str: str) -> int:
        if entity_str in local_entities:
            return local_entities[entity_str]
        global_idx = entity_vocab.get(entity_str)
        if global_idx is None:
            return -1
        local_idx = len(local_entities)
        local_entities[entity_str] = local_idx
        global_indices.append(global_idx)
        return local_idx

    src_list, dst_list, rel_list = [], [], []
    for t in triples:
        s_idx = _get_local_idx(t["subject"])
        o_idx = _get_local_idx(t["object"])
        r_idx = relation_vocab.get(t["predicate"])
        if s_idx >= 0 and o_idx >= 0 and r_idx is not None:
            src_list.append(s_idx)
            dst_list.append(o_idx)
            rel_list.append(r_idx)

    if not src_list:
        return _empty_rdf()

    num_nodes = len(local_entities)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(rel_list, dtype=torch.long)
    global_node_idx = torch.tensor(global_indices, dtype=torch.long)

    return {
        "rdf_edge_index": edge_index,
        "rdf_edge_type": edge_type,
        "rdf_num_nodes": torch.tensor(num_nodes, dtype=torch.long),
        "rdf_global_node_idx": global_node_idx,
    }


def _empty_rdf() -> Dict[str, torch.Tensor]:
    return {
        "rdf_edge_index": torch.zeros(2, 0, dtype=torch.long),
        "rdf_edge_type": torch.zeros(0, dtype=torch.long),
        "rdf_num_nodes": torch.tensor(0, dtype=torch.long),
        "rdf_global_node_idx": torch.zeros(0, dtype=torch.long),
    }
