"""Custom collation for dual-graph (LPG + RDF) batching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch_geometric.data import Data


@dataclass
class DualGraphBatch:
    """Batched dual-graph structure for LPG (GAT) + RDF (TransE) models.

    LPG batch: node features + edge_index with cumulative node offsets.
    RDF batch: edge_index with cumulative node offsets + edge_type for relation lookup.
    Text: plain Python lists (stay on CPU).
    """

    # LPG
    lpg_x: torch.Tensor = None  # [sum(N_i), 384]
    lpg_edge_index: torch.Tensor = None  # [2, sum(E_i)]
    lpg_edge_type: torch.Tensor = None  # [sum(E_i)]
    lpg_batch: torch.Tensor = None  # [sum(N_i)] graph membership
    lpg_ptr: torch.Tensor = None  # [B+1] cumulative node counts
    lpg_global_node_idx: torch.Tensor = None  # [sum(N_i)]

    # RDF
    rdf_edge_index: torch.Tensor = None  # [2, sum(T_i)]
    rdf_edge_type: torch.Tensor = None  # [sum(T_i)]
    rdf_batch: torch.Tensor = None  # [sum(N_rdf_i)]
    rdf_ptr: torch.Tensor = None  # [B+1]
    rdf_global_node_idx: torch.Tensor = None  # [sum(N_rdf_i)]

    # Text (CPU only)
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    question_ids: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    batch_size: int = 0

    def to(self, device, non_blocking: bool = False) -> DualGraphBatch:
        """Move all tensors to device. Strings stay on CPU."""
        kwargs = {"non_blocking": non_blocking}
        return DualGraphBatch(
            lpg_x=self.lpg_x.to(device, **kwargs),
            lpg_edge_index=self.lpg_edge_index.to(device, **kwargs),
            lpg_edge_type=self.lpg_edge_type.to(device, **kwargs),
            lpg_batch=self.lpg_batch.to(device, **kwargs),
            lpg_ptr=self.lpg_ptr.to(device, **kwargs),
            lpg_global_node_idx=self.lpg_global_node_idx.to(device, **kwargs),
            rdf_edge_index=self.rdf_edge_index.to(device, **kwargs),
            rdf_edge_type=self.rdf_edge_type.to(device, **kwargs),
            rdf_batch=self.rdf_batch.to(device, **kwargs),
            rdf_ptr=self.rdf_ptr.to(device, **kwargs),
            rdf_global_node_idx=self.rdf_global_node_idx.to(device, **kwargs),
            questions=self.questions,
            answers=self.answers,
            question_ids=self.question_ids,
            categories=self.categories,
            batch_size=self.batch_size,
        )

    def pin_memory(self) -> DualGraphBatch:
        """Pin tensors for async GPU transfer."""
        return DualGraphBatch(
            lpg_x=self.lpg_x.pin_memory(),
            lpg_edge_index=self.lpg_edge_index.pin_memory(),
            lpg_edge_type=self.lpg_edge_type.pin_memory(),
            lpg_batch=self.lpg_batch.pin_memory(),
            lpg_ptr=self.lpg_ptr.pin_memory(),
            lpg_global_node_idx=self.lpg_global_node_idx.pin_memory(),
            rdf_edge_index=self.rdf_edge_index.pin_memory(),
            rdf_edge_type=self.rdf_edge_type.pin_memory(),
            rdf_batch=self.rdf_batch.pin_memory(),
            rdf_ptr=self.rdf_ptr.pin_memory(),
            rdf_global_node_idx=self.rdf_global_node_idx.pin_memory(),
            questions=self.questions,
            answers=self.answers,
            question_ids=self.question_ids,
            categories=self.categories,
            batch_size=self.batch_size,
        )


def dual_graph_collate_fn(data_list: List[Data]) -> DualGraphBatch:
    """Collate a list of dual-graph Data objects into a DualGraphBatch.

    Handles cumulative node offsets for both LPG and RDF edge_index tensors.
    PyG's default Batch doesn't understand the dual-graph namespace, so we batch manually.
    """
    B = len(data_list)

    # --- LPG ---
    lpg_xs, lpg_eis, lpg_ets, lpg_gnis = [], [], [], []
    lpg_batch_list = []
    lpg_ptr = [0]
    lpg_node_offset = 0

    for i, d in enumerate(data_list):
        n = d.lpg_num_nodes.item()
        lpg_xs.append(d.lpg_x)
        lpg_gnis.append(d.lpg_global_node_idx)
        lpg_batch_list.append(torch.full((n,), i, dtype=torch.long))

        # Offset edge_index
        ei = d.lpg_edge_index
        if ei.numel() > 0:
            lpg_eis.append(ei + lpg_node_offset)
        else:
            lpg_eis.append(ei)
        lpg_ets.append(d.lpg_edge_type)

        lpg_node_offset += n
        lpg_ptr.append(lpg_node_offset)

    # --- RDF ---
    rdf_eis, rdf_ets, rdf_gnis = [], [], []
    rdf_batch_list = []
    rdf_ptr = [0]
    rdf_node_offset = 0

    for i, d in enumerate(data_list):
        n = d.rdf_num_nodes.item()
        rdf_gnis.append(d.rdf_global_node_idx)
        rdf_batch_list.append(torch.full((n,), i, dtype=torch.long))

        ei = d.rdf_edge_index
        if ei.numel() > 0:
            rdf_eis.append(ei + rdf_node_offset)
        else:
            rdf_eis.append(ei)
        rdf_ets.append(d.rdf_edge_type)

        rdf_node_offset += n
        rdf_ptr.append(rdf_node_offset)

    # --- Text ---
    questions = [d.question for d in data_list]
    answers = [d.answer for d in data_list]
    question_ids = [d.question_id for d in data_list]
    categories = [d.category for d in data_list]

    return DualGraphBatch(
        # LPG
        lpg_x=torch.cat(lpg_xs, dim=0),
        lpg_edge_index=torch.cat(lpg_eis, dim=1),
        lpg_edge_type=torch.cat(lpg_ets, dim=0),
        lpg_batch=torch.cat(lpg_batch_list, dim=0),
        lpg_ptr=torch.tensor(lpg_ptr, dtype=torch.long),
        lpg_global_node_idx=torch.cat(lpg_gnis, dim=0),
        # RDF
        rdf_edge_index=torch.cat(rdf_eis, dim=1),
        rdf_edge_type=torch.cat(rdf_ets, dim=0),
        rdf_batch=torch.cat(rdf_batch_list, dim=0),
        rdf_ptr=torch.tensor(rdf_ptr, dtype=torch.long),
        rdf_global_node_idx=torch.cat(rdf_gnis, dim=0),
        # Text
        questions=questions,
        answers=answers,
        question_ids=question_ids,
        categories=categories,
        batch_size=B,
    )
