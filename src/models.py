"""
GNN models for Graph RAG experiments.

- MessagePassingGNN (GAT-based): for LPG subgraphs
- TransEEncoder (KGE): for RDF triples
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.kge import TransE


class MessagePassingGNN(nn.Module):
    """LPG용 GNN: GAT (Graph Attention Network).

    Produces a single graph-level embedding by mean-pooling
    over GAT-refined node representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 384,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else hidden_dim * heads
            self.convs.append(GATConv(in_ch, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))
        self.output_proj = nn.Linear(hidden_dim * heads, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type=None):
        x = torch.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return self.output_proj(x.mean(dim=0))


class TransEEncoder(nn.Module):
    """RDF용 KGE: TransE (Translation-based Embedding).

    Learns entity and relation embeddings such that h + r ~ t,
    then produces a single embedding via mean-pooling.
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 256,
        output_dim: int = 384,
    ):
        super().__init__()
        self.transe = TransE(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_dim,
            margin=1.0,
            p_norm=1.0,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, head_index, rel_type, tail_index):
        if len(head_index) == 0:
            return torch.zeros(self.output_dim, device=head_index.device)
        head_emb = self.transe.node_emb(head_index)
        rel_emb = self.transe.rel_emb(rel_type)
        return self.output_proj((head_emb + rel_emb).mean(dim=0))
