"""
GNN models for Graph RAG experiments.

Existing (inference-only, graph-level pooling):
- MessagePassingGNN (GAT-based): for LPG subgraphs
- TransEEncoder (KGE): for RDF triples

Link Prediction training models (node-level embeddings):
- GATEncoder, GCNEncoder, GraphTransformerEncoder: GNN encoders for LPG
- LinkPredictor: edge score decoder (dot / MLP)
- KGEWrapper: unified wrapper for PyG KGE models (TransE/DistMult/ComplEx/RotatE)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GPSConv
from torch_geometric.nn.kge import TransE, DistMult, ComplEx, RotatE


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


# ---------------------------------------------------------------------------
# Link Prediction Training Models
# ---------------------------------------------------------------------------


class GATEncoder(nn.Module):
    """GAT encoder for link prediction. Returns node-level embeddings."""

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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode nodes. Returns [num_nodes, output_dim]."""
        x = torch.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return self.output_proj(x)


class GCNEncoder(nn.Module):
    """GCN encoder for link prediction. Returns node-level embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 384,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode nodes. Returns [num_nodes, output_dim]."""
        x = torch.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return self.output_proj(x)


class GraphTransformerEncoder(nn.Module):
    """GPS (General, Powerful, Scalable) encoder for link prediction.

    Combines local message passing (GATConv) with global Transformer attention.
    For a single full graph, pass batch=zeros(num_nodes).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 384,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        attn_type: str = "multihead",
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            local_conv = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
            gps = GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                attn_type=attn_type,
            )
            self.convs.append(gps)
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None
    ) -> torch.Tensor:
        """Encode nodes. Returns [num_nodes, output_dim]."""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = torch.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, batch=batch)
            x = norm(x)
            x = self.dropout(x)
        return self.output_proj(x)


class LinkPredictor(nn.Module):
    """Edge score decoder for link prediction."""

    def __init__(self, embed_dim: int, mode: str = "dot"):
        super().__init__()
        self.mode = mode
        if mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, 1),
            )

    def forward(
        self, z_src: torch.Tensor, z_dst: torch.Tensor
    ) -> torch.Tensor:
        """Score edges. Returns [num_edges] logits."""
        if self.mode == "dot":
            return (z_src * z_dst).sum(dim=-1)
        return self.mlp(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)


_KGE_CLASSES = {
    "transe": TransE,
    "distmult": DistMult,
    "complex": ComplEx,
    "rotate": RotatE,
}


class KGEWrapper(nn.Module):
    """Unified wrapper for PyG KGE models with embedding export."""

    def __init__(
        self,
        model_type: str,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 256,
        output_dim: int = 384,
        margin: float = 1.0,
        p_norm: float = 1.0,
    ):
        super().__init__()
        self.model_type = model_type
        self.output_dim = output_dim

        kge_cls = _KGE_CLASSES.get(model_type)
        if kge_cls is None:
            raise ValueError(f"Unknown KGE model: {model_type}. Choose from {list(_KGE_CLASSES)}")

        kwargs = dict(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_dim,
        )
        if model_type == "transe":
            kwargs.update(margin=margin, p_norm=p_norm)
        elif model_type == "rotate":
            kwargs.update(margin=margin)

        self.kge = kge_cls(**kwargs)

        # Projection from KGE dim to output dim
        kge_dim = hidden_dim * 2 if model_type in ("complex", "rotate") else hidden_dim
        self.output_proj = nn.Linear(kge_dim, output_dim)

    def loss(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KGE loss with built-in negative sampling."""
        return self.kge.loss(head_index, rel_type, tail_index)

    def test(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        batch_size: int = 256,
        k: int = 10,
    ):
        """Filtered ranking evaluation. Returns (mean_rank, mrr, hits@k)."""
        return self.kge.test(head_index, rel_type, tail_index, batch_size=batch_size, k=k)

    def get_entity_embeddings(self) -> torch.Tensor:
        """Export entity embeddings projected to output_dim. Returns [num_nodes, output_dim]."""
        with torch.no_grad():
            emb = self.kge.node_emb.weight
            if hasattr(self.kge, 'node_emb_im'):
                # ComplEx has real + imaginary parts
                emb = torch.cat([emb, self.kge.node_emb_im.weight], dim=-1)
            return self.output_proj(emb)
