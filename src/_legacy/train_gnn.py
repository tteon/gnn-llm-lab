"""
GNN link prediction training script.

Trains GAT / GCN / GraphTransformer encoders on the LPG graph
using BCEWithLogitsLoss for link prediction.

Usage:
    uv run python src/train_gnn.py --model gat --epochs 200 --lr 0.001
    uv run python src/train_gnn.py --model gcn
    uv run python src/train_gnn.py --model graph_transformer
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import LPGGraphBuilder, split_edges_gnn
from src.evaluation import compute_link_prediction_metrics, format_metrics
from src.models import (
    GATEncoder,
    GCNEncoder,
    GraphTransformerEncoder,
    LinkPredictor,
)
from src.utils import (
    Neo4jConfig,
    TrainingConfig,
    get_logger,
    set_seed,
    setup_logging,
)

logger = get_logger("train_gnn")

_ENCODER_CLASSES = {
    "gat": GATEncoder,
    "gcn": GCNEncoder,
    "graph_transformer": GraphTransformerEncoder,
}


def build_encoder(model_type: str, input_dim: int, config: TrainingConfig):
    """Instantiate the appropriate GNN encoder."""
    cls = _ENCODER_CLASSES[model_type]
    kwargs = dict(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_layers=config.gnn_num_layers,
        dropout=config.gnn_dropout,
    )
    if model_type in ("gat", "graph_transformer"):
        kwargs["heads"] = config.gnn_heads
    return cls(**kwargs)


def train_epoch(
    encoder, decoder, train_data, optimizer, device,
):
    """Run one training epoch. Returns loss value."""
    encoder.train()
    decoder.train()

    x = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)

    # Encode all nodes
    z = encoder(x, edge_index)

    # Positive edges
    pos_ei = train_data.pos_edge_label_index.to(device)
    pos_scores = decoder(z[pos_ei[0]], z[pos_ei[1]])

    # Negative edges
    neg_ei = train_data.neg_edge_label_index.to(device)
    neg_scores = decoder(z[neg_ei[0]], z[neg_ei[1]])

    # BCEWithLogitsLoss
    pos_labels = torch.ones(pos_scores.size(0), device=device)
    neg_labels = torch.zeros(neg_scores.size(0), device=device)
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])
    loss = F.binary_cross_entropy_with_logits(scores, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(encoder, decoder, eval_data, device):
    """Evaluate link prediction. Returns metrics dict."""
    encoder.eval()
    decoder.eval()

    x = eval_data.x.to(device)
    edge_index = eval_data.edge_index.to(device)

    z = encoder(x, edge_index)

    pos_ei = eval_data.pos_edge_label_index.to(device)
    pos_scores = decoder(z[pos_ei[0]], z[pos_ei[1]])

    neg_ei = eval_data.neg_edge_label_index.to(device)
    neg_scores = decoder(z[neg_ei[0]], z[neg_ei[1]])

    return compute_link_prediction_metrics(pos_scores, neg_scores)


def save_checkpoint(
    encoder, decoder, epoch, config, metrics, model_type, checkpoint_dir,
):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = checkpoint_dir / f"{ts}_{model_type}_best.pt"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "epoch": epoch,
            "config": {
                "model_type": model_type,
                "hidden_dim": config.hidden_dim,
                "output_dim": config.output_dim,
                "gnn_num_layers": config.gnn_num_layers,
                "gnn_heads": config.gnn_heads,
                "gnn_dropout": config.gnn_dropout,
                "decoder_type": config.decoder_type,
            },
            "metrics": metrics,
        },
        path,
    )
    logger.info(f"Checkpoint saved: {path}")
    return path


def save_embeddings(encoder, data, metadata, model_type, metrics, checkpoint_dir, device):
    """Export trained node embeddings."""
    encoder.eval()
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        embeddings = encoder(x, edge_index).cpu()

    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / f"{model_type}_embeddings.pt"
    torch.save(
        {
            "embeddings": embeddings,
            "node_ids": [metadata["idx_to_node"][i] for i in range(embeddings.size(0))],
            "node_to_idx": metadata["node_to_idx"],
            "model_type": model_type,
            "output_dim": embeddings.size(1),
            "training_metrics": metrics,
        },
        path,
    )
    logger.info(f"Embeddings saved: {path} ({embeddings.shape})")
    return path


def main():
    parser = argparse.ArgumentParser(description="Train GNN for link prediction on LPG")
    parser.add_argument("--model", type=str, default="gat", choices=list(_ENCODER_CLASSES))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--decoder", type=str, default="dot", choices=["dot", "mlp"])
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="results/checkpoints")
    args = parser.parse_args()

    setup_logging()

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        gnn_num_layers=args.num_layers,
        gnn_heads=args.heads,
        gnn_dropout=args.dropout,
        gnn_model_type=args.model,
        decoder_type=args.decoder,
        patience=args.patience,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    ).validate()

    device = torch.device(config.device)
    set_seed(config.seed)
    logger.info(f"Training {args.model} on {device} for {config.epochs} epochs")

    # 1. Load graph
    builder = LPGGraphBuilder()
    data, metadata = builder.build()
    logger.info(f"Graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges, "
                f"features={data.x.shape}")

    # 2. Split edges
    train_data, val_data, test_data = split_edges_gnn(
        data, val_ratio=config.val_ratio, test_ratio=config.test_ratio, seed=config.seed,
    )

    # 3. Create model
    encoder = build_encoder(args.model, data.x.size(1), config).to(device)
    decoder = LinkPredictor(config.output_dim, mode=config.decoder_type).to(device)

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # 4. Optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 5. Training loop
    best_mrr = 0.0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(encoder, decoder, train_data, optimizer, device)

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(encoder, decoder, val_data, device)
            logger.info(f"Epoch {epoch:3d} | loss={loss:.4f} | val {format_metrics(val_metrics)}")

            if val_metrics["mrr"] > best_mrr + config.min_delta:
                best_mrr = val_metrics["mrr"]
                best_metrics = val_metrics
                patience_counter = 0
                save_checkpoint(
                    encoder, decoder, epoch, config, val_metrics,
                    args.model, config.checkpoint_dir,
                )
            else:
                patience_counter += 5
                if patience_counter >= config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        elif epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d} | loss={loss:.4f}")

    # 6. Test evaluation
    test_metrics = evaluate(encoder, decoder, test_data, device)
    logger.info(f"Test results: {format_metrics(test_metrics)}")

    # 7. Save embeddings (using full graph edge_index for encoding)
    save_embeddings(encoder, data, metadata, args.model, test_metrics, config.checkpoint_dir, device)

    torch.cuda.empty_cache()
    logger.info("Done.")


if __name__ == "__main__":
    main()
