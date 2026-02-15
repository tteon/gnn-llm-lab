"""
KGE link prediction training script.

Trains TransE / DistMult / ComplEx / RotatE on the RDF graph
using PyG's built-in KGE loss with negative sampling.

Usage:
    uv run python src/train_kge.py --model transe --epochs 200 --lr 0.01
    uv run python src/train_kge.py --model distmult
    uv run python src/train_kge.py --model complex
    uv run python src/train_kge.py --model rotate
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.data import RDFTripleBuilder, split_triples_kge
from src.evaluation import format_metrics
from src.models import KGEWrapper
from src.utils import (
    Neo4jConfig,
    TrainingConfig,
    get_logger,
    set_seed,
    setup_logging,
)

logger = get_logger("train_kge")

VALID_MODELS = ("transe", "distmult", "complex", "rotate")


def train_epoch(model, train_triples, optimizer, device, batch_size):
    """Run one training epoch with mini-batching. Returns average loss."""
    model.train()
    head = train_triples["head_index"].to(device)
    rel = train_triples["rel_type"].to(device)
    tail = train_triples["tail_index"].to(device)

    n = head.size(0)
    perm = torch.randperm(n, device=device)
    total_loss = 0.0
    num_batches = 0

    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        loss = model.loss(head[idx], rel[idx], tail[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, triples, device, batch_size=256):
    """Evaluate using PyG's filtered ranking. Returns metrics dict."""
    model.eval()
    head = triples["head_index"].to(device)
    rel = triples["rel_type"].to(device)
    tail = triples["tail_index"].to(device)

    mean_rank, mrr, hits_at_10 = model.test(
        head, rel, tail, batch_size=batch_size, k=10,
    )
    return {
        "mean_rank": mean_rank,
        "mrr": mrr,
        "hits@10": hits_at_10,
    }


def save_checkpoint(model, epoch, config, metrics, model_type, checkpoint_dir):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = checkpoint_dir / f"{ts}_{model_type}_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "model_type": model_type,
                "hidden_dim": config.hidden_dim,
                "output_dim": config.output_dim,
                "kge_margin": config.kge_margin,
                "kge_p_norm": config.kge_p_norm,
            },
            "metrics": metrics,
        },
        path,
    )
    logger.info(f"Checkpoint saved: {path}")
    return path


def save_embeddings(model, metadata, model_type, metrics, checkpoint_dir):
    """Export trained entity embeddings."""
    embeddings = model.get_entity_embeddings().cpu()
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
    parser = argparse.ArgumentParser(description="Train KGE for link prediction on RDF")
    parser.add_argument("--model", type=str, default="transe", choices=list(VALID_MODELS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--p-norm", type=float, default=1.0)
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
        batch_size=args.batch_size,
        kge_model_type=args.model,
        kge_margin=args.margin,
        kge_p_norm=args.p_norm,
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

    # 1. Load triples
    builder = RDFTripleBuilder()
    triples, metadata = builder.build()
    num_nodes = triples["num_nodes"]
    num_relations = triples["num_relations"]
    logger.info(f"RDF: {num_nodes} nodes, {num_relations} relations, "
                f"{triples['head_index'].size(0)} triples")

    # 2. Split triples
    train_triples, val_triples, test_triples = split_triples_kge(
        triples, val_ratio=config.val_ratio, test_ratio=config.test_ratio, seed=config.seed,
    )

    # 3. Create model
    model = KGEWrapper(
        model_type=args.model,
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        margin=config.kge_margin,
        p_norm=config.kge_p_norm,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # 4. Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 5. Training loop
    best_mrr = 0.0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(model, train_triples, optimizer, device, config.batch_size)

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(model, val_triples, device)
            logger.info(f"Epoch {epoch:3d} | loss={loss:.4f} | val {format_metrics(val_metrics)}")

            if val_metrics["mrr"] > best_mrr + config.min_delta:
                best_mrr = val_metrics["mrr"]
                best_metrics = val_metrics
                patience_counter = 0
                save_checkpoint(
                    model, epoch, config, val_metrics,
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
    test_metrics = evaluate(model, test_triples, device)
    logger.info(f"Test results: {format_metrics(test_metrics)}")

    # 7. Save embeddings
    save_embeddings(model, metadata, args.model, test_metrics, config.checkpoint_dir)

    torch.cuda.empty_cache()
    logger.info("Done.")


if __name__ == "__main__":
    main()
