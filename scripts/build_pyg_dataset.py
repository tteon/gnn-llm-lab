#!/usr/bin/env python3
"""Build the FinDER PyG dataset (LPG + RDF dual-graph per question).

Reads FinDER_KG_Merged.parquet + lpg_full_graph.pt, filters to samples with
both LPG and RDF, performs stratified split, and saves PyG InMemoryDataset.

Usage:
    uv run python scripts/build_pyg_dataset.py
    uv run python scripts/build_pyg_dataset.py --force-reload
    uv run python scripts/build_pyg_dataset.py --output data/processed/finder_pyg --seed 42
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import FinDERGraphQADataset

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build FinDER PyG dataset (dual LPG+RDF graphs)"
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/raw/FinDER_KG_Merged.parquet",
        help="Input parquet path (default: data/raw/FinDER_KG_Merged.parquet)",
    )
    parser.add_argument(
        "--lpg-cache",
        type=str,
        default="data/processed/lpg_full_graph.pt",
        help="LPG full graph cache path (default: data/processed/lpg_full_graph.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/finder_pyg",
        help="Output root directory (default: data/processed/finder_pyg)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Val split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force re-processing even if output exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate inputs
    if not Path(args.parquet).exists():
        logger.error(f"Parquet not found: {args.parquet}")
        logger.error("Run 'make build-kg' first.")
        sys.exit(1)

    if not Path(args.lpg_cache).exists():
        logger.error(f"LPG cache not found: {args.lpg_cache}")
        logger.error("Run 'make load-neo4j' first (builds lpg_full_graph.pt).")
        sys.exit(1)

    # Build dataset (process() is called automatically if files don't exist)
    logger.info("Building FinDER PyG dataset...")
    logger.info(f"  Parquet: {args.parquet}")
    logger.info(f"  LPG cache: {args.lpg_cache}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Seed: {args.seed}, Split: {args.train_ratio}/{args.val_ratio}/{1 - args.train_ratio - args.val_ratio:.1f}")

    # Load each split (first triggers processing, rest just load)
    splits = {}
    for i, split in enumerate(("train", "val", "test")):
        ds = FinDERGraphQADataset(
            root=args.output,
            split=split,
            parquet_path=args.parquet,
            lpg_cache_path=args.lpg_cache,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            force_reload=args.force_reload if i == 0 else False,
        )
        splits[split] = ds

    # Print statistics
    print("\n" + "=" * 60)
    print("FinDER PyG Dataset Built Successfully")
    print("=" * 60)

    metadata_path = Path(args.output) / "processed" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        print(f"Total samples: {meta['total_samples']}")
        print(f"Splits: {meta['splits']}")
        print(f"Vocab sizes: {meta['vocab_sizes']}")
        print(f"LPG feature dim: {meta['lpg_feature_dim']}")

    # Sample inspection
    for split_name, ds in splits.items():
        sample = ds[0]
        print(f"\n{split_name.upper()} ({len(ds)} samples) — sample[0]:")
        print(f"  Question: {sample.question[:80]}...")
        print(f"  LPG: {sample.lpg_num_nodes.item()} nodes, {sample.lpg_edge_index.shape[1]} edges, x={sample.lpg_x.shape}")
        print(f"  RDF: {sample.rdf_num_nodes.item()} nodes, {sample.rdf_edge_index.shape[1]} edges")

    print(f"\nOutput: {args.output}/processed/")


if __name__ == "__main__":
    main()
