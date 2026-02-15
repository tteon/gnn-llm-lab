#!/usr/bin/env python3
"""
Stage 1: Download FinDER dataset from HuggingFace.

Downloads Linq-AI-Research/FinDER and saves to data/raw/FinDER.parquet.

Usage:
    uv run python scripts/download_dataset.py
    uv run python scripts/download_dataset.py --output data/raw/FinDER.parquet
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Download FinDER dataset from HuggingFace")
    parser.add_argument(
        "--output", type=str, default="data/raw/FinDER.parquet",
        help="Output parquet path (default: data/raw/FinDER.parquet)",
    )
    parser.add_argument(
        "--dataset-id", type=str, default="Linq-AI-Research/FinDER",
        help="HuggingFace dataset ID",
    )
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set in .env — may fail for gated datasets")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Output already exists: {output_path}")
        print("Delete it first if you want to re-download.")
        sys.exit(0)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Install with: uv pip install datasets")
        sys.exit(1)

    print(f"Downloading dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id, token=hf_token)

    df = ds["train"].to_pandas()
    print(f"Downloaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")

    df.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
