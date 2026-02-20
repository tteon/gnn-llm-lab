#!/bin/bash
# =============================================================================
# Colab Setup Script for G-Retrieval Comparison Experiment
# =============================================================================
#
# Usage (in Colab cell):
#   !git clone https://github.com/tteon/gnn-llm-lab.git
#   %cd gnn-llm-lab
#   !bash scripts/colab_setup.sh
#
# Prerequisites:
#   - Colab with GPU runtime (T4 or A100)
#   - Google Drive login (auto-prompted by this script)
#
# Drive data layout (pre-uploaded):
#   gnnllm_lab_data/
#   ├── raw/
#   │   ├── FinDER.parquet              (13 MB, original HF dataset)
#   │   └── FinDER_KG_Merged.parquet    (8.1 MB, with KG columns)
#   └── finder_pyg/processed/
#       ├── train.pt  (28 MB)
#       ├── val.pt    (3.3 MB)
#       ├── test.pt   (3.4 MB)
#       ├── vocab.pt  (827 KB)
#       └── metadata.json
#
# Data resolution order:
#   1. Local files already present → use as-is
#   2. Drive cache (finder_pyg/processed/) → copy .pt files
#   3. Drive parquet (raw/) → copy + build PyG dataset
#   4. All else fails → clear error with manual upload instructions
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "  G-Retrieval Colab Setup"
echo "=============================================="

# --- 0. Mount Google Drive ---
DRIVE_MOUNT="/content/drive"
if [ -d "$DRIVE_MOUNT/MyDrive" ]; then
    echo ""
    echo "[0/4] Google Drive already mounted."
else
    echo ""
    echo "[0/4] Mounting Google Drive..."
    python3 -c "
from google.colab import drive
drive.mount('/content/drive')
" || {
        echo "  WARNING: Drive mount failed. Will proceed without Drive cache."
        echo "  Data must be present locally or setup will fail."
    }
fi

# --- 1. Install system + Python dependencies ---
echo ""
echo "[1/4] Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q torch-geometric
pip install -q sentence-transformers rouge-score scikit-learn scipy
pip install -q pandas pyarrow matplotlib

# PyG optional deps (for scatter, etc.)
pip install -q torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html 2>/dev/null || true

echo "  Done."

# --- 2. Project setup ---
echo ""
echo "[2/4] Setting up project..."
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/raw data/processed/finder_pyg/processed

echo "  Project root: $REPO_ROOT"

# --- 3. Data setup ---
echo ""
echo "[3/4] Setting up data..."

DRIVE_BASE="/content/drive/MyDrive/gnnllm_lab_data"
DRIVE_PYG="$DRIVE_BASE/finder_pyg/processed"
DRIVE_RAW="$DRIVE_BASE/raw"
LOCAL_PYG="data/processed/finder_pyg/processed"
LOCAL_RAW="data/raw"

# --- 3a. PyG processed data (.pt files) ---
if [ -f "$LOCAL_PYG/train.pt" ]; then
    echo "  [PyG] Already present locally."

elif [ -d "$DRIVE_PYG" ] && [ -f "$DRIVE_PYG/train.pt" ]; then
    echo "  [PyG] Copying from Drive cache..."
    cp "$DRIVE_PYG"/*.pt "$LOCAL_PYG/"
    cp "$DRIVE_PYG"/metadata.json "$LOCAL_PYG/" 2>/dev/null || true
    echo "  [PyG] Done. Copied from $DRIVE_PYG"

elif [ -f "$LOCAL_RAW/FinDER_KG_Merged.parquet" ] || [ -f "$DRIVE_RAW/FinDER_KG_Merged.parquet" ]; then
    # Parquet available → build PyG dataset
    if [ ! -f "$LOCAL_RAW/FinDER_KG_Merged.parquet" ] && [ -f "$DRIVE_RAW/FinDER_KG_Merged.parquet" ]; then
        echo "  [Parquet] Copying from Drive..."
        cp "$DRIVE_RAW/FinDER_KG_Merged.parquet" "$LOCAL_RAW/"
    fi

    echo "  [PyG] Building dataset from parquet (this takes ~2 min)..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from src.data import FinDERGraphQADataset
ds = FinDERGraphQADataset(root='data/processed/finder_pyg', split='train')
print(f'  [PyG] Built dataset: {len(ds)} train samples')
"

    # Cache built data to Drive for next time
    if [ -d "/content/drive/MyDrive" ]; then
        echo "  [PyG] Caching to Drive for future sessions..."
        mkdir -p "$DRIVE_PYG"
        cp "$LOCAL_PYG"/*.pt "$DRIVE_PYG/"
        cp "$LOCAL_PYG"/metadata.json "$DRIVE_PYG/" 2>/dev/null || true
    fi

else
    echo ""
    echo "  *** ERROR: No data found! ***"
    echo ""
    echo "  Expected files in Google Drive (gnnllm_lab_data/):"
    echo "    finder_pyg/processed/train.pt   (PyG dataset)"
    echo "    raw/FinDER_KG_Merged.parquet     (source parquet)"
    echo ""
    echo "  Make sure Google Drive is mounted:"
    echo "    from google.colab import drive"
    echo "    drive.mount('/content/drive')"
    echo ""
    echo "  Or upload FinDER_KG_Merged.parquet manually to data/raw/"
    echo "  and re-run this script."
    exit 1
fi

# --- 3b. Raw parquet (optional, for soft-prompt experiments) ---
if [ ! -f "$LOCAL_RAW/FinDER_KG_Merged.parquet" ] && [ -f "$DRIVE_RAW/FinDER_KG_Merged.parquet" ]; then
    echo "  [Parquet] Copying FinDER_KG_Merged.parquet from Drive..."
    cp "$DRIVE_RAW/FinDER_KG_Merged.parquet" "$LOCAL_RAW/"
fi
if [ ! -f "$LOCAL_RAW/FinDER.parquet" ] && [ -f "$DRIVE_RAW/FinDER.parquet" ]; then
    echo "  [Parquet] Copying FinDER.parquet from Drive..."
    cp "$DRIVE_RAW/FinDER.parquet" "$LOCAL_RAW/"
fi

# --- 3c. Verify data ---
echo ""
python3 -c "
import json
from pathlib import Path
meta = json.loads(Path('data/processed/finder_pyg/processed/metadata.json').read_text())
print('  Data verification:')
print(f'    Total samples: {meta[\"total_samples\"]}')
for split, n in meta['splits'].items():
    print(f'    {split}: {n}')
print(f'    LPG feature dim: {meta[\"lpg_feature_dim\"]}')
print(f'    RDF entities: {meta[\"vocab_sizes\"][\"rdf_entities\"]:,}')
print(f'    RDF relations: {meta[\"vocab_sizes\"][\"rdf_relations\"]:,}')

# Check raw parquets
import os
for name in ['FinDER_KG_Merged.parquet', 'FinDER.parquet']:
    path = f'data/raw/{name}'
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f'    {name}: {size_mb:.1f} MB')
    else:
        print(f'    {name}: not present (optional)')
"

# --- 4. Verify environment ---
echo ""
echo "[4/4] Verifying environment..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name()}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import torch_geometric
print(f'  PyG: {torch_geometric.__version__}')

from torch_geometric.utils import scatter
print(f'  scatter: available')

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn.kge import TransE, DistMult
print(f'  GATConv, TransE, DistMult: available')

from sentence_transformers import SentenceTransformer
print(f'  sentence-transformers: available')

from src.data import FinDERGraphQADataset, dual_graph_collate_fn
print(f'  src.data: available')

print()
print('  All checks passed!')
"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "  Open notebooks/g_retrieval_comparison.ipynb"
echo "  and run all cells."
echo "=============================================="
