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
#   - Google Drive mounted at /content/drive (for data persistence)
#
# Data flow:
#   1. Check Drive for cached PyG data → skip processing if found
#   2. Otherwise: download FinDER parquet from HuggingFace + build PyG dataset
#   3. Install Python dependencies
#   4. Ready to run notebooks/g_retrieval_comparison.ipynb
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "  G-Retrieval Colab Setup"
echo "=============================================="

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

# Create necessary directories
mkdir -p data/raw data/processed/finder_pyg/processed

echo "  Project root: $REPO_ROOT"

# --- 3. Data: check Drive cache or download ---
echo ""
echo "[3/4] Setting up data..."

DRIVE_CACHE="/content/drive/MyDrive/gnnllm_lab_data/finder_pyg/processed"
LOCAL_DATA="data/processed/finder_pyg/processed"
PARQUET_PATH="data/raw/FinDER_KG_Merged.parquet"

# Check if PyG processed data exists in Drive
if [ -d "$DRIVE_CACHE" ] && [ -f "$DRIVE_CACHE/train.pt" ]; then
    echo "  Found cached PyG data in Drive. Copying..."
    cp -v "$DRIVE_CACHE"/*.pt "$LOCAL_DATA/"
    cp -v "$DRIVE_CACHE"/metadata.json "$LOCAL_DATA/" 2>/dev/null || true
    echo "  Copied from Drive cache."

# Check if already present locally
elif [ -f "$LOCAL_DATA/train.pt" ]; then
    echo "  PyG data already present locally."

else
    echo "  No cached data found. Will build from scratch."
    echo ""

    # Download FinDER parquet from HuggingFace if needed
    if [ ! -f "$PARQUET_PATH" ]; then
        echo "  Downloading FinDER_KG_Merged.parquet..."
        echo "  (If this fails, upload the parquet manually to data/raw/)"
        pip install -q huggingface_hub
        python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id='Linq-AI-Research/FinDER',
    filename='FinDER_KG_Merged.parquet',
    repo_type='dataset',
)
shutil.copy(path, 'data/raw/FinDER_KG_Merged.parquet')
print('  Downloaded to data/raw/FinDER_KG_Merged.parquet')
" 2>/dev/null || {
            echo ""
            echo "  *** HuggingFace download failed. ***"
            echo "  Please upload FinDER_KG_Merged.parquet manually:"
            echo "    Option A: Upload to data/raw/ in Colab file browser"
            echo "    Option B: Mount Drive and copy from your Drive"
            echo ""
            echo "  Then re-run this script."
            exit 1
        }
    fi

    # Build PyG dataset
    echo "  Building PyG dataset from parquet (this takes ~2 min)..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from src.data import FinDERGraphQADataset
# This triggers process() which builds train/val/test.pt + vocab.pt
ds = FinDERGraphQADataset(root='data/processed/finder_pyg', split='train')
print(f'  Built dataset: {len(ds)} train samples')
"

    # Cache to Drive for next time
    if [ -d "/content/drive/MyDrive" ]; then
        echo "  Caching PyG data to Drive for future sessions..."
        mkdir -p "$DRIVE_CACHE"
        cp "$LOCAL_DATA"/*.pt "$DRIVE_CACHE/"
        cp "$LOCAL_DATA"/metadata.json "$DRIVE_CACHE/" 2>/dev/null || true
        echo "  Cached to $DRIVE_CACHE"
    fi
fi

# Verify data
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
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

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
