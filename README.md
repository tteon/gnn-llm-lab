# GNN+LLM Lab

**Comparing Graph RAG approaches on the FinDER Knowledge Graph**

LPG(Labeled Property Graph)와 RDF 그래프 표현 방식에 서로 다른 GNN 방법론(GAT, TransE)을 적용하여 Graph RAG 성능을 비교하는 실험 프로젝트입니다.

## Architecture

```
                        FinDER KG (3,238 QA pairs)
                     ┌──────────┴──────────┐
                     │                     │
              LPG (Neo4j)            RDF (Neo4j)
            13,920 nodes           12,365 nodes
            18,892 edges           12,609 edges
                     │                     │
                  ┌──┘                     └──┐
                  ▼                            ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│  [C] GAT (Msg-Passing)  │  │  [D] TransE (KGE)       │
│  ───────────────────     │  │  ───────────────────     │
│  SentenceTransformer     │  │  (h, r, t) triples      │
│  384-dim node features   │  │  h + r ≈ t embedding    │
│  2-layer GAT + Pooling   │  │  Entity + Rel Embedding │
└────────────┬────────────┘  └────────────┬────────────┘
             │                             │
             ▼                             ▼
        Graph Context                 Graph Context
             │                             │
             └──────────┬──────────────────┘
                        ▼
              ┌───────────────────┐
              │  Llama 3.1 8B     │
              │  (Instruct)       │
              └─────────┬─────────┘
                        ▼
                     Answer

Baselines:
  [A] LLM Only    :  Question ──────────────────── → LLM → Answer
  [B] Text RAG    :  Question + References ──────── → LLM → Answer
```

## Experiments

| ID | Method | Data Source | Model | Description |
|----|--------|-------------|-------|-------------|
| **A** | LLM Only | Parquet (text) | Llama 3.1 8B | Pure LLM baseline |
| **B** | Text RAG | Parquet (references) | Llama 3.1 8B | Text-based RAG |
| **C** | Graph RAG (LPG) | Neo4j `finderlpg` | **GAT** | Message-Passing GNN on labeled property graph |
| **D** | Graph RAG (RDF) | Neo4j `finderrdf` | **TransE** | Knowledge graph embedding on RDF triples |

## Dataset: FinDER KG

[FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER) - Financial Domain Entity Relation dataset.

- **3,238** QA samples
- **~14K** unique entities, **~20K** edges
- Each sample includes: question, answer, LPG subgraph, RDF triples, text references

| Column | Description |
|--------|-------------|
| `text` | Question text |
| `answer` | Ground truth answer |
| `references` | Text references (for Text RAG) |
| `lpg_nodes` / `lpg_edges` | LPG graph structure (JSON) |
| `rdf_triples` | RDF triples (JSON) |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10 |
| Package Manager | [UV](https://github.com/astral-sh/uv) |
| Graph DB | Neo4j / DozerDB 5.26 (Docker) |
| GNN | PyTorch + PyTorch Geometric |
| LLM | Llama 3.1 8B Instruct (HuggingFace) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| GPU | Google Colab A100 |

## Quick Start

### 1. Setup

```bash
git clone https://github.com/tteon/gnn-llm-lab.git
cd gnn-llm-lab

# Install dependencies
./setup.sh

# Configure environment
cp .env.example .env
# Edit .env with your settings (Neo4j password, HF token, etc.)
```

### 2. Start Neo4j

```bash
docker-compose up -d

# Load FinDER KG data
uv run python src/load_finder_kg.py
```

### 3. Run Experiments

**Local:**
```bash
uv run python src/experiment_colab.py
```

**Google Colab (recommended for GPU):**

1. Start ngrok tunnel: `ngrok tcp 7687`
2. Open `notebooks/finder_full_comparison.ipynb` in Colab
3. Set runtime to A100 GPU + High-RAM
4. Update ngrok address and run

## Project Structure

```
gnn-llm-lab/
├── src/
│   ├── utils/                     # Shared utilities
│   │   ├── config.py              #   Config management + validation
│   │   ├── neo4j_client.py        #   Neo4j client with retry logic
│   │   ├── formatting.py          #   Graph → text formatting
│   │   ├── logging_config.py      #   Structured logging
│   │   ├── exceptions.py          #   Custom exception classes
│   │   └── reproducibility.py     #   Seed setting, experiment tracking
│   ├── load_finder_kg.py          # Parquet → Neo4j loader
│   ├── llm_baseline.py            # LLM baseline experiments
│   ├── experiment_colab.py        # Unified experiment runner
│   └── soft_vs_hard_experiment.py # Soft vs Hard prompt comparison
│
├── notebooks/
│   ├── finder_full_comparison.ipynb    # Main experiment notebook
│   └── soft_vs_hard_comparison.ipynb   # Prompt strategy comparison
│
├── scripts/                       # Analysis scripts
├── docs/                          # Design docs & analysis
├── tests/                         # Unit tests
├── results/                       # Experiment outputs
│   ├── experiments/               #   CSV/JSON results
│   ├── figures/                   #   Visualizations
│   └── logs/                      #   Experiment logs
│
├── data/
│   ├── raw/                       # FinDER_KG_Merged.parquet
│   └── processed/                 # Preprocessed data
│
├── docker-compose.yml             # Neo4j container config
├── pyproject.toml                 # Project config
└── setup.sh                       # Setup script
```

## References

- [FinDER Dataset](https://huggingface.co/datasets/Linq-AI-Research/FinDER) - Linq AI Research
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [STaRK Benchmark](https://github.com/snap-stanford/stark) - reference implementation

## License

MIT
