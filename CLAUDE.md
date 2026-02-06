# CLAUDE.md

Instructions for Claude Code when working with this repository.

## Project Overview

GNN+LLM experiment lab comparing Graph RAG approaches (GAT on LPG vs TransE on RDF) using the FinDER Knowledge Graph dataset with Llama 3.1 8B as the generation model.

See [README.md](README.md) for architecture details and experiment descriptions.

## Tech Stack

- **Python 3.10** with UV package manager
- **Neo4j/DozerDB 5.26.3** - Graph database (Docker, multi-database)
- **PyTorch + PyTorch Geometric** - GNN models (GAT, TransE)
- **Hugging Face Transformers** - LLM (Llama 3.1 8B Instruct)
- **Sentence-Transformers** - Node embeddings (all-MiniLM-L6-v2, 384-dim)
- **Google Colab** - A100 GPU runtime

## Commands

```bash
# Setup
./setup.sh
cp .env.example .env

# Neo4j
docker-compose up -d
uv run python src/load_finder_kg.py

# Run experiments
uv run python src/experiment_colab.py

# Tests
uv run pytest tests/

# Linting
uv run ruff check src/
uv run black --check src/
```

## Key Files

| File | Purpose |
|------|---------|
| `src/load_finder_kg.py` | Parquet → Neo4j data loader |
| `src/llm_baseline.py` | LLM baseline experiments |
| `src/experiment_colab.py` | Unified experiment runner |
| `src/soft_vs_hard_experiment.py` | Soft vs Hard prompt comparison |
| `notebooks/finder_full_comparison.ipynb` | Main Colab experiment notebook |

## Utilities (`src/utils/`)

```python
from src.utils import (
    setup_logging, get_logger,       # Structured logging
    ExperimentConfig, Neo4jClient,   # Config + DB client
    GraphFormatter, set_seed,        # Formatting + reproducibility
    ExperimentTracker,               # Experiment tracking
)
```

| Module | Purpose |
|--------|---------|
| `config.py` | Dataclass configs with validation, .env support |
| `neo4j_client.py` | Neo4j client with exponential backoff retry |
| `formatting.py` | Graph → text conversion (structured/natural/triple/csv) |
| `logging_config.py` | Colored structured logging with file output |
| `exceptions.py` | Custom exceptions: `ConfigurationError`, `Neo4jConnectionError`, `DataLoadError`, `ModelLoadError`, `GraphProcessingError` |
| `reproducibility.py` | Seed setting, experiment metadata tracking |

## Neo4j Databases

| Database | Type | Nodes | Edges |
|----------|------|-------|-------|
| `finderlpg` | LPG | 13,920 | 18,892 |
| `finderrdf` | RDF | 12,365 | 12,609 |

Connection: `bolt://localhost:7687`, credentials: `neo4j` / `password`

## Experiment Data Flow

```
[A] LLM Only:   Question → LLM → Answer
[B] Text RAG:   Question + References → LLM → Answer
[C] Graph LPG:  Question → Neo4j → GAT → Context → LLM → Answer
[D] Graph RDF:  Question → Neo4j → TransE → Context → LLM → Answer
```

## Development Guidelines

### Code Style
- Use `src/utils/` for consistent logging and error handling
- Use specific exception types (no bare `except:`)
- Include context info in error messages
- Use `Neo4jClient` retry logic for retryable operations

### Project Conventions
- New experiments go in `notebooks/` as new notebooks
- Old notebooks move to `notebooks/_legacy/`
- Edit code in `src/`, import from `notebooks/`
- Result filenames include timestamps: `{YYYYMMDD_HHMMSS}_{name}.csv`
- Save checkpoints every 5-10 samples
- Use `ExperimentTracker` for metadata

### Git Rules
- **Co-Authored-By 금지**: 커밋 메시지에 `Co-Authored-By` 문구를 절대 포함하지 않는다
- Never commit: `.env`, `data/neo4j/`, `*.parquet`, `entity_embeddings*.npz`
- `results/` - selective commit (important results only)
- Large files via Git LFS or external storage

### GPU/Memory
- `torch.bfloat16` for memory efficiency
- `torch.cuda.empty_cache()` between samples
- `attn_implementation="eager"` when flash_attn unavailable

### Reference Implementation
See `example_codebase/neo4j-gnn-llm-example/` for STaRK QA reference:
- `STaRKQADataset.py` - Full pipeline
- `compute_pcst.py` - PCST subgraph pruning
- `train.py` - GNN+LLM training loop
