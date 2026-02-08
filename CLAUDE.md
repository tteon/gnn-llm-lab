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
| `src/run_experiment.py` | **Main experiment runner** (3-axis: models × contexts × few-shot) |
| `src/kvcache_experiment.py` | **KV cache offloading experiment** (5-condition cold/warm latency comparison) |
| `src/attention_experiment.py` | **LPG vs RDF attention analysis** (5 conditions × attention metrics) |
| `src/models.py` | GAT (`MessagePassingGNN`) and TransE (`TransEEncoder`) models |
| `src/train_gnn.py` | GNN model training (GAT, GCN, GraphTransformer) |
| `src/train_kge.py` | KGE model training (TransE, DistMult, ComplEx, RotatE) |
| `src/evaluation.py` | Link prediction metrics (MRR, Hits@K) |
| `src/load_finder_kg.py` | Parquet → Neo4j data loader |
| `src/llm_baseline.py` | LLM baseline experiments |
| `src/experiment_colab.py` | Unified experiment runner (Colab) |
| `src/soft_vs_hard_experiment.py` | Soft vs Hard prompt comparison |
| `notebooks/kvcache_analysis.ipynb` | **KV cache experiment analysis** (12 sections, 9 charts) |
| `notebooks/finder_full_comparison.ipynb` | Main Colab experiment notebook |
| `docs/attention_experiment_design.md` | Attention experiment design doc (hypotheses, metrics, analysis plan) |

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
| `config.py` | Dataclass configs (`ExperimentConfig`, `AttentionConfig`, `FewShotConfig`, `TrainingConfig`) with validation, .env support |
| `neo4j_client.py` | Neo4j client with exponential backoff retry |
| `formatting.py` | Graph → text conversion (structured/natural/triple/csv) + `format_combined()` + `clean_rdf_uri()` / `format_rdf_cleaned()` for prefix-stripped RDF |
| `logging_config.py` | Colored structured logging with file output |
| `exceptions.py` | Custom exceptions: `ConfigurationError`, `Neo4jConnectionError`, `DataLoadError`, `ModelLoadError`, `GraphProcessingError` |
| `reproducibility.py` | Seed setting, experiment metadata tracking |
| `local_llm.py` | `LocalLLMManager` with MODEL_REGISTRY (llama8b/70b, mixtral, qwen_moe) |
| `attention.py` | `AttentionExtractor` for generation→context attention maps (supports per-head mode via `aggregate_heads="none"`) |
| `attention_analysis.py` | `AttentionAnalyzer` — entropy, entity coverage@K, prefix waste ratio, semantic density, per-head stats |
| `few_shot.py` | `FewShotSelector` centroid-nearest per-category sampling |
| `evaluation.py` | `Evaluator` (EM, F1, ROUGE, BERTScore) |

## Neo4j Databases

| Database | Type | Nodes | Edges |
|----------|------|-------|-------|
| `finderlpg` | LPG | 13,920 | 18,892 |
| `finderrdf` | RDF | 12,365 | 12,609 |

Connection: `bolt://localhost:7687`, credentials: `neo4j` / `password`

### Schema Details

See [docs/neo4j_schema.md](docs/neo4j_schema.md) for full schema and query patterns.

## Experiment Runner (`src/run_experiment.py`)

Two modes of operation:
- **Legacy mode** (`--experiments`): `llm`, `text_rag`, `graph_lpg`, `graph_rdf` → mapped to context modes
- **New mode** (`--contexts`): `none`, `text`, `lpg`, `rdf`, `lpg_rdf` + `--models` + `--few-shot`

### Context Conditions (5 types)

| Context | Source | Description |
|---------|--------|-------------|
| `none` | — | LLM only, no context |
| `text` | Parquet `references` column | Text RAG (parsed via `ast.literal_eval`) |
| `lpg` | `finderlpg` Neo4j DB | GAT on LPG → formatted graph context |
| `rdf` | `finderrdf` Neo4j DB | TransE on RDF triples → triple context |
| `lpg_rdf` | Both DBs | Combined LPG + RDF context |

### Experiment Data Flow

```
[A] none:     Question → LLM → Answer
[B] text:     Question + References → LLM → Answer
[C] lpg:      Question → Neo4j(finderlpg) → GAT → Context → LLM → Answer
[D] rdf:      Question → Neo4j(finderrdf) → TransE → Context → LLM → Answer
[E] lpg_rdf:  Question → Neo4j(both) → GAT+TransE → Combined Context → LLM → Answer
```

### Running Experiments

```bash
# Legacy mode
uv run python src/run_experiment.py --experiments llm text_rag graph_lpg graph_rdf --sample-size 50

# New mode (3-axis: models × contexts × few-shot)
uv run python src/run_experiment.py --contexts none text lpg rdf --sample-size 50 --no-bertscore
```

## Attention Experiment (`src/attention_experiment.py`)

Compares how LPG vs RDF graph serialization affects LLM attention patterns. Design doc: [`docs/attention_experiment_design.md`](docs/attention_experiment_design.md)

### 5 Conditions

| Condition | Context Source | Format |
|-----------|---------------|--------|
| `lpg_structured` | finderlpg | `GraphFormatter.format()` structured |
| `lpg_natural` | finderlpg | `GraphFormatter.format()` natural |
| `rdf_raw` | finderrdf | `GraphFormatter.format()` triple (raw URIs) |
| `rdf_cleaned` | finderrdf | `GraphFormatter.format_rdf_cleaned()` (prefix-stripped) |
| `no_context` | — | No graph context |

### Metrics

- **Attention**: entropy, entity_coverage@K, prefix_waste_ratio, semantic_density, per-head entropy stats
- **Quality**: EM, F1, ROUGE-L

### Running

```bash
# Full experiment (requires GPU + Neo4j running)
uv run python src/attention_experiment.py --sample-size 50 --model meta-llama/Meta-Llama-3.1-8B-Instruct

# Subset of conditions
uv run python src/attention_experiment.py --conditions lpg_structured rdf_raw no_context --sample-size 20
```

### Data Dependencies

- `data/processed/common_question_ids.json` — 1,332 questions with data in both LPG and RDF databases
- Neo4j databases `finderlpg` and `finderrdf` must be running

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
