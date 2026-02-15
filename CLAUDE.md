# CLAUDE.md

Instructions for Claude Code when working with this repository.

## Project Overview

Soft-prompt Graph RAG experiment lab comparing text-serialized graph context approaches (LPG vs RDF) using the FinDER Knowledge Graph dataset with local HuggingFace models (Llama 3.1 8B Instruct, etc.).

Subgraphs are fetched from Neo4j and formatted as text (soft prompt) — no GNN/KGE embeddings are used at inference time.

See [README.md](README.md) for architecture details.

## Tech Stack

- **Python 3.10** with UV package manager
- **Neo4j/DozerDB 5.26.3** - Graph database (Docker, multi-database)
- **PyTorch** - Model loading and GPU management
- **Hugging Face Transformers** - LLM (Llama 3.1 8B Instruct, Mixtral, Qwen MoE)
- **Sentence-Transformers** - For few-shot example selection (all-MiniLM-L6-v2, 384-dim)

## Commands

```bash
# Setup
make setup                    # Install dependencies
cp .env.example .env          # Configure API keys

# Full reproducible pipeline (download → extract → Neo4j)
make pipeline

# Individual stages
make download                 # Stage 1: HuggingFace → data/raw/FinDER.parquet
make build-kg                 # Stage 2: LLM extraction → FinDER_KG_Merged.parquet
make load-neo4j               # Stage 3: Parquet → Neo4j

# Run experiments
make experiment               # Opik experiment (default config)
uv run python src/run_experiment.py --models llama8b --contexts none lpg rdf --sample-size 50

# Tests & Lint
make test
make lint
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/download_dataset.py` | **Stage 1**: HuggingFace FinDER → `data/raw/FinDER.parquet` |
| `scripts/build_kg.py` | **Stage 2**: LLM entity extraction → `FinDER_KG_Merged.parquet` |
| `src/load_finder_kg.py` | **Stage 3**: Parquet → Neo4j (finderlpg + finderrdf) |
| `src/run_experiment.py` | **Main experiment runner** (3-axis: models × contexts × few-shot) |
| `src/opik_experiment.py` | **Opik-integrated experiment driver** (tracing, evaluation, LLM-as-Judge) |
| `prompts/*.yaml` | Entity extraction/linking prompt templates (FIBO + base) |
| `Makefile` | Pipeline automation (`make pipeline`) |
| `KNOWN_ISSUES.md` | Known issues & improvement backlog |
| `docs/INDEX.md` | Documentation index |

### Legacy Files (`src/_legacy/`)

GNN/KGE/attention/kvcache 관련 코드가 보존되어 있음:
- `models.py`, `train_gnn.py`, `train_kge.py`, `evaluation.py` — GNN/KGE 모델 및 학습
- `attention_experiment.py`, `utils/attention.py`, `utils/attention_analysis.py` — Attention 분석
- `kvcache_experiment.py` — KV cache offloading 실험
- `experiment_colab.py`, `soft_vs_hard_experiment.py`, `llm_baseline.py`, `data.py` — 이전 실험

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
| `config.py` | Dataclass configs (`ExperimentConfig`, `FewShotConfig`) with validation, .env support |
| `neo4j_client.py` | Neo4j client with exponential backoff retry |
| `formatting.py` | Graph → text conversion (structured/natural/triple/csv) + `format_combined()` + `clean_rdf_uri()` / `format_rdf_cleaned()` for prefix-stripped RDF |
| `logging_config.py` | Colored structured logging with file output |
| `exceptions.py` | Custom exceptions: `ConfigurationError`, `Neo4jConnectionError`, `DataLoadError`, `ModelLoadError`, `GraphProcessingError` |
| `reproducibility.py` | Seed setting, experiment metadata tracking |
| `local_llm.py` | `LocalLLMManager` with MODEL_REGISTRY (llama8b/70b, mixtral, qwen_moe) |
| `llm_client.py` | OpenAI-compatible API client (vLLM) |
| `few_shot.py` | `FewShotSelector` centroid-nearest per-category sampling |
| `evaluation.py` | `Evaluator` (EM, F1, ROUGE, BERTScore) |

## Neo4j Databases

| Database | Type | Nodes | Edges |
|----------|------|-------|-------|
| `finderlpg` | LPG | 13,920 | 18,892 |
| `finderrdf` | RDF | 12,365 | 12,609 |

Connection: `bolt://localhost:7687`, credentials: `neo4j` / `password`

### Schema Details

See [docs/reference/neo4j_schema.md](docs/reference/neo4j_schema.md) for full schema and query patterns.

## Experiment Runner (`src/run_experiment.py`)

Two modes of operation:
- **Legacy mode** (`--experiments`): `llm`, `text_rag`, `graph_lpg`, `graph_rdf` → mapped to context modes
- **New mode** (`--contexts`): `none`, `text`, `lpg`, `rdf`, `lpg_rdf` + `--models` + `--few-shot`

### Context Conditions (5 types)

| Context | Source | Description |
|---------|--------|-------------|
| `none` | — | LLM only, no context |
| `text` | Parquet `references` column | Text RAG (parsed via `ast.literal_eval`) |
| `lpg` | `finderlpg` Neo4j DB | Subgraph → text formatted graph context |
| `rdf` | `finderrdf` Neo4j DB | Triples → prefix-stripped text context |
| `lpg_rdf` | Both DBs | Combined LPG + RDF text context |

### Experiment Data Flow (Soft Prompt)

```
[A] none:     Question → LLM → Answer
[B] text:     Question + References → LLM → Answer
[C] lpg:      Question → Neo4j(finderlpg) → GraphFormatter → Text Context → LLM → Answer
[D] rdf:      Question → Neo4j(finderrdf) → format_rdf_cleaned → Text Context → LLM → Answer
[E] lpg_rdf:  Question → Neo4j(both) → format_combined → Text Context → LLM → Answer
```

### Running Experiments

```bash
# Quick smoke test
uv run python src/run_experiment.py --models llama8b --contexts none lpg --sample-size 2 --no-bertscore

# Full matrix
uv run python src/run_experiment.py --models llama8b --contexts none text lpg rdf lpg_rdf --few-shot --sample-size 50

# Legacy mode (API-based)
uv run python src/run_experiment.py --experiments llm text_rag graph_lpg graph_rdf --sample-size 50
```

## Opik Experiment (`src/opik_experiment.py`)

Opik-integrated driver with tracing, evaluation dashboard, and LLM-as-Judge metrics.

**Prerequisites:**
- Self-hosted Opik server: `git clone https://github.com/comet-ml/opik.git && cd opik && ./opik.sh` → `http://localhost:5173`
- Install SDK: `uv pip install opik`
- `OPENAI_API_KEY` in `.env` (for LLM-as-Judge)

### Metrics

| Type | Metrics |
|------|---------|
| Heuristic | ExactMatch, TokenF1, ROUGE-1/2/L, BERTScore |
| LLM-as-Judge | AnswerRelevance, Hallucination, ContextPrecision |

### Running with Opik

```bash
# Smoke test (heuristic only)
uv run python src/opik_experiment.py \
    --models llama8b --contexts none lpg --sample-size 2 --no-judge --no-bertscore

# Full matrix + LLM-as-Judge
uv run python src/opik_experiment.py \
    --models llama8b mixtral --contexts none lpg rdf lpg_rdf \
    --few-shot --sample-size 100 --judge-model gpt-4o-mini

# Dashboard: http://localhost:5173 → Project "FinDER_GraphRAG"
```

## Reproduction Pipeline

End-to-end pipeline for reproducing the entire experiment from scratch.

### Prerequisites

- `.env` with: `HF_TOKEN`, `OPENAI_API_KEY` (for extraction), `NEO4J_*` credentials
- Docker (for Neo4j and Opik)
- GPU (for local LLM inference)

### Data Pipeline

```
HuggingFace (Linq-AI-Research/FinDER, 5,703 QA pairs)
    ↓ scripts/download_dataset.py
data/raw/FinDER.parquet (7 columns)
    ↓ scripts/build_kg.py (LLM entity extraction + FIBO URI mapping)
data/raw/FinDER_KG_Merged.parquet (+lpg_nodes, lpg_edges, rdf_triples)
    ↓ src/load_finder_kg.py
Neo4j (finderlpg + finderrdf)
    ↓ src/opik_experiment.py
Opik Dashboard (http://localhost:5173)
```

### Prompt Templates (`prompts/`)

| Prompt | Category | Output |
|--------|----------|--------|
| `fibo_extraction.yaml` | Financials | COMPANY_NAME, FINANCIAL_TERM, CURRENCY, REGULATION, INSTRUMENT |
| `fibo_linking.yaml` | Financials | OWNS, INCURRED, REPORTED, HAS_VALUE, COMPLY_WITH |
| `base_extraction.yaml` | General | Organization, Person, Location, Event, Date, Concept |
| `base_linking.yaml` | General | LOCATED_IN, WORKS_FOR, PART_OF, RELATED_TO, HAPPENED_ON |
| `entity_dedup.yaml` | All | Duplicate detection + linked_id assignment |

### KG Build Script (`scripts/build_kg.py`)

```bash
# Full extraction (~17K API calls, ~30min at 500 RPM)
uv run python scripts/build_kg.py

# Test with small subset
uv run python scripts/build_kg.py --sample-size 10

# Resume from checkpoint
uv run python scripts/build_kg.py --resume

# Use different model/rate
uv run python scripts/build_kg.py --model gpt-4o --rpm 100
```

Checkpoints: `data/intermediate/kg_extraction.jsonl` (JSONL, line-per-sample)

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

## Known Issues & Improvement Backlog

See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) for the full prioritized list of known problems and improvement tasks.

## Documentation

See [`docs/INDEX.md`](docs/INDEX.md) for the complete documentation map organized by MECE categories (Design / Analysis / Reference / External).
