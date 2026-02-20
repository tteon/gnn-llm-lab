# GNN+LLM Lab

**Comparing Graph RAG approaches on the FinDER Knowledge Graph**

LPG(Labeled Property Graph)와 RDF 그래프 표현 방식에 서로 다른 GNN 방법론(GAT, TransE, DistMult)을 적용하여 Graph RAG 성능을 비교하는 실험 프로젝트입니다.

## Architecture

### Soft-Prompt Pipeline (Main)

서브그래프를 텍스트로 직렬화하여 LLM에 전달하는 소프트 프롬프트 방식입니다.

```
                     FinDER KG (5,703 QA pairs → 2,542 dual-graph)
                  ┌──────────┴──────────┐
                  │                     │
           LPG (Neo4j)            RDF (Neo4j)
         13,920 nodes           12,365 nodes
         18,892 edges           12,609 edges
                  │                     │
                  ▼                     ▼
           GraphFormatter        format_rdf_cleaned
                  │                     │
                  └──────────┬──────────┘
                             ▼
                      Text Context
                             │
                             ▼
                   ┌───────────────────┐
                   │  Llama 3.1 8B     │
                   │  (vLLM API)       │
                   └─────────┬─────────┘
                             ▼
                          Answer
```

### G-Retrieval Comparison (Colab)

학습된 그래프 임베딩으로 답변 정보 capture 능력을 비교합니다.

```
                  PyG Dataset (2,542 QA, train/val/test split)
                  ┌──────────┴──────────┐
                  │                     │
          LPG Subgraphs           RDF Triples
          384d node features      17K entities, 4K relations
                  │                     │
                  ▼                     ├──────────┐
         ┌──────────────┐      ┌──────────┐  ┌──────────┐
         │  GAT (2-layer │      │  TransE   │  │ DistMult │
         │  + mean pool) │      │  h+r ≈ t  │  │  h·r·t   │
         └──────┬───────┘      └────┬─────┘  └────┬─────┘
                │                    │              │
                ▼                    ▼              ▼
          384d graph emb       384d graph emb  384d graph emb
                │                    │              │
                └────────────┬───────┘──────────────┘
                             ▼
                   Retrieval Evaluation
                   (MRR, Recall@K, category breakdown)
```

## Context Conditions (5 types)

| ID | Context | Source | Description |
|----|---------|--------|-------------|
| **A** | `none` | — | LLM only, no context |
| **B** | `text` | Parquet `references` | Text RAG |
| **C** | `lpg` | Neo4j `finderlpg` | LPG subgraph → text context |
| **D** | `rdf` | Neo4j `finderrdf` | RDF triples → text context |
| **E** | `lpg_rdf` | Both DBs | Combined LPG + RDF context |

## Dataset: FinDER KG

[FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER) — Financial Domain Entity Relation dataset (Linq AI Research).

- **5,703** QA pairs (원본) → **2,542** dual-graph samples (LPG∩RDF 필터 후)
- **~14K** unique entities, **~20K** edges
- **8** categories: Accounting, Compliance, Corporate Governance, Economics, Finance, Financial Analysis, Insurance, Risk Management

| Column | Description |
|--------|-------------|
| `text` | Question text |
| `answer` | Ground truth answer |
| `category` | 8개 금융 카테고리 중 하나 |
| `references` | Text references (for Text RAG) |
| `lpg_nodes` / `lpg_edges` | LPG graph structure (JSON) |
| `rdf_triples` | RDF triples (JSON) |

### PyG Dataset

`data/processed/finder_pyg/processed/` — 사전 처리된 PyG InMemoryDataset:

| Split | Samples | File |
|-------|---------|------|
| Train | 2,030 | `train.pt` (28 MB) |
| Val | 251 | `val.pt` (3.3 MB) |
| Test | 261 | `test.pt` (3.4 MB) |

각 샘플: LPG subgraph (384d node features + edges) + RDF subgraph (entity/relation indices + edges) + question/answer/category.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10 |
| Package Manager | [UV](https://github.com/astral-sh/uv) |
| Graph DB | Neo4j / DozerDB 5.26 (Docker, multi-database) |
| GNN | PyTorch + PyTorch Geometric |
| KGE | TransE, DistMult (PyG KGE) |
| LLM | Llama 3.1 8B Instruct (vLLM API) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Experiment Tracking | [Opik](https://github.com/comet-ml/opik) (self-hosted) |
| GPU | Google Colab T4/A100, local vLLM |

## Quick Start

### Option 1: Colab (G-Retrieval Comparison)

```python
# Colab Cell 1: Clone & setup
!git clone https://github.com/tteon/gnn-llm-lab.git
%cd gnn-llm-lab

# Colab Cell 2: Mount Drive (데이터 캐시)
from google.colab import drive
drive.mount('/content/drive')

# Colab Cell 3: 의존성 설치 + 데이터 준비
!bash scripts/colab_setup.sh

# 이후 notebooks/g_retrieval_comparison.ipynb 실행
```

`colab_setup.sh`가 하는 일:
1. PyTorch + PyG + sentence-transformers 설치
2. Drive (`gnnllm_lab_data/finder_pyg/processed/`)에서 캐시된 PyG 데이터 복사
3. 캐시 없으면: HuggingFace에서 parquet 다운로드 → PyG 데이터셋 빌드 → Drive에 캐시
4. 환경 검증 (CUDA, imports)

### Option 2: Local (Soft-Prompt Experiments)

```bash
# 1. Setup
git clone https://github.com/tteon/gnn-llm-lab.git
cd gnn-llm-lab
make setup
cp .env.example .env  # API keys, Neo4j credentials 설정

# 2. Reproducible pipeline (download → KG build → Neo4j)
make pipeline

# 3. Run experiments
# Soft-prompt (text-serialized graph context)
uv run python src/run_experiment.py \
    --models llama8b --contexts none lpg rdf lpg_rdf --sample-size 50

# With Opik tracing + LLM-as-Judge
uv run python src/opik_experiment.py \
    --models llama8b --contexts none lpg rdf --sample-size 50 --no-judge
```

## Reproducible Pipeline

```
HuggingFace (Linq-AI-Research/FinDER, 5,703 QA)
    ↓ scripts/download_dataset.py
data/raw/FinDER.parquet
    ↓ scripts/build_kg.py (LLM extraction + FIBO URI mapping)
data/raw/FinDER_KG_Merged.parquet (+lpg_nodes, lpg_edges, rdf_triples)
    ↓ src/load_finder_kg.py
Neo4j (finderlpg + finderrdf)
    ↓ src/data/finder_dataset.py
data/processed/finder_pyg/ (PyG InMemoryDataset)
    ↓ notebooks/g_retrieval_comparison.ipynb  OR  src/opik_experiment.py
Results (Opik Dashboard / notebook outputs)
```

## Project Structure

```
gnn-llm-lab/
├── src/
│   ├── data/                         # PyG dataset & utilities
│   │   ├── finder_dataset.py         #   FinDERGraphQADataset (InMemoryDataset)
│   │   ├── collate.py                #   DualGraphBatch + dual_graph_collate_fn
│   │   ├── graph_builders.py         #   build_lpg_subgraph, build_rdf_subgraph
│   │   └── vocabulary.py             #   Vocabulary + VocabularyBuilder
│   ├── utils/                        # Shared utilities
│   │   ├── config.py                 #   ExperimentConfig, FewShotConfig
│   │   ├── neo4j_client.py           #   Neo4j client with retry
│   │   ├── formatting.py             #   Graph → text (structured/natural/triple/csv)
│   │   ├── evaluation.py             #   Evaluator (EM, F1, ROUGE, BERTScore)
│   │   ├── local_llm.py              #   LocalLLMManager + MODEL_REGISTRY
│   │   ├── llm_client.py             #   OpenAI-compatible API client (vLLM)
│   │   ├── few_shot.py               #   FewShotSelector (centroid-nearest)
│   │   ├── logging_config.py         #   Structured logging
│   │   ├── exceptions.py             #   Custom exceptions
│   │   └── reproducibility.py        #   Seed setting, experiment tracking
│   ├── run_experiment.py             # Main experiment runner (3-axis)
│   ├── opik_experiment.py            # Opik-integrated driver
│   ├── load_finder_kg.py             # Parquet → Neo4j loader
│   └── _legacy/                      # GNN/KGE/attention (archived)
│
├── notebooks/
│   ├── g_retrieval_comparison.ipynb   # G-Retrieval: GAT vs TransE vs DistMult
│   ├── finder_full_comparison.ipynb   # Soft-prompt full comparison
│   └── soft_vs_hard_comparison.ipynb  # Prompt strategy comparison
│
├── scripts/
│   ├── colab_setup.sh                # Colab environment setup
│   ├── download_dataset.py           # Stage 1: HuggingFace → parquet
│   └── build_kg.py                   # Stage 2: LLM extraction → KG
│
├── prompts/                          # YAML prompt templates
│   ├── fibo_extraction.yaml          #   FIBO entity extraction (Financials)
│   ├── fibo_linking.yaml             #   FIBO relationship linking
│   ├── base_extraction.yaml          #   General entity extraction
│   ├── base_linking.yaml             #   General relationship linking
│   └── entity_dedup.yaml             #   Entity deduplication
│
├── docs/                             # Documentation (MECE structure)
│   ├── design/                       #   Architecture & design decisions
│   ├── analysis/                     #   Experiment analysis notes
│   ├── reference/                    #   Neo4j schema, API docs
│   └── INDEX.md                      #   Documentation map
│
├── data/
│   ├── raw/                          # FinDER.parquet, FinDER_KG_Merged.parquet
│   └── processed/finder_pyg/         # PyG dataset (train/val/test.pt, vocab.pt)
│
├── results/                          # Experiment outputs
├── tests/                            # Unit tests
├── Makefile                          # Pipeline automation
├── pyproject.toml                    # Project config
├── CLAUDE.md                         # Claude Code instructions
└── KNOWN_ISSUES.md                   # Known issues backlog
```

## Key Notebooks

| Notebook | Purpose | Environment |
|----------|---------|-------------|
| `g_retrieval_comparison.ipynb` | GAT vs TransE vs DistMult 비교 실험 | **Colab (GPU)** |
| `finder_full_comparison.ipynb` | 5-context soft-prompt 비교 | Local + Neo4j |
| `soft_vs_hard_comparison.ipynb` | Prompt strategy 비교 | Local + Neo4j |

## References

- [FinDER Dataset](https://huggingface.co/datasets/Linq-AI-Research/FinDER) — Linq AI Research
- [G-Retriever](https://arxiv.org/abs/2402.07630) — He et al., 2024
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [FIBO Ontology](https://spec.edmcouncil.org/fibo/) — Financial Industry Business Ontology
- [Opik](https://github.com/comet-ml/opik) — LLM evaluation platform

## License

MIT
