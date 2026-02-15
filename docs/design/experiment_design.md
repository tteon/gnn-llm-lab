# FinDER KG 실험 설계서

> **목표**: LPG(GNN) vs RDF(KGE) 기반 Graph RAG 성능 비교
> **참조**: [gnn_kge_focused_analysis.md](../analysis/gnn_kge_focused_analysis.md)

---

## 1. 실험 개요

### 1.1 Research Questions

1. **RQ1**: Graph 구조 정보가 LLM의 금융 QA 성능을 향상시키는가?
2. **RQ2**: LPG(GNN) vs RDF(KGE) 중 어떤 표현이 더 효과적인가?
3. **RQ3**: Soft prompt vs Hard prompt 중 어떤 통합 방식이 우수한가?

### 1.2 실험 조건 (4 + 2 Variants)

| ID | 실험명 | Data Source | Graph Model | LLM Integration |
|----|--------|-------------|-------------|-----------------|
| **A** | LLM Only | question text | - | Direct prompt |
| **B** | Text RAG | references | - | Context injection |
| **C1** | GNN-Soft | finderlpg | GATv2 | Soft prompt (MLP) |
| **C2** | GNN-Hard | finderlpg | GATv2 | Hard prompt (text) |
| **D1** | KGE-Soft | finderrdf | RotatE | Soft prompt (MLP) |
| **D2** | KGE-Hard | finderrdf | RotatE | Hard prompt (text) |

### 1.3 데이터셋 분할

```
Total: 3,238 samples (FinDER_KG_Merged.parquet)

Split:
├── Train: 2,590 (80%) - GNN/KGE 학습용
├── Val:   324 (10%)   - 하이퍼파라미터 튜닝
└── Test:  324 (10%)   - 최종 평가
```

**Stratified split by category** 권장 (category 분포 유지)

---

## 2. 모델 아키텍처

### 2.1 Experiment A: LLM Only (Baseline)

```
Input:  Question
Output: Answer

Prompt Template:
┌────────────────────────────────────────┐
│ You are a financial expert.            │
│ Answer the following question.         │
│                                        │
│ Question: {question}                   │
│ Answer:                                │
└────────────────────────────────────────┘
```

### 2.2 Experiment B: Text RAG

```
Input:  Question + References (from parquet)
Output: Answer

Prompt Template:
┌────────────────────────────────────────┐
│ You are a financial expert.            │
│ Use the following context to answer.   │
│                                        │
│ Context:                               │
│ {references}                           │
│                                        │
│ Question: {question}                   │
│ Answer:                                │
└────────────────────────────────────────┘
```

### 2.3 Experiment C: GNN (finderlpg)

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GNN Pipeline (LPG)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Question ──► Entity Extraction ──► Neo4j Subgraph Query   │
│                                           │                 │
│                                           ▼                 │
│                              ┌─────────────────────┐        │
│                              │   Node Features     │        │
│                              │  (SentenceTransf.)  │        │
│                              │      [N, 384]       │        │
│                              └──────────┬──────────┘        │
│                                         │                   │
│                                         ▼                   │
│                              ┌─────────────────────┐        │
│                              │      GATv2          │        │
│                              │   Layer 1 (4 heads) │        │
│                              │   [N, 384] → [N, 256×4]     │
│                              └──────────┬──────────┘        │
│                                         │                   │
│                                         ▼                   │
│                              ┌─────────────────────┐        │
│                              │      GATv2          │        │
│                              │   Layer 2 (1 head)  │        │
│                              │   [N, 1024] → [N, 256]      │
│                              └──────────┬──────────┘        │
│                                         │                   │
│                                         ▼                   │
│                              ┌─────────────────────┐        │
│                              │   Graph Pooling     │        │
│                              │   (Mean/Attention)  │        │
│                              │   [N, 256] → [256]  │        │
│                              └──────────┬──────────┘        │
│                                         │                   │
│              ┌──────────────────────────┴───────────────┐   │
│              │                                          │   │
│              ▼                                          ▼   │
│    ┌─────────────────┐                      ┌───────────────┐
│    │   C1: Soft      │                      │  C2: Hard     │
│    │   MLP Projection│                      │  Text Format  │
│    │   [256] → [4096]│                      │  Graph → Text │
│    └────────┬────────┘                      └───────┬───────┘
│             │                                       │        │
│             ▼                                       ▼        │
│    ┌─────────────────────────────────────────────────────┐  │
│    │                    LLM (Llama 3.1 8B)               │  │
│    │              Generate Answer                         │  │
│    └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### Hyperparameters (from analysis)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Node feature dim | 384 | SentenceTransformer output |
| GATv2 hidden dim | 256 | Balance capacity/efficiency |
| GATv2 heads | 4, 1 | Multi-head → single-head |
| GATv2 layers | 2 | 2-hop neighborhood |
| Dropout | 0.3 | Prevent overfitting |
| Pooling | Mean | Simple, effective |
| Neighbor sampling | [15, 10] | Handle hub nodes (max 3121) |

### 2.4 Experiment D: KGE (finderrdf)

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KGE Pipeline (RDF)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Question ──► Entity Extraction ──► Neo4j Triple Query     │
│                                           │                 │
│                                           ▼                 │
│                              ┌─────────────────────┐        │
│                              │   Triples (h, r, t) │        │
│                              │      [M, 3]         │        │
│                              └──────────┬──────────┘        │
│                                         │                   │
│                                         ▼                   │
│                              ┌─────────────────────┐        │
│                              │      RotatE         │        │
│                              │  Entity Emb [E, 400]│        │
│                              │  Relation Emb [R,200]       │
│                              └──────────┬──────────┘        │
│                                         │                   │
│                                         ▼                   │
│                              ┌─────────────────────┐        │
│                              │   Triple Pooling    │        │
│                              │  Aggregate relevant │        │
│                              │  entity embeddings  │        │
│                              │   → [256]           │        │
│                              └──────────┬──────────┘        │
│                                         │                   │
│              ┌──────────────────────────┴───────────────┐   │
│              │                                          │   │
│              ▼                                          ▼   │
│    ┌─────────────────┐                      ┌───────────────┐
│    │   D1: Soft      │                      │  D2: Hard     │
│    │   MLP Projection│                      │  Text Format  │
│    │   [256] → [4096]│                      │  Triples→Text │
│    └────────┬────────┘                      └───────┬───────┘
│             │                                       │        │
│             ▼                                       ▼        │
│    ┌─────────────────────────────────────────────────────┐  │
│    │                    LLM (Llama 3.1 8B)               │  │
│    │              Generate Answer                         │  │
│    └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### Hyperparameters (from analysis)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Entity emb dim | 400 | Complex space (200 real + 200 imag) |
| Relation emb dim | 200 | Rotation phase |
| Margin | 6.0 | Standard for RotatE |
| Negative samples | 256 | Per positive triple |
| Batch size | 1024 | Triple batches |
| Learning rate | 1e-4 | Adam optimizer |
| Epochs | 500 | Early stopping on MRR |

---

## 3. 서브그래프 추출 전략

### 3.1 LPG Subgraph Extraction (finderlpg)

```cypher
// Step 1: Find entities mentioned in question
MATCH (q:Question {id: $question_id})
MATCH (e:Entity)
WHERE e.question_ids CONTAINS $question_id
RETURN e

// Step 2: Expand 2-hop neighborhood
MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
WHERE e.question_ids CONTAINS $question_id
RETURN e, r1, n1, r2, n2
LIMIT 100  // Prevent explosion
```

**PCST Pruning** (Optional, G-Retriever 방식):
```python
def pcst_subgraph(nodes, edges, question_embedding, max_nodes=50):
    """Prize-Collecting Steiner Tree for subgraph selection"""
    # Prize: relevance to question (cosine similarity)
    prizes = cosine_similarity(node_embeddings, question_embedding)

    # Cost: inverse of edge importance
    costs = 1 / edge_weights

    # Solve PCST
    selected_nodes, selected_edges = pcst_fast(prizes, costs)

    return selected_nodes[:max_nodes], selected_edges
```

### 3.2 RDF Triple Extraction (finderrdf)

```cypher
// Find triples related to question entities
MATCH (s:Resource)-[p]->(o:Resource)
WHERE s.question_ids CONTAINS $question_id
   OR o.question_ids CONTAINS $question_id
RETURN s.uri as head, type(p) as relation, o.uri as tail
LIMIT 50
```

**Relevance-based Filtering**:
```python
def filter_triples_by_relevance(triples, question, kge_model, top_k=20):
    """Select most relevant triples using KGE embeddings"""
    question_entities = extract_entities(question)

    scores = []
    for h, r, t in triples:
        # Score based on entity similarity to question entities
        h_emb = kge_model.entity_embedding(h)
        t_emb = kge_model.entity_embedding(t)
        score = max(
            cosine_similarity(h_emb, q_emb) for q_emb in question_entities
        )
        scores.append(score)

    top_indices = np.argsort(scores)[-top_k:]
    return [triples[i] for i in top_indices]
```

---

## 4. LLM Integration Methods

### 4.1 Soft Prompt (C1, D1)

**G-Retriever 방식**: Graph embedding을 LLM input space로 projection

```python
class SoftPromptProjector(nn.Module):
    """Project graph embedding to LLM token space"""
    def __init__(self, graph_dim=256, llm_dim=4096, num_tokens=10):
        super().__init__()
        self.num_tokens = num_tokens
        self.projector = nn.Sequential(
            nn.Linear(graph_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim * num_tokens),
        )

    def forward(self, graph_embedding):
        # graph_embedding: [batch, 256]
        projected = self.projector(graph_embedding)  # [batch, 4096 * 10]
        return projected.view(-1, self.num_tokens, 4096)  # [batch, 10, 4096]
```

**LLM Forward with Soft Prompt**:
```python
def forward_with_soft_prompt(llm, tokenizer, question, graph_embedding, projector):
    # 1. Project graph to soft tokens
    soft_tokens = projector(graph_embedding)  # [1, 10, 4096]

    # 2. Tokenize question
    text_tokens = tokenizer(question, return_tensors="pt")
    text_embeds = llm.get_input_embeddings()(text_tokens.input_ids)

    # 3. Concatenate: [soft_tokens] + [text_tokens]
    combined_embeds = torch.cat([soft_tokens, text_embeds], dim=1)

    # 4. Generate
    outputs = llm.generate(
        inputs_embeds=combined_embeds,
        max_new_tokens=256,
    )
    return tokenizer.decode(outputs[0])
```

### 4.2 Hard Prompt (C2, D2)

**Graph → Text Formatting**:

```python
# LPG (Structured format)
def format_lpg_for_llm(nodes, edges, max_nodes=20, max_edges=30):
    text = "Knowledge Graph Context:\n\n"

    # Entities
    text += "ENTITIES:\n"
    for node in nodes[:max_nodes]:
        label = node.get('label', 'Entity')
        name = node.get('name', node['id'])
        text += f"- [{label}] {name}\n"

    # Relationships
    text += "\nRELATIONSHIPS:\n"
    for edge in edges[:max_edges]:
        text += f"- {edge['source']} --[{edge['type']}]--> {edge['target']}\n"

    return text

# RDF (Triple format)
def format_rdf_for_llm(triples, max_triples=30):
    text = "Knowledge Graph Triples:\n\n"

    for h, r, t in triples[:max_triples]:
        # Clean predicate name
        r_clean = r.split(':')[-1] if ':' in r else r
        text += f"({h}, {r_clean}, {t})\n"

    return text
```

**Hard Prompt Template**:
```
┌────────────────────────────────────────┐
│ You are a financial expert.            │
│ Use the following knowledge graph      │
│ context to answer the question.        │
│                                        │
│ {graph_context}                        │
│                                        │
│ Question: {question}                   │
│ Answer:                                │
└────────────────────────────────────────┘
```

---

## 5. Training Strategy

### 5.1 Two-Stage Training

```
Stage 1: Pre-train Graph Models (GNN/KGE)
├── GNN: Link prediction or node classification on full graph
└── KGE: Triple classification with negative sampling

Stage 2: End-to-End Fine-tuning (Optional)
├── Freeze LLM (LoRA optional)
├── Train projector (soft prompt) or freeze all (hard prompt)
└── Optimize on QA loss
```

### 5.2 GNN Pre-training

```python
# Task: Link Prediction
def pretrain_gnn(model, data, epochs=100):
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()

        # Positive edges
        pos_edge_index = data.edge_index

        # Negative sampling
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
        )

        # Forward
        z = model(data.x, pos_edge_index)

        # Link prediction loss
        pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 5.3 KGE Pre-training

```python
# Task: Triple Classification with RotatE
def pretrain_kge(model, triples, epochs=500):
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()

        # Sample batch
        batch_idx = torch.randint(0, len(triples), (1024,))
        batch = triples[batch_idx]  # [1024, 3]

        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

        # Positive scores
        pos_score = model.score(h, r, t)

        # Negative sampling (corrupt tail)
        neg_t = torch.randint(0, model.num_entities, (1024,))
        neg_score = model.score(h, r, neg_t)

        # Margin ranking loss
        loss = F.relu(6.0 - pos_score + neg_score).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 6. Evaluation Metrics

### 6.1 Primary Metrics (QA Performance)

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **Exact Match (EM)** | 정답과 완전 일치 | `pred.strip().lower() == gold.strip().lower()` |
| **F1 Score** | Token-level overlap | `2 * (P * R) / (P + R)` |
| **BERTScore** | Semantic similarity | `bert_score.score(preds, golds)` |

### 6.2 Secondary Metrics (Graph Model Quality)

**GNN Metrics**:
| Metric | Task | Target |
|--------|------|--------|
| AUC-ROC | Link Prediction | > 0.85 |
| AP | Link Prediction | > 0.80 |

**KGE Metrics**:
| Metric | Task | Target |
|--------|------|--------|
| MRR | Triple Ranking | > 0.30 |
| Hits@10 | Triple Ranking | > 0.50 |

### 6.3 Evaluation Code

```python
from bert_score import score as bert_score
from collections import Counter

def compute_em(pred: str, gold: str) -> float:
    """Exact Match"""
    return float(pred.strip().lower() == gold.strip().lower())

def compute_f1(pred: str, gold: str) -> float:
    """Token-level F1"""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def evaluate_all(predictions: list, golds: list) -> dict:
    """Compute all metrics"""
    ems = [compute_em(p, g) for p, g in zip(predictions, golds)]
    f1s = [compute_f1(p, g) for p, g in zip(predictions, golds)]

    # BERTScore (batch)
    P, R, F1 = bert_score(predictions, golds, lang="en", verbose=False)

    return {
        "EM": np.mean(ems),
        "F1": np.mean(f1s),
        "BERTScore_P": P.mean().item(),
        "BERTScore_R": R.mean().item(),
        "BERTScore_F1": F1.mean().item(),
    }
```

---

## 7. Implementation Checklist

### 7.1 Phase 1: Infrastructure (Week 1)

- [ ] Neo4j 연결 확인 (finderlpg, finderrdf)
- [ ] 데이터 로딩 파이프라인 구현
- [ ] Train/Val/Test 분할
- [ ] Baseline A, B 구현 및 테스트

### 7.2 Phase 2: Graph Models (Week 2-3)

- [ ] GNN 모델 구현 (GATv2)
  - [ ] Node feature extraction (SentenceTransformer)
  - [ ] Subgraph extraction from Neo4j
  - [ ] PyG Data 변환
  - [ ] Link prediction pre-training
- [ ] KGE 모델 구현 (RotatE)
  - [ ] Triple extraction from Neo4j
  - [ ] Entity/Relation vocabulary
  - [ ] RotatE training loop
  - [ ] Triple ranking evaluation

### 7.3 Phase 3: LLM Integration (Week 4)

- [ ] Soft prompt projector 구현
- [ ] Hard prompt formatter 구현
- [ ] LLM inference pipeline
- [ ] End-to-end evaluation

### 7.4 Phase 4: Experiments (Week 5-6)

- [ ] 모든 실험 조건 실행 (A, B, C1, C2, D1, D2)
- [ ] 결과 수집 및 분석
- [ ] Ablation studies
- [ ] 논문/보고서 작성

---

## 8. Expected Results & Hypotheses

### 8.1 가설

| 가설 | 예상 | 근거 |
|------|------|------|
| H1: Graph > LLM Only | B, C, D > A | 구조 정보가 추론에 도움 |
| H2: Graph > Text RAG | C, D ≥ B | 구조적 관계가 텍스트보다 명확 |
| H3: GNN ≈ KGE | C ≈ D | 표현 방식 차이는 크지 않음 |
| H4: Soft > Hard | C1, D1 > C2, D2 | Soft prompt가 정보 손실 적음 |

### 8.2 예상 결과 범위

| Experiment | EM (예상) | F1 (예상) |
|------------|-----------|-----------|
| A: LLM Only | 20-30% | 35-45% |
| B: Text RAG | 35-45% | 50-60% |
| C1: GNN-Soft | 40-50% | 55-65% |
| C2: GNN-Hard | 35-45% | 50-60% |
| D1: KGE-Soft | 40-50% | 55-65% |
| D2: KGE-Hard | 35-45% | 50-60% |

---

## 9. Resource Requirements

### 9.1 Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | A100 40GB | H100 80GB |
| RAM | 32GB | 64GB |
| Storage | 50GB | 100GB |

### 9.2 Estimated Training Time (A100)

| Component | Time |
|-----------|------|
| GNN Pre-training | 1-2 hours |
| KGE Pre-training | 2-4 hours |
| LLM Inference (3K samples) | 3-5 hours |
| **Total per experiment** | ~6-10 hours |

### 9.3 Memory Optimization

```python
# LLM quantization (if needed)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache between samples
torch.cuda.empty_cache()
```

---

## 10. File Structure

```
src/
├── data/
│   ├── loader.py           # Parquet & Neo4j 데이터 로딩
│   ├── subgraph.py         # 서브그래프 추출 (LPG/RDF)
│   └── dataset.py          # PyTorch Dataset 클래스
│
├── models/
│   ├── gnn/
│   │   ├── gatv2.py        # GATv2 인코더
│   │   └── pooling.py      # Graph pooling
│   ├── kge/
│   │   ├── rotate.py       # RotatE 모델
│   │   └── transe.py       # TransE 모델 (비교용)
│   └── projector.py        # Soft prompt projector
│
├── experiments/
│   ├── baseline.py         # Exp A, B
│   ├── gnn_experiment.py   # Exp C1, C2
│   └── kge_experiment.py   # Exp D1, D2
│
├── evaluation/
│   ├── metrics.py          # EM, F1, BERTScore
│   └── analysis.py         # 결과 분석 & 시각화
│
└── utils/                  # 기존 유틸리티 (logging, config, etc.)
```

---

## Appendix A: Key Findings from Analysis

### finderlpg (for GNN)

- **Sparse graph**: avg degree 2.21 → GAT attention 효과적
- **Hub node**: max degree 3,121 → NeighborLoader 필수
- **Multi-label**: 79%가 2개 레이블 → primary label 사용
- **Rich properties**: name, type, description → 텍스트 임베딩

### finderrdf (for KGE)

- **Long-tail predicates**: 3,371 unique → 클러스터링 권장
- **1-to-N relations**: 다수 → RotatE 적합 (TransE 제한)
- **Inverse relations**: 존재 → RotatE/ComplEx 권장
- **FIBO ontology**: 의미론적 계층 → hierarchical embedding 가능

---

## Appendix B: Quick Start

```python
# 1. Load data
from src.data.loader import FinDERDataLoader
loader = FinDERDataLoader(parquet_path, neo4j_uri)
train, val, test = loader.split(train=0.8, val=0.1, test=0.1)

# 2. Run baseline
from src.experiments.baseline import run_llm_only, run_text_rag
results_a = run_llm_only(test, llm, tokenizer)
results_b = run_text_rag(test, llm, tokenizer)

# 3. Run GNN experiment
from src.experiments.gnn_experiment import run_gnn_soft, run_gnn_hard
results_c1 = run_gnn_soft(test, gnn_model, llm, projector)
results_c2 = run_gnn_hard(test, gnn_model, llm)

# 4. Run KGE experiment
from src.experiments.kge_experiment import run_kge_soft, run_kge_hard
results_d1 = run_kge_soft(test, kge_model, llm, projector)
results_d2 = run_kge_hard(test, kge_model, llm)

# 5. Evaluate
from src.evaluation.metrics import evaluate_all
for name, results in [("A", results_a), ("B", results_b), ...]:
    metrics = evaluate_all(results.predictions, results.golds)
    print(f"{name}: {metrics}")
```
