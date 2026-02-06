# FinDER KG: GNN vs KGE Focused Analysis

> **분석 목적**: GNN은 finderlpg에서, Knowledge Graph Embedding은 finderrdf에서 사용할 예정. 각 데이터베이스의 특성을 해당 모델 관점에서 분석.

---

## Part 1: finderlpg → GNN (Message-Passing)

### 1.1 개요

| 항목 | 값 | GNN 관점 |
|------|-----|----------|
| Total Nodes | 17,060 | 적절한 규모, 메모리 효율적 |
| Total Edges | 18,892 | Sparse graph (avg degree 2.21) |
| Unique Labels | 2,497+ | Multi-label 고려 필요 |
| Unique Rel Types | 2,971 | Edge feature로 활용 가능 |

### 1.2 Node Features for GNN

#### 텍스트 기반 속성 (Primary Features)

| Property | Coverage | Feature Generation |
|----------|----------|-------------------|
| `name` | 7,401 nodes (43%) | SentenceTransformer → 384-dim |
| `label` | 13,920 nodes (82%) | One-hot 또는 Label Embedding |
| `type` | 5,021 nodes (29%) | Categorical encoding |
| `description` | 1,281 nodes (8%) | Text embedding (optional) |
| `text` | 3,141 nodes (18%) | Long-form embedding |

**권장 Feature 조합**:
```python
# Primary: 텍스트 임베딩
text = f"{label}: {name}"  # "Company: Apple Inc"
node_feat = sentence_transformer.encode(text)  # [384]

# Optional: Concatenate with label embedding
label_emb = label_encoder[node_label]  # [32]
node_feat = concat(text_emb, label_emb)  # [416]
```

#### 구조적 속성 (Structural Features)

| Property | Coverage | Feature Type |
|----------|----------|--------------|
| `category` | 3,184 nodes | Categorical |
| `sentiment` | 1,426 nodes | Float [-1, 1] |
| `risk` | 1,516 nodes | Categorical/Ordinal |
| `amount` | 760 nodes | Numerical (normalize) |
| `year` | 622 nodes | Temporal |

### 1.3 Degree Distribution Analysis

```
Min: 0, Max: 3,121, Avg: 2.21
Median: 1, P90: 4, P99: 18
```

**문제점 및 해결책**:

| 문제 | 값 | 해결책 |
|------|-----|--------|
| Isolated nodes | 3,765 (22%) | Question 노드 포함. 서브그래프 추출시 제외 가능 |
| Hub node | 1개 (degree 3,121) | Neighbor sampling 필수 |
| Low-degree nodes | 14,552 (85%, deg≤2) | PCST pruning으로 관리 |

**GAT/GraphSAGE를 위한 샘플링 전략**:
```python
# NeighborLoader 설정 (PyG)
loader = NeighborLoader(
    data,
    num_neighbors=[15, 10],  # 2-hop: 15 → 10
    batch_size=32,
)
```

### 1.4 Multi-Label Analysis

| # Labels per Node | Count | 비율 |
|-------------------|-------|------|
| 1 | 3,521 | 21% |
| 2 | 13,539 | 79% |

**대부분 2개 레이블** (e.g., `Entity:Company`). GNN 설계 시:

```python
# Option 1: Primary label만 사용
primary_label = labels[0] if 'Entity' not in labels[0] else labels[1]

# Option 2: Multi-hot encoding
multi_hot = torch.zeros(num_label_types)
for lbl in labels:
    multi_hot[label2idx[lbl]] = 1
```

### 1.5 Relationship Types (Edge Features)

**Top 10 Relationship Types**:

| Relationship | Count | 비율 |
|--------------|-------|------|
| COMPETES_WITH | 2,229 | 11.8% |
| INCLUDES | 1,358 | 7.2% |
| INVOLVES | 247 | 1.3% |
| EMPLOYS | 208 | 1.1% |
| isDomiciledIn | 199 | 1.1% |
| includes | 196 | 1.0% |
| RELATED_TO | 186 | 1.0% |
| OPERATES_IN | 173 | 0.9% |
| ... | ... | ... |

**2,971개의 고유 관계 타입** → Edge type embedding 권장

```python
# Heterogeneous GNN approach (HGT/HAN)
edge_types = [('Company', 'COMPETES_WITH', 'Company'),
              ('Company', 'EMPLOYS', 'Person'), ...]

# Or: Edge feature embedding
edge_feat = relation_embedding[rel_type]  # [64]
```

### 1.6 Graph Connectivity

| Metric | Value | 의미 |
|--------|-------|------|
| Isolated nodes | 3,765 | Question 노드 대부분 |
| Self-loops | 23 | 무시 가능 |
| Bidirectional pairs | 961 | 약 5% 대칭 관계 |
| Avg clustering coeff | 0.53 | 적당한 클러스터링 |

### 1.7 GNN Architecture 권장사항

#### GAT (Graph Attention Network) ✅ 추천

```python
class GATEncoder(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=256, out_dim=128, heads=4):
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  # [N, 128]
```

**적합 이유**:
- Sparse graph에서 attention이 효과적
- Hub node에서 중요 이웃만 집중
- Self-attention으로 노드별 가중치 학습

#### GATv2 (Improved Attention) ✅ 추천

```python
# Static attention problem 해결
self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=4)
```

**GAT vs GATv2**:
- GAT: `attention(q, k) = LeakyReLU(a^T [Wq || Wk])`
- GATv2: `attention(q, k) = a^T LeakyReLU(W [q || k])` → **동적 attention**

#### GraphTransformer 🔶 대안

```python
# Positional encoding 추가 필요
from torch_geometric.nn import TransformerConv

class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        self.conv1 = TransformerConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim)
```

**고려사항**:
- RWPE (Random Walk Positional Encoding) 추가 시 성능 향상
- 계산 비용이 GAT보다 높음

### 1.8 Feature Engineering Pipeline

```python
def prepare_lpg_for_gnn(nodes, edges, sentence_transformer):
    """LPG 데이터를 GNN 입력으로 변환"""

    # 1. Node features: 텍스트 임베딩
    texts = [f"{n.get('label', 'Entity')}: {n.get('name', n['id'])}"
             for n in nodes]
    node_features = sentence_transformer.encode(texts)  # [N, 384]

    # 2. Label encoding (optional)
    label_vocab = build_label_vocab(nodes)
    label_ids = [label_vocab[n['label']] for n in nodes]

    # 3. Edge index
    node2idx = {n['id']: i for i, n in enumerate(nodes)}
    edge_index = torch.tensor([
        [node2idx[e['source']], node2idx[e['target']]]
        for e in edges if e['source'] in node2idx and e['target'] in node2idx
    ]).T  # [2, E]

    # 4. Edge type encoding
    rel_vocab = build_relation_vocab(edges)
    edge_type = torch.tensor([rel_vocab[e['type']] for e in edges])

    return Data(
        x=torch.tensor(node_features),
        edge_index=edge_index,
        edge_attr=edge_type,
        y=label_ids  # for node classification, if needed
    )
```

---

## Part 2: finderrdf → KGE (TransE/RotatE)

### 2.1 개요

| 항목 | 값 | KGE 관점 |
|------|-----|----------|
| Total Entities | 15,505 | Entity embedding 크기 |
| Total Triples | 12,609 | 학습 데이터 크기 |
| Unique Predicates | 3,371 | Relation embedding 크기 |
| Head Entities | 3,136 | 주로 주어 역할 |
| Tail Entities | 7,955 | 주로 목적어 역할 |

### 2.2 Predicate (Relation) Distribution

**Top 15 Predicates**:

| Predicate | Count | 비율 |
|-----------|-------|------|
| `fibo-fnd-rel-rel:competesWith` | 986 | 7.8% |
| `fibo-fnd-rel-rel:includes` | 581 | 4.6% |
| `rdf:type` | 341 | 2.7% |
| `fibo-fnd-rel-rel:hasCompetitor` | 198 | 1.6% |
| `fibo-fnd-rel-rel:involves` | 180 | 1.4% |
| `fibo-fnd-rel-rel:hasPosition` | 156 | 1.2% |
| `fibo-be-le-lp:isDomiciledIn` | 147 | 1.2% |
| `fibo:includes` | 129 | 1.0% |
| `fibo-fnd-rel-rel:hasEmployee` | 100 | 0.8% |
| `fibo-fnd-rel-rel:hasCharacteristic` | 93 | 0.7% |
| `fibo-fnd-rel-rel:competesOn` | 87 | 0.7% |
| `fibo-fnd-rel-rel:operatesIn` | 86 | 0.7% |
| `fibo-fnd-rel-rel:employs` | 85 | 0.7% |
| `fibo-fnd-rel-rel:provides` | 65 | 0.5% |
| `fibo-fnd-rel-rel:hasPart` | 63 | 0.5% |

**3,371개의 고유 predicate** → Long-tail distribution

### 2.3 Relation Pattern Analysis

#### Cardinality Patterns (1-to-1, 1-to-N, N-to-1, N-to-N)

**High Fan-out Relations (1-to-N)**:

| Relation | Avg Tails per Head |
|----------|-------------------|
| `hasRemedy` | 15.00 |
| `isAffectedBy` | 12.00 |
| `hasEmployeeInRegion` | 11.00 |
| `hasFactor` | 10.00 |
| `hasPotentialClaim` | 10.00 |

**대부분의 관계가 1-to-N 또는 N-to-N 패턴** → TransE보다 **RotatE/ComplEx** 적합

#### Inverse Relations (역관계 패턴)

| Relation 1 | Relation 2 | Count |
|------------|------------|-------|
| `promiseInContract` | `hasPerformanceObligation` | 14 |
| `hasRevenueRecognition` | `hasPerformanceObligation` | 10 |

**역관계 존재** → TransE는 역관계 모델링 어려움, **RotatE** 권장

#### Symmetric Relations (대칭 관계)

| Relation | Symmetric Pairs |
|----------|-----------------|
| `fibo-fnd-rel-rel:defines` | 1 |

**대칭 관계 거의 없음** → TransE 사용 가능하나 RotatE가 더 일반적

### 2.4 FIBO Ontology Structure

finderrdf는 **FIBO (Financial Industry Business Ontology)** 기반:

```
fibo-fnd-rel-rel:  (Fundamental Relations)
├── competesWith
├── includes
├── hasCompetitor
├── involves
├── employs
└── ...

fibo-be-le-lp:  (Business Entities - Legal Persons)
└── isDomiciledIn

fibo-fnd-agr-ctr:  (Agreements - Contracts)
├── hasPerformanceObligation
├── promiseInContract
└── hasRevenueRecognition
```

**의미론적 계층 구조** → Relation clustering 또는 hierarchical relation embedding 고려

### 2.5 KGE Model 권장사항

#### TransE ⚠️ 제한적 추천

```python
# TransE: h + r ≈ t
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200):
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

    def score(self, h, r, t):
        # Distance: ||h + r - t||
        return -torch.norm(self.entity_emb(h) + self.relation_emb(r)
                          - self.entity_emb(t), dim=-1)
```

**문제점**:
- ❌ 1-to-N 관계 모델링 어려움 (많음)
- ❌ 대칭/역관계 표현 제한
- ✅ 단순하고 빠른 학습
- ✅ 해석 가능성 높음

#### RotatE ✅ 추천

```python
# RotatE: t = h ∘ r (rotation in complex space)
class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=200):
        self.entity_emb = nn.Embedding(num_entities, dim * 2)  # complex
        self.relation_emb = nn.Embedding(num_relations, dim)  # phase

    def score(self, h, r, t):
        # h, t: complex vectors [re, im]
        # r: rotation angle
        h_re, h_im = h[..., :dim], h[..., dim:]
        t_re, t_im = t[..., :dim], t[..., dim:]

        r_phase = self.relation_emb(r) / (embedding_range / pi)

        # Rotate h by r
        rotated_re = h_re * cos(r_phase) - h_im * sin(r_phase)
        rotated_im = h_re * sin(r_phase) + h_im * cos(r_phase)

        return -torch.norm(rotated_re - t_re, dim=-1) \
               -torch.norm(rotated_im - t_im, dim=-1)
```

**장점**:
- ✅ 1-to-N 관계 자연스럽게 처리
- ✅ 대칭 관계 (r = 0° or 180°)
- ✅ 역관계 (r₁ = -r₂)
- ✅ Composition (r₁ ∘ r₂)

#### ComplEx 🔶 대안

```python
# ComplEx: Re(<h, r, conj(t)>)
class ComplEx(nn.Module):
    def score(self, h, r, t):
        # All embeddings are complex
        return torch.sum(h_re * r_re * t_re + h_im * r_im * t_re
                        + h_re * r_im * t_im - h_im * r_re * t_im, dim=-1)
```

**장점**:
- ✅ Symmetric/antisymmetric 관계 모두 처리
- ✅ 이론적으로 가장 표현력 높음

### 2.6 Long-tail Predicate 문제

```
Unique predicates: 3,371
Total triples: 12,609
Average triples per predicate: 3.74
```

**대부분의 predicate가 매우 sparse** → 해결책:

1. **Predicate Clustering**: 의미적으로 유사한 predicate 그룹화
   ```python
   # FIBO namespace 기반 그룹화
   def cluster_predicates(pred):
       if 'competes' in pred.lower():
           return 'COMPETITION'
       elif 'employ' in pred.lower() or 'position' in pred.lower():
           return 'EMPLOYMENT'
       ...
   ```

2. **Hierarchical Relation Embedding**: FIBO 계층 구조 활용
   ```python
   # Parent relation embedding 공유
   rel_emb = parent_emb + specific_emb
   ```

3. **Relation Frequency Filtering**: 희소 predicate 제거
   ```python
   # 최소 5개 이상 triple만 사용
   filtered_triples = [t for t in triples if pred_count[t.pred] >= 5]
   ```

### 2.7 Training Pipeline for KGE

```python
def prepare_rdf_for_kge(triples):
    """RDF triples를 KGE 학습용으로 변환"""

    # 1. Entity/Relation vocabulary
    entities = set()
    relations = set()
    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    entity2idx = {e: i for i, e in enumerate(entities)}
    relation2idx = {r: i for i, r in enumerate(relations)}

    # 2. Index triples
    indexed_triples = torch.tensor([
        [entity2idx[h], relation2idx[r], entity2idx[t]]
        for h, r, t in triples
    ])  # [N, 3]

    return indexed_triples, entity2idx, relation2idx

def train_kge(model, triples, epochs=100, lr=0.001):
    """Negative sampling + margin ranking loss"""
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Positive triples
        pos_scores = model.score(triples[:, 0], triples[:, 1], triples[:, 2])

        # Negative sampling (corrupt tail)
        neg_t = torch.randint(0, num_entities, (len(triples),))
        neg_scores = model.score(triples[:, 0], triples[:, 1], neg_t)

        # Margin ranking loss
        loss = F.relu(margin - pos_scores + neg_scores).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 2.8 KGE → LLM Integration

```python
def kge_enhanced_context(question, model, entity2idx, idx2entity, top_k=10):
    """KGE 임베딩을 활용한 context 생성"""

    # 1. Question에서 entity 추출
    question_entities = extract_entities(question)

    # 2. 각 entity의 관련 triple 검색
    relevant_triples = []
    for ent in question_entities:
        if ent in entity2idx:
            ent_idx = entity2idx[ent]
            # Find nearest entities in embedding space
            ent_emb = model.entity_emb(ent_idx)
            similarities = cosine_similarity(ent_emb, model.entity_emb.weight)
            top_entities = similarities.topk(top_k).indices
            # ... retrieve triples involving these entities

    # 3. Triple을 자연어로 변환
    context = format_triples_for_llm(relevant_triples)

    return context
```

---

## Part 3: Implementation Roadmap

### 3.1 GNN Pipeline (finderlpg)

```
1. Data Extraction (Neo4j → PyG Data)
   └── Query subgraph per question
   └── Build node features (text embedding)
   └── Build edge index & edge types

2. GNN Training
   └── GATv2 or GraphTransformer
   └── 2-3 layers, 4 heads
   └── Output: graph-level embedding [256-dim]

3. LLM Integration
   └── Soft prompt: project GNN output → LLM space
   └── Or: Hard prompt: format graph as text
```

### 3.2 KGE Pipeline (finderrdf)

```
1. Triple Extraction (Neo4j → PyKEEN/Custom)
   └── Query triples per question
   └── Build entity/relation vocabularies

2. KGE Training
   └── RotatE (recommended) or TransE
   └── Embedding dim: 200-400
   └── Negative sampling ratio: 10-50

3. LLM Integration
   └── Retrieve relevant triples via embedding similarity
   └── Format as structured text for LLM
```

### 3.3 Comparison Experiment Design

| Experiment | Data Source | Model | Context Format |
|------------|-------------|-------|----------------|
| **[A] LLM Only** | Question text | Llama 3.1 8B | None |
| **[B] Text RAG** | references | Llama 3.1 8B | Text chunks |
| **[C] Graph LPG** | finderlpg | GAT/GATv2 + LLM | Soft prompt |
| **[D] Graph RDF** | finderrdf | RotatE + LLM | Triple text |

---

## Part 4: Key Recommendations

### For GNN (finderlpg)

1. **Architecture**: GATv2 > GAT > GraphTransformer
2. **Node Features**: SentenceTransformer (384-dim) + Label embedding (32-dim)
3. **Edge Features**: Relation type embedding (64-dim)
4. **Sampling**: NeighborLoader with [15, 10] neighbors
5. **Hub Handling**: Attention-based neighbor selection

### For KGE (finderrdf)

1. **Architecture**: RotatE > ComplEx > TransE
2. **Embedding Dim**: 200-400
3. **Predicate Handling**: Cluster or filter sparse predicates
4. **Training**: Negative sampling with margin ranking loss
5. **Integration**: Embedding-based triple retrieval

### Common

1. **PCST**: Prize-Collecting Steiner Tree로 서브그래프 pruning
2. **Soft Prompting**: G-Retriever 방식의 MLP projection
3. **Evaluation**: EM, F1, BERTScore on FinDER QA
