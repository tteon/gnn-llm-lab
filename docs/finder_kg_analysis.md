# FinDER Knowledge Graph 데이터셋 분석 보고서

> 생성일: 2026-02-06 22:42:57

## 목차
1. [개요](#1-개요)
2. [LPG (Labeled Property Graph) 분석](#2-lpg-분석)
3. [RDF (Resource Description Framework) 분석](#3-rdf-분석)
4. [LPG vs RDF 비교](#4-lpg-vs-rdf-비교)
5. [GNN Feature Engineering 권장사항](#5-gnn-feature-engineering)
6. [데이터 품질 이슈 및 개선점](#6-데이터-품질-이슈)

---

## 1. 개요

### 1.1 데이터셋 요약

| 지표 | LPG (finderlpg) | RDF (finderrdf) |
|------|-----------------|-----------------|
| **총 노드** | 17,060 | 15,505 |
| **총 엣지** | 18,892 | 12,609 |
| **질문 수** | 3,140 | 3,140 |
| **엔티티/리소스** | 13,920 | 12,365 |
| **그래프 밀도** | 0.000065 | 0.000052 |
| **평균 Degree** | 2.21 | 1.63 |

### 1.2 핵심 발견사항

```
✅ 강점:
• 3,140개 질문에 대한 Knowledge Graph 구축 완료
• FIBO 온톨로지 기반 금융 도메인 구조화
• 질문당 평균 7.18개 엔티티 연결 (LPG)

⚠️ 주의사항:
• 매우 희소한 그래프 (밀도 < 0.0001)
• Hub 노드 불균형 (max degree: 3,121)
• 관계 타입 과다 (2,971개 unique types)
• Isolated 노드 존재 (LPG: 22%, RDF: 37%)
```

---

## 2. LPG 분석

### 2.1 노드 라벨 분포

| 라벨 | 개수 | 비율 |
|------|------|------|
| Entity | 13,920 | 81.6% |
| Question | 3,140 | 18.4% |
| LegalEntity | 1,974 | 11.6% |
| Company | 588 | 3.4% |
| Competitor | 421 | 2.5% |
| Person | 409 | 2.4% |
| Share | 390 | 2.3% |
| DebtInstrument | 289 | 1.7% |
| Product | 208 | 1.2% |
| Revenue | 183 | 1.1% |
| Service | 160 | 0.9% |
| Organization | 138 | 0.8% |
| Market | 114 | 0.7% |
| Segment | 102 | 0.6% |
| Commitment | 97 | 0.6% |
| Region | 88 | 0.5% |
| Factor | 88 | 0.5% |
| Activity | 87 | 0.5% |
| Bond | 87 | 0.5% |
| Industry | 86 | 0.5% |
| *... 10개 더* | - | - |

### 2.2 관계 타입 분포

| 관계 타입 | 개수 | 비율 |
|----------|------|------|
| `COMPETES_WITH` | 2,229 | 11.8% |
| `INCLUDES` | 1,358 | 7.2% |
| `INVOLVES` | 247 | 1.3% |
| `EMPLOYS` | 208 | 1.1% |
| `isDomiciledIn` | 199 | 1.1% |
| `includes` | 196 | 1.0% |
| `RELATED_TO` | 186 | 1.0% |
| `OPERATES_IN` | 173 | 0.9% |
| `INVOLVED_IN` | 163 | 0.9% |
| `HAS_COMPETITOR` | 158 | 0.8% |
| `WORKS_FOR` | 156 | 0.8% |
| `HAS_OBLIGATION` | 147 | 0.8% |
| `PART_OF` | 145 | 0.8% |
| `HAS_SUBSIDIARY` | 140 | 0.7% |
| `HAS_SEGMENT` | 135 | 0.7% |

### 2.3 Degree 분포 통계

| 지표 | 값 |
|------|-----|
| 최소 | 0 |
| 최대 | 3,121 |
| 평균 | 2.21 |
| 표준편차 | 25.88 |
| 중앙값 (P50) | 1.0 |
| P75 | 2.0 |
| P90 | 4.0 |
| P95 | 6.0 |
| P99 | 18.0 |

**분석**: 중앙값(1.0)과 평균(2.21)의 차이가 크고, P99(18.0)와 최대값(3,121)의 차이가 매우 큼 → **극심한 Hub 노드 존재**

### 2.4 방향성 분석 (In/Out Degree)

| 지표 | In-Degree | Out-Degree |
|------|-----------|------------|
| 평균 | 1.11 | 1.11 |
| 최대 | 260 | 2,919 |
| 표준편차 | 3.97 | 23.77 |

### 2.5 Top Hub 노드

| 순위 | 노드 | 라벨 | Total Degree | Out | In |
|------|------|------|--------------|-----|-----|
| 1 | The Company | Entity/Company | 3,121 | 2,919 | 203 |
| 2 | Our Company | Entity/LegalEntity | 882 | 815 | 67 |
| 3 | Revenue Recognition | Entity/Process | 427 | 395 | 32 |
| 4 | Board of Directors | Entity/Board | 293 | 141 | 152 |
| 5 | United States | Entity/Country | 261 | 1 | 260 |
| 6 | Company | Entity | 256 | 157 | 99 |
| 7 | ex:LegalProceedings | Entity/LegalProceedings | 212 | 168 | 44 |
| 8 | ex:PerformanceObligation | Entity/PerformanceObligation | 200 | 105 | 95 |
| 9 | Our Business | Entity/Business | 193 | 191 | 2 |
| 10 | Audit Committee | Entity/Committee | 159 | 59 | 100 |
| 11 | Revenue | Entity/FinancialConcept | 157 | 102 | 55 |
| 12 | ex:Competitors | Entity | 146 | 96 | 50 |
| 13 | ex:CISO | Entity/Person | 143 | 66 | 77 |
| 14 | ex:CybersecurityProgram | Entity/Program | 133 | 101 | 32 |
| 15 | ex:VariableConsideration | Entity/VariableConsideration | 120 | 77 | 43 |

### 2.6 속성(Property) 커버리지

| 속성명 | 노드 수 | 커버리지 |
|--------|---------|----------|
| `id` | 17,060 | 100.0% |
| `label` | 13,920 | 81.6% |
| `question_ids` | 13,920 | 81.6% |
| `name` | 7,401 | 43.4% |
| `type` | 5,021 | 29.4% |
| `category` | 3,184 | 18.7% |
| `text` | 3,141 | 18.4% |
| `reasoning` | 3,140 | 18.4% |
| `answer` | 3,140 | 18.4% |
| `risk` | 1,516 | 8.9% |
| `sentiment` | 1,426 | 8.4% |
| `description` | 1,281 | 7.5% |
| `amount` | 760 | 4.5% |
| `age` | 635 | 3.7% |
| `year` | 622 | 3.6% |

### 2.7 질문-엔티티 연결

| 지표 | 값 |
|------|-----|
| 질문당 평균 엔티티 수 | 7.18 |
| 질문당 최대 엔티티 수 | 41 |
| 질문당 중앙값 엔티티 수 | 6.0 |

### 2.8 관계 패턴 (Subject → Relation → Object)

| Subject Type | Relation | Object Type | 개수 |
|--------------|----------|-------------|------|
| Entity | `COMPETES_WITH` | Entity | 2,229 |
| Entity | `INCLUDES` | Entity | 1,358 |
| Entity | `INVOLVES` | Entity | 247 |
| Entity | `EMPLOYS` | Entity | 208 |
| Entity | `isDomiciledIn` | Entity | 199 |
| Entity | `includes` | Entity | 196 |
| Entity | `RELATED_TO` | Entity | 186 |
| Entity | `OPERATES_IN` | Entity | 173 |
| Entity | `INVOLVED_IN` | Entity | 163 |
| Entity | `HAS_COMPETITOR` | Entity | 158 |
| Entity | `WORKS_FOR` | Entity | 156 |
| Entity | `HAS_OBLIGATION` | Entity | 147 |
| Entity | `PART_OF` | Entity | 145 |
| Entity | `HAS_SUBSIDIARY` | Entity | 140 |
| Entity | `HAS_SEGMENT` | Entity | 135 |

---

## 3. RDF 분석

### 3.1 기본 통계

| 지표 | 값 |
|------|-----|
| 총 노드 | 15,505 |
| 총 Triple (엣지) | 12,609 |
| Resource 노드 | 12,365 |
| Question 노드 | 3,140 |
| Isolated Resource | 2,662 (21.5%) |

### 3.2 Predicate 분포 (상위 30개)

| Predicate | 개수 | 비율 |
|-----------|------|------|
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
| `fibo-fnd-rel-rel:offers` | 61 | 0.5% |
| `fibo-fnd-agr-ctr:hasPerformanceObligation` | 60 | 0.5% |
| `fibo-fnd-rel-rel:hasCustomer` | 60 | 0.5% |
| `fibo-fnd-rel-rel:facesCompetitionFrom` | 59 | 0.5% |
| `fibo-fnd-rel-rel:hasLocation` | 59 | 0.5% |
| `fibo-fnd-rel-rel:competesOnBasisOf` | 53 | 0.4% |
| `fibo-fnd-rel-rel:hasFactor` | 50 | 0.4% |
| `fibo-fnd-rel-rel:affectedBy` | 47 | 0.4% |
| `fibo-fnd-rel-rel:relatedTo` | 46 | 0.4% |
| `fibo-fnd-rel-rel:hasSubsidiary` | 46 | 0.4% |
| `fibo-fnd-rel-rel:focusesOn` | 45 | 0.4% |
| `fibo-fnd-rel-rel:subjectTo` | 44 | 0.3% |
| `fibo-fnd-agr-ctr:includes` | 42 | 0.3% |
| `fibo-fnd-utl-av:hasObjective` | 41 | 0.3% |
| `fibo-fnd-rel-rel:hasStep` | 41 | 0.3% |

### 3.3 FIBO 온톨로지 Predicate

| FIBO Predicate | 개수 |
|----------------|------|
| `fibo-fnd-rel-rel:competesWith` | 986 |
| `fibo-fnd-rel-rel:includes` | 581 |
| `fibo-fnd-rel-rel:hasCompetitor` | 198 |
| `fibo-fnd-rel-rel:involves` | 180 |
| `fibo-fnd-rel-rel:hasPosition` | 156 |
| `fibo-be-le-lp:isDomiciledIn` | 147 |
| `fibo:includes` | 129 |
| `fibo-fnd-rel-rel:hasEmployee` | 100 |
| `fibo-fnd-rel-rel:hasCharacteristic` | 93 |
| `fibo-fnd-rel-rel:competesOn` | 87 |
| `fibo-fnd-rel-rel:operatesIn` | 86 |
| `fibo-fnd-rel-rel:employs` | 85 |
| `fibo-fnd-rel-rel:provides` | 65 |
| `fibo-fnd-rel-rel:hasPart` | 63 |
| `fibo-fnd-rel-rel:offers` | 61 |
| `fibo-fnd-rel-rel:hasCustomer` | 60 |
| `fibo-fnd-agr-ctr:hasPerformanceObligation` | 60 |
| `fibo-fnd-rel-rel:hasLocation` | 59 |
| `fibo-fnd-rel-rel:facesCompetitionFrom` | 59 |
| `fibo-fnd-rel-rel:competesOnBasisOf` | 53 |

### 3.4 URI 네임스페이스 분석

| Namespace | 개수 |
|-----------|------|
| `ex:` | 12,299 |
| `fibo-fbc-fi-fi:` | 9 |
| `fibo-fnd-agr-ctr:` | 7 |
| `fibo-fnd-acc-4217:` | 6 |
| `fibo:` | 5 |
| `other` | 4 |
| `fibo-fnd-acc-std:` | 4 |
| `fibo-fnd-law-jur:` | 4 |
| `fibo-fnd-pty-pty:` | 4 |
| `fibo-fbc-fi-ip:` | 3 |

### 3.5 Degree 분포

| 지표 | 값 |
|------|-----|
| 최소 | 0 |
| 최대 | 1,901 |
| 평균 | 1.63 |
| 중앙값 | 1.0 |
| P90 | 3.0 |
| P99 | 15.0 |

### 3.6 Top Hub Resources

| 순위 | URI | Degree |
|------|-----|--------|
| 1 | `ex:Company` | 1,901 |
| 2 | `ex:OurCompany` | 526 |
| 3 | `ex:RevenueRecognition` | 412 |
| 4 | `ex:LegalProceedings` | 208 |
| 5 | `fibo-fbc-fi-fi:Share` | 195 |
| 6 | `ex:PerformanceObligation` | 193 |
| 7 | `ex:OurBusiness` | 182 |
| 8 | `ex:Revenue` | 155 |
| 9 | `ex:UnitedStates` | 120 |
| 10 | `ex:VariableConsideration` | 115 |
| 11 | `ex:LegalProceeding` | 110 |
| 12 | `ex:CybersecurityProgram` | 107 |
| 13 | `ex:Competitors` | 97 |
| 14 | `ex:Entity` | 93 |
| 15 | `ex:TransactionPrice` | 90 |

### 3.7 Literal 속성 (데이터 프로퍼티)

| Property | 노드 수 |
|----------|---------|
| `fibo-fnd-rel-rel:hasName` | 2,285 |
| `fibo-fbc-fi-fi:hasPrincipalAmount` | 386 |
| `fibo:hasAmount` | 129 |
| `fibo-fnd-acc-cur:hasAmount` | 111 |
| `fibo-fnd-rel-rel:hasEmployeeCount` | 111 |
| `fibo-fbc-fi-fi:hasMaturityDate` | 88 |
| `fibo-fnd-agr-agr:hasCount` | 63 |
| `fibo-fnd-acc-cur:hasMonetaryAmount` | 56 |
| `fibo-fbc-fi-fi:hasVotingRight` | 54 |
| `fibo-fnd-rel-rel:hasAge` | 49 |
| `fibo-fnd-rel-rel:hasAmount` | 46 |
| `fibo-fnd-rel-rel:hasPercentage` | 44 |
| `fibo-fnd-acc-4217:hasAmount` | 42 |
| `fibo-fnd-rel-rel:hasCount` | 35 |
| `fibo-fnd-acc-acc:hasAmount` | 32 |

---

## 4. LPG vs RDF 비교

### 4.1 구조적 차이

| 특성 | LPG | RDF | 차이 |
|------|-----|-----|------|
| 노드 수 | 17,060 | 15,505 | +1,555 |
| 엣지 수 | 18,892 | 12,609 | +6,283 |
| 밀도 | 0.000065 | 0.000052 | - |
| 평균 Degree | 2.21 | 1.63 | - |
| 최대 Degree | 3,121 | 1,901 | - |

### 4.2 모델링 차이점

```
┌─────────────────────────────────────────────────────────────────┐
│                         LPG vs RDF                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LPG (Labeled Property Graph)          RDF (Triple Store)        │
│  ──────────────────────────           ──────────────────        │
│  • Multi-label nodes (avg 1.79)       • Single label (Resource)  │
│  • Properties on nodes                • Properties as predicates │
│  • Typed relationships                • URI-based predicates     │
│  • question_ids 직접 연결              • question_ids 미연결      │
│                                                                  │
│  장점:                                 장점:                      │
│  ├─ 직관적 쿼리                        ├─ 표준화된 구조            │
│  ├─ 빠른 탐색                          ├─ FIBO 온톨로지 호환       │
│  └─ GNN에 적합                         └─ 추론 지원 가능           │
│                                                                  │
│  GNN 적합성: ★★★★★                    GNN 적합성: ★★★☆☆         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. GNN Feature Engineering

### 5.1 노드 피처 (Node Features)

#### A. 텍스트 임베딩 (Primary Feature)

```python
# 현재 구현
text = f"{label}: {name}"
embedding = sentence_transformer.encode(text)  # 384-dim

# 권장 개선
text = f"{label}: {name}. {description}. {key_properties}"
embedding = mpnet.encode(text)  # 768-dim (더 높은 품질)
```

| 모델 | 차원 | 속도 | 품질 | 권장 |
|------|------|------|------|------|
| all-MiniLM-L6-v2 | 384 | ⭐⭐⭐ | ⭐⭐ | 빠른 실험 |
| all-mpnet-base-v2 | 768 | ⭐⭐ | ⭐⭐⭐ | 프로덕션 |
| e5-large-v2 | 1024 | ⭐ | ⭐⭐⭐⭐ | 최고 품질 |

#### B. 구조적 피처 (Structural Features)

```python
structural_features = [
    # 기본 Centrality
    degree / max_degree,                           # Normalized degree
    in_degree / (in_degree + out_degree + 1e-6),   # In-degree ratio
    out_degree / (in_degree + out_degree + 1e-6),  # Out-degree ratio

    # Graph Centrality (Neo4j GDS로 사전 계산)
    pagerank_score,                                # PageRank
    betweenness_centrality,                        # Bridge 노드 식별

    # Local Structure
    local_clustering_coefficient,                  # 지역 밀도
    avg_neighbor_degree / max_degree,              # 이웃 중요도
]
# Total: 7-dim
```

**Neo4j GDS 계산 예시**:
```cypher
-- PageRank
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score
SET n.pagerank = score;

-- Betweenness
CALL gds.betweenness.stream('myGraph')
YIELD nodeId, score
SET gds.util.asNode(nodeId).betweenness = score;
```

#### C. 라벨 인코딩 (Label Encoding)

```python
# Option 1: Multi-hot encoding
# 2,497개 라벨 → 너무 sparse → 상위 100개만 사용
top_100_labels = [...]  # 빈도순 상위 100개
label_encoding = multi_hot(node_labels, top_100_labels)  # 100-dim

# Option 2: Learnable embedding
class LabelEmbedding(nn.Module):
    def __init__(self, num_labels=100, embed_dim=32):
        self.embedding = nn.Embedding(num_labels, embed_dim)

    def forward(self, label_indices):
        # Multi-label: average embeddings
        return self.embedding(label_indices).mean(dim=0)  # 32-dim
```

#### D. 위치 인코딩 (Positional Encoding)

```python
# Random Walk Positional Encoding (RWPE)
from torch_geometric.transforms import AddRandomWalkPE

transform = AddRandomWalkPE(walk_length=16, attr_name='pe')
data = transform(data)  # 16-dim PE added
```

### 5.2 엣지 피처 (Edge Features)

#### A. 관계 타입 임베딩

```python
# 2,971개 관계 타입 → 클러스터링 필요

# Step 1: 관계 타입 그룹화
relation_groups = {
    "competition": ["COMPETES_WITH", "HAS_COMPETITOR", "facesCompetitionFrom"],
    "composition": ["INCLUDES", "PART_OF", "HAS_SEGMENT"],
    "employment": ["EMPLOYS", "WORKS_FOR", "HAS_EMPLOYEE"],
    "location": ["isDomiciledIn", "OPERATES_IN", "HAS_LOCATION"],
    # ... 약 50개 그룹으로 축소
}

# Step 2: Learnable embedding
class RelationEmbedding(nn.Module):
    def __init__(self, num_relations=50, embed_dim=64):
        self.embedding = nn.Embedding(num_relations, embed_dim)
```

#### B. 엣지 가중치

```python
# Inverse Document Frequency 스타일
edge_weight = log(total_edges / relation_count[rel_type])

# 또는 learnable attention
class EdgeAttention(nn.Module):
    def __init__(self, hidden_dim):
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, src_emb, tgt_emb):
        return torch.sigmoid(self.attn(torch.cat([src_emb, tgt_emb], dim=-1)))
```

### 5.3 통합 피처 구성

```python
@dataclass
class FeatureConfig:
    # Text embedding
    text_dim: int = 384            # or 768 for mpnet

    # Structural features
    structural_dim: int = 7

    # Label encoding
    label_dim: int = 32            # learnable

    # Positional encoding
    pe_dim: int = 16               # RWPE

    # Edge features
    relation_dim: int = 64

    @property
    def node_dim(self) -> int:
        return self.text_dim + self.structural_dim + self.label_dim + self.pe_dim
        # 384 + 7 + 32 + 16 = 439-dim
        # or 768 + 7 + 32 + 16 = 823-dim

# 실제 사용
node_features = torch.cat([
    text_embedding,      # [N, 384]
    structural_feat,     # [N, 7]
    label_embedding,     # [N, 32]
    position_encoding,   # [N, 16]
], dim=-1)              # [N, 439]
```

### 5.4 Hub 노드 처리 전략

```python
# 문제: "The Company" (degree=3,121)가 너무 많은 메시지 집중

# 해결책 1: Degree normalization in GATv2
class DegreeNormalizedGAT(nn.Module):
    def forward(self, x, edge_index):
        # Symmetric normalization
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        # Apply in attention
        ...

# 해결책 2: Virtual node (GraphGPS style)
class VirtualNode(nn.Module):
    def __init__(self, hidden_dim):
        self.virtual_node = nn.Parameter(torch.randn(1, hidden_dim))

    def forward(self, x, batch):
        # Aggregate all nodes to virtual
        # Broadcast back
        ...

# 해결책 3: Hub sampling
def sample_hub_neighbors(edge_index, hub_mask, max_neighbors=50):
    # For hub nodes, randomly sample neighbors
    ...
```

---

## 6. 데이터 품질 이슈

### 6.1 발견된 이슈

| 이슈 | 심각도 | 설명 | 해결 방안 |
|------|--------|------|-----------|
| Hub 불균형 | 🔴 높음 | "The Company" 3,121 edges | Degree normalization |
| Isolated 노드 | 🟡 중간 | LPG 22%, RDF 37% | 필터링 또는 self-loop |
| 관계 타입 과다 | 🟡 중간 | 2,971개 타입 | 클러스터링 (~50개) |
| RDF question 연결 | 🔴 높음 | question_ids 누락 | 데이터 재구축 필요 |
| Generic 노드 | 🟡 중간 | "Our Company" 등 | 특수 처리 필요 |

### 6.2 데이터 클리닝 권장

```python
# 1. Hub 노드 필터링/샘플링
hub_threshold = 500  # degree > 500인 노드 특수 처리

# 2. Isolated 노드 처리
# Option A: 제거
# Option B: Self-loop 추가

# 3. 관계 타입 정규화
def normalize_relation(rel_type: str) -> str:
    rel_lower = rel_type.lower().replace('_', '')
    # Map to canonical form
    mapping = {
        'competeswith': 'COMPETES',
        'hascompetitor': 'COMPETES',
        'includes': 'CONTAINS',
        'partof': 'CONTAINS',
        # ...
    }
    return mapping.get(rel_lower, 'OTHER')

# 4. RDF question_ids 복구
# Parquet에서 다시 매핑 필요
```

### 6.3 권장 전처리 파이프라인

```
Raw Data
    │
    ▼
┌─────────────────┐
│ 1. 데이터 검증   │ → 누락 필드 체크, 타입 검증
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 2. Isolated 처리 │ → 제거 또는 self-loop
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 3. Hub 정규화    │ → degree normalization
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 4. 관계 클러스터 │ → 2,971 → ~50 그룹
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 5. 피처 계산     │ → PageRank, 구조적 피처
└─────────────────┘
    │
    ▼
Clean Data (PyG Data)
```

---

## 부록: 샘플 질문


### 샘플 1
- **ID**: `30eb0cd9`
- **Category**: Shareholder return
- **Type**: None
- **Text**: Cboe’s repurchase costs & the impact of its authorization on cap alloc.

### 샘플 2
- **ID**: `1a4cebce`
- **Category**: Risk
- **Type**: None
- **Text**: Cboe's operational stability, governance in cybersecurity, and financial health.

### 샘플 3
- **ID**: `f8e1242c`
- **Category**: Governance
- **Type**: None
- **Text**: Impact of proactive regulatory engagement on competitive positioning and future growth, CBOE.

### 샘플 4
- **ID**: `6d00752f`
- **Category**: Accounting
- **Type**: None
- **Text**: Impact of fee recognition rev volatility on Cboe Global Markets.

### 샘플 5
- **ID**: `a54fecf0`
- **Category**: Footnotes
- **Type**: None
- **Text**: Cboe (CBOE) allocates capital primarily towards dividends/share buybacks vs growth investments.

---

*이 보고서는 자동 생성되었습니다.*
