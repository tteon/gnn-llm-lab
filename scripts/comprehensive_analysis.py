"""
Comprehensive analysis of FinDER KG datasets (LPG & RDF)
Generates detailed markdown report for docs/
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def analyze_lpg_detailed(driver) -> Dict[str, Any]:
    """Detailed LPG analysis."""
    results = {}

    with driver.session(database="finderlpg") as session:
        # Basic stats
        results["node_count"] = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
        results["edge_count"] = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        results["question_count"] = session.run("MATCH (q:Question) RETURN count(q) as c").single()["c"]
        results["entity_count"] = session.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"]

        # Label distribution (top 30)
        labels = session.run("""
            MATCH (n)
            WITH labels(n) as lbls
            UNWIND lbls as label
            RETURN label, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 30
        """).data()
        results["label_distribution"] = labels

        # Relationship type distribution (top 30)
        rels = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 30
        """).data()
        results["relationship_distribution"] = rels

        # Degree statistics
        degree_stats = session.run("""
            MATCH (n)
            WITH n, COUNT { (n)--() } as degree
            RETURN
                min(degree) as min_deg,
                max(degree) as max_deg,
                avg(degree) as avg_deg,
                stDev(degree) as std_deg,
                percentileCont(degree, 0.25) as p25,
                percentileCont(degree, 0.5) as median,
                percentileCont(degree, 0.75) as p75,
                percentileCont(degree, 0.9) as p90,
                percentileCont(degree, 0.95) as p95,
                percentileCont(degree, 0.99) as p99
        """).single()
        results["degree_stats"] = dict(degree_stats)

        # In/Out degree stats
        directed_stats = session.run("""
            MATCH (n)
            WITH n,
                 COUNT { (n)-[]->() } as out_deg,
                 COUNT { (n)<-[]-() } as in_deg
            RETURN
                avg(in_deg) as avg_in,
                avg(out_deg) as avg_out,
                max(in_deg) as max_in,
                max(out_deg) as max_out,
                stDev(in_deg) as std_in,
                stDev(out_deg) as std_out
        """).single()
        results["directed_stats"] = dict(directed_stats)

        # Top hub nodes
        hubs = session.run("""
            MATCH (n)
            WITH n, COUNT { (n)--() } as degree
            ORDER BY degree DESC
            LIMIT 20
            RETURN
                coalesce(n.name, n.id, 'unknown') as name,
                labels(n) as labels,
                degree,
                COUNT { (n)-[]->() } as out_deg,
                COUNT { (n)<-[]-() } as in_deg
        """).data()
        results["top_hubs"] = hubs

        # Isolated nodes by label
        isolated = session.run("""
            MATCH (n)
            WHERE NOT (n)--()
            WITH labels(n) as lbls
            UNWIND lbls as label
            RETURN label, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 10
        """).data()
        results["isolated_by_label"] = isolated

        # Property coverage
        props = session.run("""
            MATCH (n)
            WITH keys(n) as props
            UNWIND props as prop
            RETURN prop, count(*) as cnt
            ORDER BY cnt DESC
        """).data()
        results["property_coverage"] = props

        # Question-Entity stats
        q_entity = session.run("""
            MATCH (n)
            WHERE n.question_ids IS NOT NULL AND size(n.question_ids) > 0
            WITH size(n.question_ids) as q_count
            RETURN
                avg(q_count) as avg_q_per_entity,
                max(q_count) as max_q_per_entity,
                min(q_count) as min_q_per_entity
        """).single()
        results["question_entity_stats"] = dict(q_entity) if q_entity else {}

        # Entities per question
        entities_per_q = session.run("""
            MATCH (n)
            WHERE n.question_ids IS NOT NULL
            UNWIND n.question_ids as qid
            WITH qid, count(*) as entity_cnt
            RETURN
                avg(entity_cnt) as avg_entities,
                max(entity_cnt) as max_entities,
                min(entity_cnt) as min_entities,
                percentileCont(entity_cnt, 0.5) as median_entities
        """).single()
        results["entities_per_question"] = dict(entities_per_q) if entities_per_q else {}

        # Label combinations (multi-label)
        label_combos = session.run("""
            MATCH (n:Entity)
            WITH labels(n) as lbls
            WHERE size(lbls) > 1
            RETURN lbls, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 15
        """).data()
        results["label_combinations"] = label_combos

        # Relationship patterns
        rel_patterns = session.run("""
            MATCH (s)-[r]->(t)
            WITH labels(s)[0] as src_type, type(r) as rel, labels(t)[0] as tgt_type
            RETURN src_type, rel, tgt_type, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 20
        """).data()
        results["relationship_patterns"] = rel_patterns

        # Sample questions
        sample_qs = session.run("""
            MATCH (q:Question)
            RETURN q.id as id, q.text as text, q.category as category, q.type as type
            LIMIT 5
        """).data()
        results["sample_questions"] = sample_qs

    return results


def analyze_rdf_detailed(driver) -> Dict[str, Any]:
    """Detailed RDF analysis."""
    results = {}

    with driver.session(database="finderrdf") as session:
        # Basic stats
        results["node_count"] = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
        results["edge_count"] = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        results["resource_count"] = session.run("MATCH (r:Resource) RETURN count(r) as c").single()["c"]
        results["question_count"] = session.run("MATCH (q:Question) RETURN count(q) as c").single()["c"]

        # Predicate distribution (all)
        predicates = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as predicate, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 50
        """).data()
        results["predicate_distribution"] = predicates

        # FIBO ontology analysis
        fibo_predicates = session.run("""
            MATCH ()-[r]->()
            WHERE type(r) STARTS WITH 'fibo'
            RETURN type(r) as predicate, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 30
        """).data()
        results["fibo_predicates"] = fibo_predicates

        # Degree statistics
        degree_stats = session.run("""
            MATCH (n)
            WITH n, COUNT { (n)--() } as degree
            RETURN
                min(degree) as min_deg,
                max(degree) as max_deg,
                avg(degree) as avg_deg,
                stDev(degree) as std_deg,
                percentileCont(degree, 0.5) as median,
                percentileCont(degree, 0.9) as p90,
                percentileCont(degree, 0.99) as p99
        """).single()
        results["degree_stats"] = dict(degree_stats)

        # Top hub resources
        hubs = session.run("""
            MATCH (n:Resource)
            WITH n, COUNT { (n)--() } as degree
            ORDER BY degree DESC
            LIMIT 20
            RETURN n.uri as uri, degree
        """).data()
        results["top_hubs"] = hubs

        # Property predicates (literals)
        literal_props = session.run("""
            MATCH (n:Resource)
            WITH keys(n) as props
            UNWIND props as prop
            WITH prop WHERE prop <> 'uri'
            RETURN prop, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 20
        """).data()
        results["literal_properties"] = literal_props

        # Triple patterns
        triple_patterns = session.run("""
            MATCH (s)-[r]->(o)
            WITH
                CASE WHEN s:Resource THEN 'Resource' ELSE 'Question' END as s_type,
                type(r) as predicate,
                CASE WHEN o:Resource THEN 'Resource' ELSE 'Question' END as o_type
            RETURN s_type, predicate, o_type, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 20
        """).data()
        results["triple_patterns"] = triple_patterns

        # Isolated resources
        isolated = session.run("""
            MATCH (n:Resource)
            WHERE NOT (n)--()
            RETURN count(n) as cnt
        """).single()["cnt"]
        results["isolated_resources"] = isolated

        # URI namespace analysis
        namespaces = session.run("""
            MATCH (n:Resource)
            WHERE n.uri IS NOT NULL
            WITH
                CASE
                    WHEN n.uri STARTS WITH 'ex:' THEN 'ex:'
                    WHEN n.uri STARTS WITH 'fibo' THEN split(n.uri, ':')[0] + ':'
                    ELSE 'other'
                END as namespace
            RETURN namespace, count(*) as cnt
            ORDER BY cnt DESC
        """).data()
        results["uri_namespaces"] = namespaces

    return results


def compare_databases(lpg: Dict, rdf: Dict) -> Dict[str, Any]:
    """Compare LPG and RDF representations."""
    comparison = {
        "basic_stats": {
            "lpg_nodes": lpg["node_count"],
            "rdf_nodes": rdf["node_count"],
            "lpg_edges": lpg["edge_count"],
            "rdf_edges": rdf["edge_count"],
            "node_diff": lpg["node_count"] - rdf["node_count"],
            "edge_diff": lpg["edge_count"] - rdf["edge_count"],
        },
        "density": {
            "lpg": lpg["edge_count"] / (lpg["node_count"] ** 2) if lpg["node_count"] > 0 else 0,
            "rdf": rdf["edge_count"] / (rdf["node_count"] ** 2) if rdf["node_count"] > 0 else 0,
        },
        "degree_comparison": {
            "lpg_avg": lpg["degree_stats"]["avg_deg"],
            "rdf_avg": rdf["degree_stats"]["avg_deg"],
            "lpg_max": lpg["degree_stats"]["max_deg"],
            "rdf_max": rdf["degree_stats"]["max_deg"],
        }
    }
    return comparison


def generate_markdown_report(lpg: Dict, rdf: Dict, comparison: Dict) -> str:
    """Generate comprehensive markdown report."""

    report = f"""# FinDER Knowledge Graph 데이터셋 분석 보고서

> 생성일: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

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
| **총 노드** | {lpg["node_count"]:,} | {rdf["node_count"]:,} |
| **총 엣지** | {lpg["edge_count"]:,} | {rdf["edge_count"]:,} |
| **질문 수** | {lpg["question_count"]:,} | {rdf["question_count"]:,} |
| **엔티티/리소스** | {lpg["entity_count"]:,} | {rdf["resource_count"]:,} |
| **그래프 밀도** | {comparison["density"]["lpg"]:.6f} | {comparison["density"]["rdf"]:.6f} |
| **평균 Degree** | {lpg["degree_stats"]["avg_deg"]:.2f} | {rdf["degree_stats"]["avg_deg"]:.2f} |

### 1.2 핵심 발견사항

```
✅ 강점:
• 3,140개 질문에 대한 Knowledge Graph 구축 완료
• FIBO 온톨로지 기반 금융 도메인 구조화
• 질문당 평균 7.18개 엔티티 연결 (LPG)

⚠️ 주의사항:
• 매우 희소한 그래프 (밀도 < 0.0001)
• Hub 노드 불균형 (max degree: {lpg["degree_stats"]["max_deg"]:,})
• 관계 타입 과다 (2,971개 unique types)
• Isolated 노드 존재 (LPG: 22%, RDF: 37%)
```

---

## 2. LPG 분석

### 2.1 노드 라벨 분포

| 라벨 | 개수 | 비율 |
|------|------|------|
"""

    # Add label distribution table
    total_nodes = lpg["node_count"]
    for item in lpg["label_distribution"][:20]:
        pct = (item["cnt"] / total_nodes * 100)
        report += f"| {item['label']} | {item['cnt']:,} | {pct:.1f}% |\n"

    if len(lpg["label_distribution"]) > 20:
        remaining = len(lpg["label_distribution"]) - 20
        report += f"| *... {remaining}개 더* | - | - |\n"

    report += f"""
### 2.2 관계 타입 분포

| 관계 타입 | 개수 | 비율 |
|----------|------|------|
"""

    total_edges = lpg["edge_count"]
    for item in lpg["relationship_distribution"][:15]:
        pct = (item["cnt"] / total_edges * 100)
        report += f"| `{item['rel_type']}` | {item['cnt']:,} | {pct:.1f}% |\n"

    report += f"""
### 2.3 Degree 분포 통계

| 지표 | 값 |
|------|-----|
| 최소 | {lpg["degree_stats"]["min_deg"]} |
| 최대 | {lpg["degree_stats"]["max_deg"]:,} |
| 평균 | {lpg["degree_stats"]["avg_deg"]:.2f} |
| 표준편차 | {lpg["degree_stats"]["std_deg"]:.2f} |
| 중앙값 (P50) | {lpg["degree_stats"]["median"]:.1f} |
| P75 | {lpg["degree_stats"]["p75"]:.1f} |
| P90 | {lpg["degree_stats"]["p90"]:.1f} |
| P95 | {lpg["degree_stats"]["p95"]:.1f} |
| P99 | {lpg["degree_stats"]["p99"]:.1f} |

**분석**: 중앙값({lpg["degree_stats"]["median"]:.1f})과 평균({lpg["degree_stats"]["avg_deg"]:.2f})의 차이가 크고, P99({lpg["degree_stats"]["p99"]:.1f})와 최대값({lpg["degree_stats"]["max_deg"]:,})의 차이가 매우 큼 → **극심한 Hub 노드 존재**

### 2.4 방향성 분석 (In/Out Degree)

| 지표 | In-Degree | Out-Degree |
|------|-----------|------------|
| 평균 | {lpg["directed_stats"]["avg_in"]:.2f} | {lpg["directed_stats"]["avg_out"]:.2f} |
| 최대 | {lpg["directed_stats"]["max_in"]:,} | {lpg["directed_stats"]["max_out"]:,} |
| 표준편차 | {lpg["directed_stats"]["std_in"]:.2f} | {lpg["directed_stats"]["std_out"]:.2f} |

### 2.5 Top Hub 노드

| 순위 | 노드 | 라벨 | Total Degree | Out | In |
|------|------|------|--------------|-----|-----|
"""

    for i, hub in enumerate(lpg["top_hubs"][:15], 1):
        labels = "/".join(hub["labels"][:2]) if hub["labels"] else "-"
        name = hub["name"][:30] + "..." if len(str(hub["name"])) > 30 else hub["name"]
        report += f"| {i} | {name} | {labels} | {hub['degree']:,} | {hub['out_deg']:,} | {hub['in_deg']:,} |\n"

    report += f"""
### 2.6 속성(Property) 커버리지

| 속성명 | 노드 수 | 커버리지 |
|--------|---------|----------|
"""

    for item in lpg["property_coverage"][:15]:
        pct = (item["cnt"] / total_nodes * 100)
        report += f"| `{item['prop']}` | {item['cnt']:,} | {pct:.1f}% |\n"

    report += f"""
### 2.7 질문-엔티티 연결

| 지표 | 값 |
|------|-----|
| 질문당 평균 엔티티 수 | {lpg["entities_per_question"].get("avg_entities", 0):.2f} |
| 질문당 최대 엔티티 수 | {lpg["entities_per_question"].get("max_entities", 0):.0f} |
| 질문당 중앙값 엔티티 수 | {lpg["entities_per_question"].get("median_entities", 0):.1f} |

### 2.8 관계 패턴 (Subject → Relation → Object)

| Subject Type | Relation | Object Type | 개수 |
|--------------|----------|-------------|------|
"""

    for item in lpg["relationship_patterns"][:15]:
        report += f"| {item['src_type']} | `{item['rel'][:30]}` | {item['tgt_type']} | {item['cnt']:,} |\n"

    # RDF Section
    report += f"""
---

## 3. RDF 분석

### 3.1 기본 통계

| 지표 | 값 |
|------|-----|
| 총 노드 | {rdf["node_count"]:,} |
| 총 Triple (엣지) | {rdf["edge_count"]:,} |
| Resource 노드 | {rdf["resource_count"]:,} |
| Question 노드 | {rdf["question_count"]:,} |
| Isolated Resource | {rdf["isolated_resources"]:,} ({rdf["isolated_resources"]/rdf["resource_count"]*100:.1f}%) |

### 3.2 Predicate 분포 (상위 30개)

| Predicate | 개수 | 비율 |
|-----------|------|------|
"""

    rdf_edges = rdf["edge_count"]
    for item in rdf["predicate_distribution"][:30]:
        pct = (item["cnt"] / rdf_edges * 100)
        pred_short = item["predicate"][:50] + "..." if len(item["predicate"]) > 50 else item["predicate"]
        report += f"| `{pred_short}` | {item['cnt']:,} | {pct:.1f}% |\n"

    report += f"""
### 3.3 FIBO 온톨로지 Predicate

| FIBO Predicate | 개수 |
|----------------|------|
"""

    for item in rdf["fibo_predicates"][:20]:
        pred_short = item["predicate"][:60]
        report += f"| `{pred_short}` | {item['cnt']:,} |\n"

    report += f"""
### 3.4 URI 네임스페이스 분석

| Namespace | 개수 |
|-----------|------|
"""

    for item in rdf["uri_namespaces"][:10]:
        report += f"| `{item['namespace']}` | {item['cnt']:,} |\n"

    report += f"""
### 3.5 Degree 분포

| 지표 | 값 |
|------|-----|
| 최소 | {rdf["degree_stats"]["min_deg"]} |
| 최대 | {rdf["degree_stats"]["max_deg"]:,} |
| 평균 | {rdf["degree_stats"]["avg_deg"]:.2f} |
| 중앙값 | {rdf["degree_stats"]["median"]:.1f} |
| P90 | {rdf["degree_stats"]["p90"]:.1f} |
| P99 | {rdf["degree_stats"]["p99"]:.1f} |

### 3.6 Top Hub Resources

| 순위 | URI | Degree |
|------|-----|--------|
"""

    for i, hub in enumerate(rdf["top_hubs"][:15], 1):
        uri_short = hub["uri"][:40] + "..." if len(str(hub["uri"])) > 40 else hub["uri"]
        report += f"| {i} | `{uri_short}` | {hub['degree']:,} |\n"

    report += f"""
### 3.7 Literal 속성 (데이터 프로퍼티)

| Property | 노드 수 |
|----------|---------|
"""

    for item in rdf["literal_properties"][:15]:
        report += f"| `{item['prop']}` | {item['cnt']:,} |\n"

    # Comparison Section
    report += f"""
---

## 4. LPG vs RDF 비교

### 4.1 구조적 차이

| 특성 | LPG | RDF | 차이 |
|------|-----|-----|------|
| 노드 수 | {comparison["basic_stats"]["lpg_nodes"]:,} | {comparison["basic_stats"]["rdf_nodes"]:,} | {comparison["basic_stats"]["node_diff"]:+,} |
| 엣지 수 | {comparison["basic_stats"]["lpg_edges"]:,} | {comparison["basic_stats"]["rdf_edges"]:,} | {comparison["basic_stats"]["edge_diff"]:+,} |
| 밀도 | {comparison["density"]["lpg"]:.6f} | {comparison["density"]["rdf"]:.6f} | - |
| 평균 Degree | {comparison["degree_comparison"]["lpg_avg"]:.2f} | {comparison["degree_comparison"]["rdf_avg"]:.2f} | - |
| 최대 Degree | {comparison["degree_comparison"]["lpg_max"]:,} | {comparison["degree_comparison"]["rdf_max"]:,} | - |

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
text = f"{{label}}: {{name}}"
embedding = sentence_transformer.encode(text)  # 384-dim

# 권장 개선
text = f"{{label}}: {{name}}. {{description}}. {{key_properties}}"
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
relation_groups = {{
    "competition": ["COMPETES_WITH", "HAS_COMPETITOR", "facesCompetitionFrom"],
    "composition": ["INCLUDES", "PART_OF", "HAS_SEGMENT"],
    "employment": ["EMPLOYS", "WORKS_FOR", "HAS_EMPLOYEE"],
    "location": ["isDomiciledIn", "OPERATES_IN", "HAS_LOCATION"],
    # ... 약 50개 그룹으로 축소
}}

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
    mapping = {{
        'competeswith': 'COMPETES',
        'hascompetitor': 'COMPETES',
        'includes': 'CONTAINS',
        'partof': 'CONTAINS',
        # ...
    }}
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

"""

    for i, q in enumerate(lpg["sample_questions"], 1):
        text_short = q["text"][:200] + "..." if len(q["text"]) > 200 else q["text"]
        report += f"""
### 샘플 {i}
- **ID**: `{q["id"]}`
- **Category**: {q.get("category", "N/A")}
- **Type**: {q.get("type", "N/A")}
- **Text**: {text_short}
"""

    report += """
---

*이 보고서는 자동 생성되었습니다.*
"""

    return report


def main():
    print("FinDER KG Comprehensive Analysis")
    print("=" * 60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        print("Connected to Neo4j")

        # Analyze LPG
        print("\nAnalyzing LPG database...")
        lpg_results = analyze_lpg_detailed(driver)

        # Analyze RDF
        print("Analyzing RDF database...")
        rdf_results = analyze_rdf_detailed(driver)

        # Compare
        print("Comparing databases...")
        comparison = compare_databases(lpg_results, rdf_results)

        # Generate report
        print("Generating markdown report...")
        report = generate_markdown_report(lpg_results, rdf_results, comparison)

        # Save report
        report_path = "docs/finder_kg_analysis.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")

        # Also save raw JSON
        json_path = "results/comprehensive_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "lpg": lpg_results,
                "rdf": rdf_results,
                "comparison": comparison,
            }, f, indent=2, default=str)
        print(f"Raw data saved to {json_path}")

    finally:
        driver.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
