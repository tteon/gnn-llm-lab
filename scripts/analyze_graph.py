"""
FinDER KG 데이터셋 분석 스크립트

Neo4j 데이터베이스의 LPG와 RDF 그래프를 분석하여:
1. 기본 통계
2. 그래프 구조 분석
3. GNN에 유용한 피처 추출
"""

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List

from neo4j import GraphDatabase
import numpy as np

# Neo4j 연결 설정
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def analyze_database(driver, db_name: str) -> Dict[str, Any]:
    """Analyze a single database."""
    print(f"\n{'='*60}")
    print(f"Analyzing database: {db_name}")
    print('='*60)

    results = {"database": db_name}

    with driver.session(database=db_name) as session:
        # 1. Basic counts
        print("\n[1] Basic Statistics")
        print("-" * 40)

        node_count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
        edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]

        results["node_count"] = node_count
        results["edge_count"] = edge_count
        print(f"  Nodes: {node_count:,}")
        print(f"  Edges: {edge_count:,}")
        print(f"  Avg degree: {(2 * edge_count / node_count):.2f}" if node_count > 0 else "  N/A")

        # 2. Node labels distribution
        print("\n[2] Node Labels Distribution")
        print("-" * 40)

        labels_result = session.run("""
            MATCH (n)
            WITH labels(n) as lbls
            UNWIND lbls as label
            RETURN label, count(*) as cnt
            ORDER BY cnt DESC
        """)
        labels_dist = {r["label"]: r["cnt"] for r in labels_result}
        results["labels"] = labels_dist

        for label, cnt in list(labels_dist.items())[:15]:
            pct = (cnt / node_count * 100) if node_count > 0 else 0
            print(f"  {label}: {cnt:,} ({pct:.1f}%)")
        if len(labels_dist) > 15:
            print(f"  ... and {len(labels_dist) - 15} more labels")

        # 3. Relationship types distribution
        print("\n[3] Relationship Types Distribution")
        print("-" * 40)

        rels_result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(*) as cnt
            ORDER BY cnt DESC
        """)
        rels_dist = {r["rel_type"]: r["cnt"] for r in rels_result}
        results["relationship_types"] = rels_dist

        for rel_type, cnt in list(rels_dist.items())[:15]:
            pct = (cnt / edge_count * 100) if edge_count > 0 else 0
            print(f"  {rel_type}: {cnt:,} ({pct:.1f}%)")
        if len(rels_dist) > 15:
            print(f"  ... and {len(rels_dist) - 15} more types")

        # 4. Degree distribution (Neo4j 5.x compatible)
        print("\n[4] Degree Distribution")
        print("-" * 40)

        degree_result = session.run("""
            MATCH (n)
            WITH n, COUNT { (n)--() } as degree
            RETURN
                min(degree) as min_deg,
                max(degree) as max_deg,
                avg(degree) as avg_deg,
                percentileCont(degree, 0.5) as median_deg,
                percentileCont(degree, 0.9) as p90_deg,
                percentileCont(degree, 0.99) as p99_deg
        """).single()

        results["degree_stats"] = {
            "min": degree_result["min_deg"],
            "max": degree_result["max_deg"],
            "avg": degree_result["avg_deg"],
            "median": degree_result["median_deg"],
            "p90": degree_result["p90_deg"],
            "p99": degree_result["p99_deg"],
        }

        print(f"  Min: {degree_result['min_deg']}")
        print(f"  Max: {degree_result['max_deg']}")
        print(f"  Avg: {degree_result['avg_deg']:.2f}")
        print(f"  Median: {degree_result['median_deg']:.1f}")
        print(f"  P90: {degree_result['p90_deg']:.1f}")
        print(f"  P99: {degree_result['p99_deg']:.1f}")

        # 5. Top hub nodes
        print("\n[5] Top Hub Nodes (highest degree)")
        print("-" * 40)

        hubs_result = session.run("""
            MATCH (n)
            WITH n, COUNT { (n)--() } as degree
            ORDER BY degree DESC
            LIMIT 10
            RETURN
                coalesce(n.name, n.id, n.uri, 'unknown') as name,
                labels(n) as labels,
                degree
        """)

        hubs = []
        for r in hubs_result:
            hubs.append({
                "name": r["name"],
                "labels": r["labels"],
                "degree": r["degree"]
            })
            label_str = "/".join(r["labels"][:2]) if r["labels"] else "?"
            name_short = r["name"][:40] + "..." if len(str(r["name"])) > 40 else r["name"]
            print(f"  [{label_str}] {name_short}: degree={r['degree']}")

        results["top_hubs"] = hubs

        # 6. Isolated nodes check
        print("\n[6] Graph Connectivity")
        print("-" * 40)

        isolated = session.run("""
            MATCH (n)
            WHERE NOT (n)--()
            RETURN count(n) as cnt
        """).single()["cnt"]

        print(f"  Isolated nodes (no edges): {isolated:,}")
        results["isolated_nodes"] = isolated

        # Connected nodes
        connected = node_count - isolated
        print(f"  Connected nodes: {connected:,} ({connected/node_count*100:.1f}%)")

        # 7. Property analysis
        print("\n[7] Node Properties Analysis")
        print("-" * 40)

        props_result = session.run("""
            MATCH (n)
            WITH keys(n) as props
            UNWIND props as prop
            RETURN prop, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 20
        """)

        props_dist = {r["prop"]: r["cnt"] for r in props_result}
        results["node_properties"] = props_dist

        for prop, cnt in list(props_dist.items())[:10]:
            pct = (cnt / node_count * 100) if node_count > 0 else 0
            print(f"  {prop}: {cnt:,} nodes ({pct:.1f}%)")

        # 8. Question-Entity relationship analysis
        print("\n[8] Question Coverage Analysis")
        print("-" * 40)

        try:
            q_count = session.run("MATCH (q:Question) RETURN count(q) as c").single()["c"]
            print(f"  Total Questions: {q_count:,}")
            results["question_count"] = q_count

            if q_count > 0:
                # Entities with question_ids
                entities_with_q = session.run("""
                    MATCH (n)
                    WHERE n.question_ids IS NOT NULL AND size(n.question_ids) > 0
                    RETURN count(n) as cnt
                """).single()["cnt"]
                print(f"  Entities linked to questions: {entities_with_q:,}")

                # Average entities per question (sample)
                avg_entities = session.run("""
                    MATCH (n)
                    WHERE n.question_ids IS NOT NULL
                    WITH n.question_ids as qids
                    UNWIND qids as qid
                    WITH qid, count(*) as entity_count
                    RETURN avg(entity_count) as avg_e, max(entity_count) as max_e
                """).single()

                if avg_entities:
                    print(f"  Avg entities per question: {avg_entities['avg_e']:.2f}")
                    print(f"  Max entities per question: {avg_entities['max_e']}")
                    results["avg_entities_per_question"] = avg_entities["avg_e"]
        except Exception as e:
            print(f"  Could not analyze questions: {e}")

        # 9. In-degree vs Out-degree analysis
        print("\n[9] Directed Graph Analysis")
        print("-" * 40)

        directed_stats = session.run("""
            MATCH (n)
            WITH n,
                 COUNT { (n)-[]->() } as out_deg,
                 COUNT { (n)<-[]-() } as in_deg
            RETURN
                avg(in_deg) as avg_in,
                avg(out_deg) as avg_out,
                max(in_deg) as max_in,
                max(out_deg) as max_out
        """).single()

        print(f"  Avg in-degree: {directed_stats['avg_in']:.2f}")
        print(f"  Avg out-degree: {directed_stats['avg_out']:.2f}")
        print(f"  Max in-degree: {directed_stats['max_in']}")
        print(f"  Max out-degree: {directed_stats['max_out']}")

        results["directed_stats"] = {
            "avg_in_degree": directed_stats["avg_in"],
            "avg_out_degree": directed_stats["avg_out"],
            "max_in_degree": directed_stats["max_in"],
            "max_out_degree": directed_stats["max_out"],
        }

        # 10. Label co-occurrence (multi-label analysis)
        print("\n[10] Multi-Label Analysis")
        print("-" * 40)

        multi_label = session.run("""
            MATCH (n)
            WITH n, labels(n) as lbls, size(labels(n)) as label_count
            RETURN
                avg(label_count) as avg_labels,
                max(label_count) as max_labels,
                count(CASE WHEN label_count > 1 THEN 1 END) as multi_label_nodes
        """).single()

        print(f"  Avg labels per node: {multi_label['avg_labels']:.2f}")
        print(f"  Max labels per node: {multi_label['max_labels']}")
        print(f"  Multi-label nodes: {multi_label['multi_label_nodes']:,}")

        results["multi_label_stats"] = {
            "avg_labels": multi_label["avg_labels"],
            "max_labels": multi_label["max_labels"],
            "multi_label_nodes": multi_label["multi_label_nodes"],
        }

        # 11. Sample subgraph for a question
        print("\n[11] Sample Question Subgraph")
        print("-" * 40)

        sample_q = session.run("""
            MATCH (q:Question)
            RETURN q.id as qid, q.text as text
            LIMIT 1
        """).single()

        if sample_q:
            print(f"  Question ID: {sample_q['qid']}")
            print(f"  Text: {sample_q['text'][:80]}...")

            subgraph = session.run("""
                MATCH (e:Entity)
                WHERE $qid IN e.question_ids
                WITH collect(e) as seeds
                UNWIND seeds as seed
                OPTIONAL MATCH (seed)-[r]-(neighbor:Entity)
                WHERE neighbor IN seeds OR $qid IN neighbor.question_ids
                RETURN
                    count(DISTINCT seed) + count(DISTINCT neighbor) as node_count,
                    count(DISTINCT r) as edge_count
            """, qid=sample_q['qid']).single()

            if subgraph:
                print(f"  Subgraph nodes: {subgraph['node_count']}")
                print(f"  Subgraph edges: {subgraph['edge_count']}")

    return results


def analyze_rdf_specific(driver, db_name: str = "finderrdf") -> Dict[str, Any]:
    """RDF-specific analysis."""
    print("\n" + "="*60)
    print(f"RDF-Specific Analysis: {db_name}")
    print("="*60)

    results = {}

    with driver.session(database=db_name) as session:
        # Predicate distribution
        print("\n[RDF-1] Predicate Distribution")
        print("-" * 40)

        preds = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as pred, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 20
        """)

        pred_dist = {}
        for r in preds:
            pred_dist[r["pred"]] = r["cnt"]
            print(f"  {r['pred']}: {r['cnt']:,}")

        results["predicates"] = pred_dist

        # Subject/Object type analysis
        print("\n[RDF-2] Triple Pattern Analysis")
        print("-" * 40)

        patterns = session.run("""
            MATCH (s)-[r]->(o)
            WITH labels(s)[0] as s_type, type(r) as pred, labels(o)[0] as o_type
            RETURN s_type, pred, o_type, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 15
        """)

        print("  Top triple patterns (subject_type, predicate, object_type):")
        for r in patterns:
            print(f"    ({r['s_type']}, {r['pred']}, {r['o_type']}): {r['cnt']:,}")

    return results


def suggest_gnn_features(lpg_results: Dict, rdf_results: Dict) -> None:
    """Suggest GNN features based on analysis."""

    print("\n" + "="*60)
    print("GNN FEATURE RECOMMENDATIONS")
    print("="*60)

    # Calculate some derived statistics
    node_count = lpg_results.get("node_count", 0)
    edge_count = lpg_results.get("edge_count", 0)
    density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0
    num_labels = len(lpg_results.get("labels", {}))
    num_rel_types = len(lpg_results.get("relationship_types", {}))

    print(f"""
=== Dataset Summary ===
  Nodes: {node_count:,}
  Edges: {edge_count:,}
  Density: {density:.6f} (SPARSE)
  Node Labels: {num_labels:,}
  Relation Types: {num_rel_types:,}
  Avg Degree: {lpg_results.get('degree_stats', {}).get('avg', 0):.2f}
  Max Degree: {lpg_results.get('degree_stats', {}).get('max', 0)}
""")

    print("""
[A] NODE FEATURES (for LPG - GAT/GNN)
=====================================

1. TEXT EMBEDDINGS (Primary - 현재 사용)
   ├─ Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
   ├─ Input: f"{{label}}: {{name}}. {{properties}}"
   └─ 권장: all-mpnet-base-v2 (768-dim) for better quality

2. STRUCTURAL FEATURES (추가 권장) ⭐
   ├─ Degree centrality: normalize by max_degree
   ├─ In/Out degree ratio: for directed importance
   ├─ Local clustering coefficient
   ├─ PageRank score (precompute with Neo4j GDS)
   └─ Betweenness centrality (for bridge nodes)

   구현 예시:
   ```python
   structural_features = torch.tensor([
       degree / max_degree,           # normalized degree
       in_degree / (in_degree + out_degree + 1),  # in-ratio
       pagerank_score,                # from Neo4j GDS
       clustering_coeff,              # local clustering
   ])
   node_features = torch.cat([text_embedding, structural_features], dim=-1)
   ```

3. LABEL ENCODING (Multi-label 지원 필요)
   ├─ Multi-hot encoding: {num_labels} labels detected
   ├─ Label embedding: learnable embedding per label
   └─ Hierarchy-aware: if label hierarchy exists

4. POSITIONAL ENCODING
   ├─ Random Walk PE (RWPE)
   ├─ Laplacian PE (eigenvectors)
   └─ Distance encoding (to query entities)
""")

    print(f"""
[B] EDGE FEATURES (for Message Passing)
=======================================

1. RELATION TYPE EMBEDDING
   ├─ {num_rel_types:,} unique relation types
   ├─ One-hot: sparse, {num_rel_types}-dim
   ├─ Learnable: 32-64 dim embedding
   └─ Top types to focus on:""")

    for rel, cnt in list(lpg_results.get("relationship_types", {}).items())[:5]:
        print(f"      - {rel}: {cnt:,}")

    print("""
2. EDGE DIRECTION
   ├─ Forward/backward encoding
   └─ Bidirectional aggregation

3. EDGE WEIGHTS
   ├─ Inverse frequency (rare relations more important)
   ├─ Confidence scores (if available)
   └─ Co-occurrence based
""")

    print("""
[C] KNOWLEDGE GRAPH EMBEDDINGS (for RDF)
========================================

1. TransE (현재)
   ├─ h + r ≈ t
   ├─ 빠르고 단순
   └─ 1-to-1 관계에 적합

2. RotatE (권장 업그레이드) ⭐
   ├─ h ∘ r ≈ t (rotation in complex space)
   ├─ 대칭/반대칭/역관계 모델링
   └─ 구현: torch_geometric.nn.RotatE

3. CompGCN (최고 성능) ⭐⭐
   ├─ GNN + KGE 통합 모델
   ├─ Message passing with relation-aware aggregation
   └─ 구현: torch_geometric.nn.CompGCN

   ```python
   from torch_geometric.nn import CompGCN

   class RDFEncoder(nn.Module):
       def __init__(self, num_entities, num_relations, hidden_dim):
           self.compgcn = CompGCN(
               in_channels=hidden_dim,
               out_channels=hidden_dim,
               num_relations=num_relations,
           )
   ```
""")

    print("""
[D] G-RETRIEVER SPECIFIC RECOMMENDATIONS
========================================

1. PCST SUBGRAPH RETRIEVAL
   ├─ Graph is SPARSE (density < 0.001) → PCST works well
   ├─ Avg degree ~2.2 → 2-hop neighborhood sufficient
   ├─ Top-k: 3-5 seed nodes from question
   └─ Edge cost: 0.5-1.0 (tune based on subgraph size)

2. CONTEXT WINDOW MANAGEMENT
   ├─ Max nodes: 30-50 (based on max_degree analysis)
   ├─ Max edges: 50-100
   └─ Prioritize by: degree centrality + question relevance

3. GRAPH TRANSFORMER CONFIG (G-Retriever original)
   ├─ Layers: 4
   ├─ Heads: 4-8
   ├─ Hidden: 1024
   └─ Pooling: Mean (or attention-based)

4. TRAINING TIPS
   ├─ Freeze LLM, train GNN + Projector
   ├─ Use soft prompting (8-10 virtual tokens)
   └─ LoRA for LLM fine-tuning (r=8, alpha=16)
""")

    print("""
[E] RECOMMENDED FEATURE COMBINATION
===================================

Option 1: Simple (Fast Experiments)
───────────────────────────────────
  Node: Text embedding (384-dim)
  Edge: One-hot relation type
  GNN: GAT 2-layer

Option 2: Balanced (Production) ⭐ RECOMMENDED
──────────────────────────────────────────────
  Node: Text embedding (384) + Structural (4) = 388-dim
  Edge: Learnable relation embedding (32-dim)
  GNN: GATv2 4-layer with edge features

Option 3: Advanced (Best Quality)
─────────────────────────────────
  Node: Text (768) + Structural (8) + Positional (16) = 792-dim
  Edge: CompGCN-style relation composition
  GNN: GraphTransformer 4-layer
""")


def main():
    print("FinDER Knowledge Graph Analysis")
    print("=" * 60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        print("Connected to Neo4j")

        # Check available databases
        with driver.session(database="system") as session:
            dbs = session.run("SHOW DATABASES").data()
            db_names = [d["name"] for d in dbs if d["name"] not in ["system", "neo4j"]]
            print(f"Available databases: {db_names}")

        # Analyze each database
        results = {}

        if "finderlpg" in db_names:
            results["lpg"] = analyze_database(driver, "finderlpg")

        if "finderrdf" in db_names:
            results["rdf"] = analyze_database(driver, "finderrdf")
            analyze_rdf_specific(driver, "finderrdf")

        # GNN Feature suggestions
        if results:
            suggest_gnn_features(
                results.get("lpg", {}),
                results.get("rdf", {})
            )

        # Save results
        with open("results/graph_analysis.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n{'='*60}")
        print("Analysis complete! Results saved to results/graph_analysis.json")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
