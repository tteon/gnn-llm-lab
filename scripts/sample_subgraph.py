#!/usr/bin/env python3
"""각 DB에서 question 1개씩 샘플링하여 서브그래프를 출력."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import Neo4jConfig
from src.utils.neo4j_client import Neo4jClient


def show_lpg_subgraph(client: Neo4jClient):
    """finderlpg: 중간 크기 서브그래프를 가진 질문 1개 샘플링."""
    print("=" * 80)
    print("  [finderlpg] Entity 서브그래프 샘플")
    print("=" * 80)

    # 중간 크기(5~15 entities) 질문 1개 선택
    pick = client.query(
        """
        MATCH (e:Entity)
        WHERE e.question_ids IS NOT NULL
        UNWIND e.question_ids AS qid
        WITH qid, count(e) AS n_entities
        WHERE n_entities >= 5 AND n_entities <= 15
        RETURN qid, n_entities
        ORDER BY n_entities DESC
        LIMIT 1
        """,
        database="finderlpg",
    )
    if not pick:
        print("  (조건에 맞는 질문 없음)")
        return

    qid = pick[0]["qid"]
    n_ent = pick[0]["n_entities"]

    # Question 정보
    q_info = client.query(
        "MATCH (q:Question {id: $qid}) RETURN q.text AS text, q.category AS category",
        {"qid": qid},
        database="finderlpg",
    )
    q_text = q_info[0]["text"] if q_info else "(not found)"
    q_cat = q_info[0]["category"] if q_info else "?"

    print(f"\n  Question ID : {qid}")
    print(f"  Category   : {q_cat}")
    print(f"  Text       : {q_text[:120]}")
    print(f"  Entities   : {n_ent}")

    # Entity 노드 목록
    entities = client.query(
        """
        MATCH (e:Entity)
        WHERE $qid IN e.question_ids
        RETURN e.name AS name, e.type AS type, labels(e) AS labels
        """,
        {"qid": qid},
        database="finderlpg",
    )

    print(f"\n  --- Entity Nodes ({len(entities)}) ---")
    for i, e in enumerate(entities, 1):
        print(f"  {i:2d}. [{e.get('type', '?')}] {e.get('name', '?')}")

    # Entity 간 Edge
    edges = client.query(
        """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE $qid IN a.question_ids AND $qid IN b.question_ids
        RETURN a.name AS src, type(r) AS rel, b.name AS dst
        """,
        {"qid": qid},
        database="finderlpg",
    )

    print(f"\n  --- Edges ({len(edges)}) ---")
    for i, e in enumerate(edges, 1):
        print(f"  {i:2d}. {e['src']} --[{e['rel']}]--> {e['dst']}")

    # Question 노드 edge 확인
    q_edges = client.query(
        """
        MATCH (q:Question {id: $qid})-[r]-()
        RETURN type(r) AS rel, count(r) AS cnt
        """,
        {"qid": qid},
        database="finderlpg",
    )
    print(f"\n  --- Question 노드 직접 연결 edge ---")
    if q_edges:
        for e in q_edges:
            print(f"  {e['rel']}: {e['cnt']}")
    else:
        print("  (없음 — 속성 기반 연결만 존재)")


def show_rdf_subgraph(client: Neo4jClient):
    """finderrdf: 중간 크기 서브그래프를 가진 질문 1개 샘플링."""
    print("\n" + "=" * 80)
    print("  [finderrdf] Resource 트리플 서브그래프 샘플")
    print("=" * 80)

    # 중간 크기(5~15 triples) 질문 1개 선택
    pick = client.query(
        """
        MATCH (a:Resource)-[r]->(b:Resource)
        WHERE r.question_id IS NOT NULL
        WITH r.question_id AS qid, count(r) AS n_triples
        WHERE n_triples >= 5 AND n_triples <= 15
        RETURN qid, n_triples
        ORDER BY n_triples DESC
        LIMIT 1
        """,
        database="finderrdf",
    )
    if not pick:
        print("  (조건에 맞는 질문 없음)")
        return

    qid = pick[0]["qid"]
    n_tri = pick[0]["n_triples"]

    # Question 정보
    q_info = client.query(
        "MATCH (q:Question {id: $qid}) RETURN q.text AS text, q.category AS category",
        {"qid": qid},
        database="finderrdf",
    )
    q_text = q_info[0]["text"] if q_info else "(not found)"
    q_cat = q_info[0]["category"] if q_info else "?"

    print(f"\n  Question ID : {qid}")
    print(f"  Category   : {q_cat}")
    print(f"  Text       : {q_text[:120]}")
    print(f"  Triples    : {n_tri}")

    # 트리플 목록
    triples = client.query(
        """
        MATCH (a:Resource)-[r]->(b:Resource)
        WHERE r.question_id = $qid
        RETURN a.uri AS src, type(r) AS rel, b.uri AS dst
        """,
        {"qid": qid},
        database="finderrdf",
    )

    print(f"\n  --- Triples ({len(triples)}) ---")
    for i, t in enumerate(triples, 1):
        # URI 마지막 부분만 표시
        src = t["src"].split("/")[-1] if t["src"] else "?"
        dst = t["dst"].split("/")[-1] if t["dst"] else "?"
        rel = t["rel"]
        print(f"  {i:2d}. {src} --[{rel}]--> {dst}")

    # Question 노드 edge 확인
    q_edges = client.query(
        """
        MATCH (q:Question {id: $qid})-[r]-()
        RETURN type(r) AS rel, count(r) AS cnt
        """,
        {"qid": qid},
        database="finderrdf",
    )
    print(f"\n  --- Question 노드 직접 연결 edge ---")
    if q_edges:
        for e in q_edges:
            print(f"  {e['rel']}: {e['cnt']}")
    else:
        print("  (없음 — 속성 기반 연결만 존재)")


def main():
    config = Neo4jConfig()
    client = Neo4jClient(config)
    client.connect()

    try:
        show_lpg_subgraph(client)
        show_rdf_subgraph(client)
    finally:
        client.close()

    print("\n" + "=" * 80)
    print("  요약: Question 노드는 양쪽 DB 모두에서 isolated (edge 없음)")
    print("  연결 방식: LPG=Entity.question_ids 배열, RDF=edge.question_id 속성")
    print("=" * 80)


if __name__ == "__main__":
    main()
