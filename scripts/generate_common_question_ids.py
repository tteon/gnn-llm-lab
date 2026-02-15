#!/usr/bin/env python3
"""
LPG ∩ RDF 공통 질문 ID 생성 스크립트.

양쪽 Neo4j DB(finderlpg, finderrdf)에서 충분한 서브그래프가 존재하는
질문만 필터링하여 data/processed/common_question_ids.json 으로 저장한다.

기준: LPG entity+edge >= min_graph_size AND RDF triples >= min_graph_size

Usage:
    # 전체 생성
    uv run python scripts/generate_common_question_ids.py

    # 샘플 5개만 조회 (dry run)
    uv run python scripts/generate_common_question_ids.py --sample 5

    # 최소 서브그래프 크기 변경
    uv run python scripts/generate_common_question_ids.py --min-graph-size 5
"""

import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Neo4jConfig
from src.utils.neo4j_client import Neo4jClient
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "common_question_ids.json"


def get_lpg_question_counts(client: Neo4jClient) -> dict[str, int]:
    """finderlpg에서 질문별 entity+edge 수를 조회."""
    # Entity 수
    entity_results = client.query(
        """
        MATCH (e:Entity)
        WHERE e.question_ids IS NOT NULL
        UNWIND e.question_ids AS qid
        RETURN qid, count(e) AS n_entities
        """,
        database="finderlpg",
    )
    entity_counts = {r["qid"]: r["n_entities"] for r in entity_results}

    # Edge 수
    edge_results = client.query(
        """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.question_ids IS NOT NULL AND b.question_ids IS NOT NULL
        UNWIND a.question_ids AS qid
        WITH qid, a, r, b
        WHERE qid IN b.question_ids
        RETURN qid, count(r) AS n_edges
        """,
        database="finderlpg",
    )
    edge_counts = {r["qid"]: r["n_edges"] for r in edge_results}

    # 합산
    all_qids = set(entity_counts) | set(edge_counts)
    return {
        qid: entity_counts.get(qid, 0) + edge_counts.get(qid, 0)
        for qid in all_qids
    }


def get_rdf_question_counts(client: Neo4jClient) -> dict[str, int]:
    """finderrdf에서 질문별 triple 수를 조회."""
    results = client.query(
        """
        MATCH (a:Resource)-[r]->(b:Resource)
        WHERE r.question_id IS NOT NULL
        WITH r.question_id AS qid, count(r) AS n_triples
        RETURN qid, n_triples AS graph_size
        """,
        database="finderrdf",
    )
    return {r["qid"]: r["graph_size"] for r in results}


def find_common_questions(
    lpg_counts: dict[str, int],
    rdf_counts: dict[str, int],
    min_graph_size: int,
) -> list[str]:
    """양쪽 모두 min_graph_size 이상인 질문 ID를 반환."""
    common = []
    for qid in sorted(set(lpg_counts) & set(rdf_counts)):
        if lpg_counts[qid] >= min_graph_size and rdf_counts[qid] >= min_graph_size:
            common.append(qid)
    return common


def sample_and_show(
    client: Neo4jClient,
    question_ids: list[str],
    n: int,
) -> None:
    """샘플 n개의 질문 상세 정보를 출력."""
    sample_ids = question_ids[:n]
    questions = client.query(
        """
        MATCH (q:Question)
        WHERE q.id IN $qids
        RETURN q.id AS id, q.text AS text, q.category AS category
        """,
        {"qids": sample_ids},
        database="finderlpg",
    )

    q_map = {q["id"]: q for q in questions}

    print(f"\n{'='*80}")
    print(f"  Sample {len(sample_ids)} questions")
    print(f"{'='*80}\n")

    for i, qid in enumerate(sample_ids, 1):
        q = q_map.get(qid, {})
        text = q.get("text", "(not found)")
        category = q.get("category", "?")

        # LPG subgraph size
        lpg = client.query(
            """
            MATCH (e:Entity) WHERE $qid IN e.question_ids
            WITH collect(e) AS entities
            OPTIONAL MATCH (a:Entity)-[r]->(b:Entity)
            WHERE $qid IN a.question_ids AND $qid IN b.question_ids
            RETURN size(entities) AS nodes, count(r) AS edges
            """,
            {"qid": qid},
            database="finderlpg",
        )

        # RDF triple count
        rdf = client.query(
            """
            MATCH (a:Resource)-[r]->(b:Resource)
            WHERE r.question_id = $qid
            RETURN count(r) AS triples
            """,
            {"qid": qid},
            database="finderrdf",
        )

        lpg_nodes = lpg[0]["nodes"] if lpg else 0
        lpg_edges = lpg[0]["edges"] if lpg else 0
        rdf_triples = rdf[0]["triples"] if rdf else 0

        print(f"[{i}] {qid}")
        print(f"    Category: {category}")
        print(f"    Question: {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"    LPG: {lpg_nodes} nodes, {lpg_edges} edges (total {lpg_nodes + lpg_edges})")
        print(f"    RDF: {rdf_triples} triples")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate common question IDs (LPG ∩ RDF)",
    )
    parser.add_argument(
        "--min-graph-size", type=int, default=3,
        help="Minimum entities+edges (LPG) or triples (RDF) per question (default: 3)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Print sample N questions with details (0 = skip)",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help=f"Output JSON path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print results without saving to file",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    # Connect
    neo4j_config = Neo4jConfig()
    client = Neo4jClient(neo4j_config)
    client.connect()

    try:
        # Step 1: LPG question counts
        logger.info("Querying finderlpg for question graph sizes...")
        lpg_counts = get_lpg_question_counts(client)
        logger.info(f"  finderlpg: {len(lpg_counts)} questions with data")

        # Step 2: RDF question counts
        logger.info("Querying finderrdf for question graph sizes...")
        rdf_counts = get_rdf_question_counts(client)
        logger.info(f"  finderrdf: {len(rdf_counts)} questions with data")

        # Step 3: Find common
        common_ids = find_common_questions(lpg_counts, rdf_counts, args.min_graph_size)
        logger.info(
            f"Common questions (both >= {args.min_graph_size}): "
            f"{len(common_ids)} / {len(set(lpg_counts) | set(rdf_counts))} total"
        )

        # Step 4: Sample
        if args.sample > 0:
            sample_and_show(client, common_ids, args.sample)

        # Step 5: Save
        if not args.dry_run:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "question_ids": common_ids,
                "total": len(common_ids),
                "criteria": f"LPG >= {args.min_graph_size} entities+edges AND RDF >= {args.min_graph_size} triples",
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved to {output_path}")
        else:
            logger.info("Dry run — not saving to file")

    finally:
        client.close()


if __name__ == "__main__":
    main()
