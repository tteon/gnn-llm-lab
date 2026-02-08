"""
LPG vs RDF Context → LLM Attention 비교 실험.

5가지 context format 조건에서 attention 분포 차이를 정량적으로 측정한다.

Conditions:
    C1: lpg_structured  - LPG data → structured format
    C2: lpg_natural     - LPG data → natural language format
    C3: rdf_raw         - RDF triples with FIBO URIs
    C4: rdf_cleaned     - RDF triples with cleaned URIs
    C5: no_context      - Baseline (question only)

Usage (Colab A100):
    python src/attention_experiment.py --model llama8b --samples 100

    # Quick test (5 samples, 2 conditions):
    python src/attention_experiment.py --model llama8b --samples 5 \
        --conditions lpg_structured rdf_raw
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils import (
    AttentionConfig,
    AttentionExtractor,
    AttentionResult,
    Evaluator,
    ExperimentTracker,
    GraphFormatter,
    Neo4jClient,
    Neo4jConfig,
    ModelConfig,
    get_logger,
    set_seed,
    setup_logging,
)
from src.utils.attention_analysis import AttentionAnalyzer

logger = get_logger("attention_experiment")

ALL_CONDITIONS = [
    "lpg_structured",
    "lpg_natural",
    "rdf_raw",
    "rdf_cleaned",
    "no_context",
]


@dataclass
class ExperimentArgs:
    """Experiment configuration."""

    model_alias: str = "llama8b"
    quant: Optional[str] = None
    samples: int = 100
    conditions: List[str] = field(default_factory=lambda: list(ALL_CONDITIONS))
    seed: int = 42
    max_new_tokens: int = 256
    output_dir: str = "results/attention_experiment"
    checkpoint_every: int = 10
    # Attention extraction
    layers_to_extract: List[int] = field(default_factory=lambda: [-1, -2, -3, -4, -5])
    top_k_tokens: int = 30
    entity_coverage_k: int = 20
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


# ──────────────────────────────────────────────────────────
#  Data Fetching
# ──────────────────────────────────────────────────────────


def load_common_question_ids(path: str = "data/processed/common_question_ids.json") -> List[str]:
    """Load pre-computed common question IDs."""
    with open(path) as f:
        data = json.load(f)
    return data["question_ids"]


def sample_questions(
    client: Neo4jClient,
    question_ids: List[str],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample n questions with stratified category sampling."""
    rng = np.random.default_rng(seed)

    # Fetch question metadata
    questions = client.query(
        """
        MATCH (q:Question)
        WHERE q.id IN $qids
        RETURN q.id AS id, q.text AS text, q.answer AS answer,
               q.category AS category
        """,
        {"qids": question_ids},
        database="finderlpg",
    )

    if len(questions) <= n:
        selected = questions
    else:
        # Stratified by category
        by_cat: Dict[str, list] = {}
        for q in questions:
            cat = q.get("category", "unknown")
            by_cat.setdefault(cat, []).append(q)

        selected = []
        remaining = n
        cats = sorted(by_cat.keys())
        for cat in cats:
            pool = by_cat[cat]
            cat_quota = max(1, int(n * len(pool) / len(questions)))
            cat_quota = min(cat_quota, len(pool), remaining)
            indices = rng.choice(len(pool), size=cat_quota, replace=False)
            selected.extend(pool[i] for i in indices)
            remaining -= cat_quota
            if remaining <= 0:
                break

        # Fill any remainder
        if remaining > 0:
            used_ids = {q["id"] for q in selected}
            extras = [q for q in questions if q["id"] not in used_ids]
            rng.shuffle(extras)
            selected.extend(extras[:remaining])

    rng.shuffle(selected)
    logger.info(f"Sampled {len(selected)} questions from {len(questions)} candidates")
    return list(selected)


def fetch_lpg_subgraph(
    client: Neo4jClient,
    question_id: str,
) -> Dict[str, Any]:
    """Fetch LPG nodes and edges for a question from finderlpg."""
    nodes = client.query(
        """
        MATCH (e:Entity)
        WHERE $qid IN e.question_ids
        RETURN e.id AS id, e.label AS label, e.name AS name
        """,
        {"qid": question_id},
        database="finderlpg",
    )

    edges = client.query(
        """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE $qid IN a.question_ids AND $qid IN b.question_ids
        WITH DISTINCT a.name AS source, type(r) AS type, b.name AS target
        RETURN source, type, target
        """,
        {"qid": question_id},
        database="finderlpg",
    )

    return {"nodes": nodes, "edges": edges}


def fetch_rdf_subgraph(
    client: Neo4jClient,
    question_id: str,
) -> Dict[str, Any]:
    """Fetch RDF triples for a question from finderrdf."""
    edges = client.query(
        """
        MATCH (a:Resource)-[r]->(b:Resource)
        WHERE r.question_id = $qid
        RETURN a.uri AS source, type(r) AS type, b.uri AS target
        """,
        {"qid": question_id},
        database="finderrdf",
    )

    return {"edges": edges}


def extract_entity_names(lpg_data: Dict[str, Any]) -> List[str]:
    """Extract entity names from LPG subgraph for attention aggregation."""
    names = []
    for node in lpg_data.get("nodes", []):
        name = node.get("name")
        if name:
            names.append(name)
    return names


# ──────────────────────────────────────────────────────────
#  Context Building
# ──────────────────────────────────────────────────────────


def build_context(
    condition: str,
    lpg_data: Dict[str, Any],
    rdf_data: Dict[str, Any],
) -> Optional[str]:
    """Build context text for a given condition."""
    if condition == "lpg_structured":
        return GraphFormatter.format(
            lpg_data["nodes"], lpg_data["edges"], style="structured",
        )
    elif condition == "lpg_natural":
        return GraphFormatter.format(
            lpg_data["nodes"], lpg_data["edges"], style="natural",
        )
    elif condition == "rdf_raw":
        return GraphFormatter.format(
            [], rdf_data["edges"], style="triple",
        )
    elif condition == "rdf_cleaned":
        return GraphFormatter.format_rdf_cleaned(rdf_data["edges"])
    elif condition == "no_context":
        return None
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ──────────────────────────────────────────────────────────
#  Main Experiment
# ──────────────────────────────────────────────────────────


def run_experiment(args: ExperimentArgs) -> None:
    """Run the full attention comparison experiment."""
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    setup_logging(level="INFO", log_dir=str(output_dir))
    logger.info(f"Starting attention experiment: {args.conditions}")
    logger.info(f"Output: {output_dir}")

    # ── Neo4j setup ──
    neo4j_config = Neo4jConfig(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )
    client = Neo4jClient(neo4j_config)
    client.connect()

    # ── Load & sample questions ──
    common_ids = load_common_question_ids()
    questions = sample_questions(client, common_ids, args.samples, args.seed)
    logger.info(f"Questions sampled: {len(questions)}")

    # ── LLM setup ──
    from src.utils.local_llm import LocalLLMManager

    model_config = ModelConfig()
    llm = LocalLLMManager(model_config)
    attn_config = AttentionConfig(
        enabled=True,
        layers_to_extract=args.layers_to_extract,
        aggregate_heads="none",  # Per-head extraction
        save_context_attention_only=False,
        top_k_tokens=args.top_k_tokens,
        output_dir=str(output_dir / "attention_data"),
    )
    llm.load_model(args.model_alias, quant_override=args.quant, attention_config=attn_config)

    # ── Analysis tools ──
    analyzer = AttentionAnalyzer()
    evaluator = Evaluator()

    # ── Run experiment ──
    all_results: List[Dict[str, Any]] = []
    total_runs = len(questions) * len(args.conditions)
    run_idx = 0

    for q_idx, question in enumerate(questions):
        qid = question["id"]

        # Fetch subgraphs once per question
        lpg_data = fetch_lpg_subgraph(client, qid)
        rdf_data = fetch_rdf_subgraph(client, qid)
        entity_names = extract_entity_names(lpg_data)

        for condition in args.conditions:
            run_idx += 1
            context = build_context(condition, lpg_data, rdf_data)

            # Skip conditions with no data
            if condition.startswith("lpg") and not lpg_data["edges"]:
                logger.debug(f"[{qid}] Skipping {condition}: no LPG edges")
                continue
            if condition.startswith("rdf") and not rdf_data["edges"]:
                logger.debug(f"[{qid}] Skipping {condition}: no RDF edges")
                continue

            try:
                # Generate with attention extraction
                extract_attn = condition != "no_context"
                response = llm.generate(
                    question=question["text"],
                    context=context,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    extract_attention=extract_attn,
                    entity_names=entity_names if extract_attn else None,
                )

                # Compute attention metrics
                attn_metrics: Dict[str, float] = {}
                if response.attention_data and len(response.attention_data.context_attention_scores) > 0:
                    attn = response.attention_data

                    # Layer-averaged scores already in attn.context_attention_scores
                    # Per-head from first extracted layer for head analysis
                    first_layer = args.layers_to_extract[0] % 32
                    per_head_for_analysis = None
                    if attn.per_head_attention and first_layer in attn.per_head_attention:
                        per_head_for_analysis = attn.per_head_attention[first_layer]

                    attn_metrics = analyzer.compute_all_metrics(
                        token_scores=attn.context_attention_scores,
                        token_strings=attn.context_tokens,
                        ground_truth_entities=entity_names,
                        per_head_scores=per_head_for_analysis,
                        k=args.entity_coverage_k,
                    )

                    # Save raw attention data
                    attn_path = output_dir / "attention_data" / f"{qid}_{condition}.npz"
                    AttentionExtractor.save_attention(attn, str(attn_path))

                # Compute answer quality
                eval_result = evaluator.evaluate_single(
                    response.text, question["answer"],
                )

                result = {
                    "question_id": qid,
                    "category": question.get("category", "unknown"),
                    "condition": condition,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "generation_time": response.generation_time,
                    # Attention metrics
                    **{f"attn_{k}": v for k, v in attn_metrics.items()},
                    # Quality metrics
                    "exact_match": eval_result.exact_match,
                    "substring_match": eval_result.substring_match,
                    "token_f1": eval_result.token_f1,
                    "rouge_l": eval_result.rouge_l,
                }
                all_results.append(result)

                if run_idx % 10 == 0:
                    logger.info(
                        f"[{run_idx}/{total_runs}] {qid} | {condition} | "
                        f"entropy={attn_metrics.get('entropy', 'N/A'):.3f} | "
                        f"F1={eval_result.token_f1:.3f}"
                        if attn_metrics else
                        f"[{run_idx}/{total_runs}] {qid} | {condition} | "
                        f"F1={eval_result.token_f1:.3f}"
                    )

            except Exception as e:
                logger.error(f"[{qid}] {condition} failed: {e}")
                all_results.append({
                    "question_id": qid,
                    "condition": condition,
                    "error": str(e),
                })

        # Checkpoint
        if (q_idx + 1) % args.checkpoint_every == 0:
            _save_checkpoint(all_results, output_dir, q_idx + 1)

    # ── Save final results ──
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary CSV
    _save_summary_csv(all_results, output_dir)

    logger.info(f"Experiment complete. {len(all_results)} results saved to {output_dir}")

    # Cleanup
    llm.unload_model()
    client.close()


def _save_checkpoint(results: List[Dict], output_dir: Path, q_count: int) -> None:
    """Save intermediate checkpoint."""
    ckpt_path = output_dir / f"checkpoint_q{q_count}.json"
    with open(ckpt_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Checkpoint saved: {ckpt_path} ({len(results)} records)")


def _save_summary_csv(results: List[Dict], output_dir: Path) -> None:
    """Save per-condition summary statistics as CSV."""
    import csv
    from collections import defaultdict

    by_condition: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_condition[r["condition"]].append(r)

    summary_path = output_dir / "summary.csv"
    metric_keys = [
        "attn_entropy", "attn_prefix_waste_ratio", "attn_semantic_density",
        "attn_entity_coverage_at_k", "attn_num_context_tokens",
        "attn_head_entropy_mean", "attn_head_entropy_std",
        "token_f1", "exact_match", "rouge_l",
        "input_tokens", "generation_time",
    ]

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["condition", "n"] + [f"{k}_mean" for k in metric_keys] + [f"{k}_std" for k in metric_keys]
        writer.writerow(header)

        for cond in ALL_CONDITIONS:
            records = by_condition.get(cond, [])
            if not records:
                continue
            row = [cond, len(records)]
            for k in metric_keys:
                vals = [r.get(k, 0.0) for r in records if r.get(k) is not None]
                row.append(f"{np.mean(vals):.4f}" if vals else "")
            for k in metric_keys:
                vals = [r.get(k, 0.0) for r in records if r.get(k) is not None]
                row.append(f"{np.std(vals):.4f}" if vals else "")
            writer.writerow(row)

    logger.info(f"Summary saved: {summary_path}")


# ──────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(
        description="LPG vs RDF Attention Comparison Experiment",
    )
    parser.add_argument("--model", default="llama8b", help="Model alias from MODEL_REGISTRY")
    parser.add_argument("--quant", default=None, help="Quantization override (4bit/8bit/None)")
    parser.add_argument("--samples", type=int, default=100, help="Number of questions")
    parser.add_argument(
        "--conditions", nargs="+", default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS, help="Context conditions to test",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="results/attention_experiment")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-password", default="password")

    parsed = parser.parse_args()
    return ExperimentArgs(
        model_alias=parsed.model,
        quant=parsed.quant,
        samples=parsed.samples,
        conditions=parsed.conditions,
        seed=parsed.seed,
        max_new_tokens=parsed.max_new_tokens,
        output_dir=parsed.output_dir,
        checkpoint_every=parsed.checkpoint_every,
        neo4j_uri=parsed.neo4j_uri,
        neo4j_password=parsed.neo4j_password,
    )


if __name__ == "__main__":
    experiment_args = parse_args()
    run_experiment(experiment_args)
