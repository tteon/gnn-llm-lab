"""
KV Cache Offloading + Attention Distribution Experiment.

5 conditions (lpg, rdf, text, lpg_rdf, vanilla) × vLLM API with LMCache.
Measures: latency, prefix cache hit rate, KV cache metrics per condition.

The API model (openai/gpt-oss-120b) is a reasoning model that returns
`reasoning_content` (chain-of-thought) + `content` (final answer).
`content` may be null if `max_tokens` is insufficient.

Usage:
    # Test metrics logging
    uv run python src/kvcache_experiment.py --test-metrics

    # Run full experiment
    uv run python src/kvcache_experiment.py --sample-size 50

    # Specific conditions
    uv run python src/kvcache_experiment.py --conditions lpg rdf vanilla --sample-size 20

    # Resume from checkpoint
    uv run python src/kvcache_experiment.py --resume results/kvcache_experiment/20260208_143000/
"""

import argparse
import ast
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from src.utils import setup_logging, get_logger
from src.utils.config import Neo4jConfig
from src.utils.neo4j_client import Neo4jClient
from src.utils.formatting import GraphFormatter
from src.utils.evaluation import Evaluator, EvaluationResult
from src.utils.reproducibility import set_seed

logger = get_logger(__name__)

# ── Constants ──

API_URL = "http://89.169.103.68:30080"
API_KEY = "vllm_sk_e62386e7fbde521b6f9af6048765af9f100ddb1613a1a4cf16d7775a"
MODEL = "openai/gpt-oss-120b"
METRICS_URL = f"{API_URL}/metrics"

# The user's specific pod in the cluster
TARGET_SERVER = "http://192.168.0.53:8000"

ALL_CONDITIONS = ["lpg", "rdf", "text", "lpg_rdf", "vanilla"]

DEFAULT_PARQUET_PATH = "data/raw/FinDER_KG_Merged.parquet"
DEFAULT_COMMON_IDS_PATH = "data/processed/common_question_ids.json"


# ── Experiment Args ──


@dataclass
class KVCacheExperimentArgs:
    """Experiment configuration."""

    conditions: List[str] = field(default_factory=lambda: list(ALL_CONDITIONS))
    sample_size: int = 50
    seed: int = 42
    max_tokens: int = 512
    lmcache_wait_sec: float = 3.0
    inter_question_wait: float = 1.0
    output_dir: str = "results/kvcache_experiment"
    checkpoint_every: int = 10
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    # Data paths
    parquet_path: str = DEFAULT_PARQUET_PATH
    common_ids_path: str = DEFAULT_COMMON_IDS_PATH
    # Resume
    resume_dir: Optional[str] = None


# ── Metrics Collector ──


class VLLMMetricsCollector:
    """Scrape and parse Prometheus metrics from vLLM /metrics endpoint."""

    def __init__(
        self,
        metrics_url: str = METRICS_URL,
        target_server: Optional[str] = TARGET_SERVER,
    ):
        self.metrics_url = metrics_url
        self.target_server = target_server

    def scrape_raw(self) -> str:
        """Fetch raw Prometheus text from /metrics."""
        resp = requests.get(self.metrics_url, timeout=10)
        resp.raise_for_status()
        return resp.text

    def parse_metrics(self, raw: str) -> Dict[str, float]:
        """Parse Prometheus text format into {metric_name: value} dict.

        Collects ALL vllm:* metrics (no filter). If target_server is set,
        only returns metrics for that server.
        """
        result = {}
        for line in raw.splitlines():
            if line.startswith("#") or not line.strip():
                continue

            # Parse: metric_name{labels} value
            match = re.match(
                r'^([\w:]+)(?:\{([^}]*)\})?\s+([\d.eE+\-]+|NaN|Inf|-Inf)$',
                line,
            )
            if not match:
                continue

            name, labels_str, value_str = match.groups()

            # Filter by target server if specified
            if self.target_server and labels_str:
                if f'server="{self.target_server}"' not in labels_str:
                    continue

            try:
                value = float(value_str)
            except ValueError:
                value = float("nan")

            result[name] = value

        return result

    def snapshot(self) -> Dict[str, Any]:
        """Take a timestamped metrics snapshot. Returns both raw text and parsed dict."""
        raw = self.scrape_raw()
        metrics = self.parse_metrics(raw)
        return {
            "timestamp": datetime.now().isoformat(),
            "epoch": time.time(),
            "server": self.target_server,
            "metrics": metrics,
            "raw": raw,
        }

    def delta(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute metric deltas between two snapshots."""
        deltas = {}
        for key in after["metrics"]:
            if key in before["metrics"]:
                deltas[f"delta_{key}"] = (
                    after["metrics"][key] - before["metrics"][key]
                )
        deltas["elapsed_sec"] = after["epoch"] - before["epoch"]
        return deltas


# ── API Client ──


class VLLMClient:
    """OpenAI-compatible chat completion client for vLLM.

    Handles reasoning model responses where `content` may be null
    and `reasoning_content` contains the chain-of-thought.
    """

    MAX_RETRIES = 5
    RETRY_BACKOFF_BASE = 2.0

    def __init__(
        self,
        api_url: str = API_URL,
        api_key: str = API_KEY,
        model: str = MODEL,
    ):
        self.api_url = f"{api_url}/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.model = model

    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Send a chat completion request and return response + timing.

        Returns:
            Dict with keys: response_text, reasoning_text, finish_reason,
            latency_sec, prompt_tokens, completion_tokens, total_tokens,
            model, request_id, prompt_content
        """
        if context:
            content = (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        else:
            content = f"Question: {question}\n\nAnswer:"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                start = time.time()
                resp = requests.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers,
                    timeout=300,
                )
                latency = time.time() - start

                if resp.status_code == 429:
                    wait = self.RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        f"Rate limited (429), retrying in {wait:.1f}s "
                        f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                choice = data["choices"][0]
                message = choice["message"]
                usage = data.get("usage", {})

                # Reasoning model: content may be null
                response_text = message.get("content") or ""
                reasoning_text = message.get("reasoning_content") or ""

                return {
                    "response_text": response_text,
                    "reasoning_text": reasoning_text,
                    "finish_reason": choice.get("finish_reason"),
                    "latency_sec": latency,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "model": data.get("model", self.model),
                    "request_id": data.get("id", ""),
                    "prompt_content": content,
                }

            except requests.exceptions.HTTPError:
                raise
            except Exception as e:
                last_exc = e
                wait = self.RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    f"Request failed: {e}, retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                time.sleep(wait)

        raise last_exc or RuntimeError("All retries exhausted")


# ── Context Builders ──


def build_context(
    condition: str,
    lpg_data: Optional[Dict[str, Any]] = None,
    rdf_data: Optional[Dict[str, Any]] = None,
    text_references: Optional[str] = None,
) -> Optional[str]:
    """Build context text for a given condition."""
    if condition == "vanilla":
        return None

    if condition == "lpg":
        if not lpg_data or not lpg_data.get("edges"):
            return None
        return GraphFormatter.format(
            lpg_data["nodes"], lpg_data["edges"], style="structured",
        )

    if condition == "rdf":
        if not rdf_data or not rdf_data.get("edges"):
            return None
        return GraphFormatter.format(
            [], rdf_data["edges"], style="triple",
        )

    if condition == "text":
        if not text_references:
            return None
        try:
            parsed = ast.literal_eval(text_references) if isinstance(text_references, str) else text_references
            if hasattr(parsed, "__iter__") and not isinstance(parsed, str):
                return "\n".join(str(s) for s in parsed)
            return str(text_references)
        except (ValueError, SyntaxError):
            return str(text_references) if text_references else None

    if condition == "lpg_rdf":
        parts = []
        if lpg_data and lpg_data.get("edges"):
            lpg_ctx = GraphFormatter.format(
                lpg_data["nodes"], lpg_data["edges"], style="structured",
            )
            if lpg_ctx:
                parts.append(f"=== LPG Graph Context ===\n{lpg_ctx}")
        if rdf_data and rdf_data.get("edges"):
            rdf_ctx = GraphFormatter.format(
                [], rdf_data["edges"], style="triple",
            )
            if rdf_ctx:
                parts.append(f"=== RDF Triple Context ===\n{rdf_ctx}")
        return "\n\n".join(parts) if parts else None

    raise ValueError(f"Unknown condition: {condition}")


# ── Neo4j Data Fetchers ──


def fetch_lpg_subgraph(client: Neo4jClient, question_id: str) -> Dict[str, Any]:
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


def fetch_rdf_subgraph(client: Neo4jClient, question_id: str) -> Dict[str, Any]:
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


# ── Experiment Logger ──


class ExperimentLogger:
    """Log experiment results to structured JSON Lines files."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_raw_dir = output_dir / "metrics_raw"
        self.metrics_raw_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_log = output_dir / "metrics_log.jsonl"
        self.results_log = output_dir / "results_log.jsonl"
        self.summary_path = output_dir / "summary.json"
        self.summary_csv_path = output_dir / "summary.csv"
        self.config_path = output_dir / "config.json"

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration snapshot."""
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def save_raw_metrics(
        self, question_id: str, condition: str, phase: str, raw_text: str,
    ) -> None:
        """Save raw Prometheus metrics dump to a file."""
        filename = f"{question_id}_{condition}_{phase}.txt"
        with open(self.metrics_raw_dir / filename, "w") as f:
            f.write(raw_text)

    def log_metrics(self, entry: Dict[str, Any]) -> None:
        """Append a metrics snapshot to the log (without raw text)."""
        # Strip raw text before logging to JSONL
        clean = {k: v for k, v in entry.items() if k != "raw"}
        # Also strip raw from nested snapshots
        for key in ["before", "after", "pre_cold", "post_cold", "pre_warm", "post_warm"]:
            if key in clean and isinstance(clean[key], dict) and "raw" in clean[key]:
                clean[key] = {k: v for k, v in clean[key].items() if k != "raw"}
        with open(self.metrics_log, "a") as f:
            f.write(json.dumps(clean, default=str) + "\n")

    def log_result(self, entry: Dict[str, Any]) -> None:
        """Append a result entry to the log."""
        with open(self.results_log, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """Save experiment summary as JSON and CSV."""
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Also save per-condition stats as CSV
        if "per_condition" in summary:
            rows = []
            for cond, stats in summary["per_condition"].items():
                rows.append({"condition": cond, **stats})
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(self.summary_csv_path, index=False)

    def save_checkpoint(self, checkpoint_data: Dict[str, Any], question_idx: int) -> None:
        """Save periodic checkpoint."""
        path = self.output_dir / f"checkpoint_q{question_idx}.json"
        with open(path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)


# ── Data Loading (reuses attention_experiment logic) ──


def load_common_question_ids(path: str = DEFAULT_COMMON_IDS_PATH) -> List[str]:
    """Load pre-computed common question IDs (LPG ∩ RDF)."""
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

        if remaining > 0:
            used_ids = {q["id"] for q in selected}
            extras = [q for q in questions if q["id"] not in used_ids]
            rng.shuffle(extras)
            selected.extend(extras[:remaining])

    rng.shuffle(selected)
    logger.info(f"Sampled {len(selected)} questions from {len(questions)} candidates")
    return list(selected)


def load_text_references(parquet_path: str) -> Dict[str, str]:
    """Load question_id → references mapping from parquet.

    The parquet uses `_id` as the question ID column and `references`
    as a numpy array of reference strings.
    """
    df = pd.read_parquet(parquet_path)
    # Detect ID column: _id > question_id > id
    id_col = None
    for candidate in ["_id", "question_id", "id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        logger.warning(f"No ID column found in {parquet_path}, columns: {list(df.columns)}")
        return {}

    ref_lookup = {}
    for _, row in df.iterrows():
        qid = str(row[id_col])
        refs = row.get("references")
        if qid and refs is not None:
            # references may be a numpy array or a string
            if hasattr(refs, "__iter__") and not isinstance(refs, str):
                ref_lookup[qid] = "\n".join(str(s) for s in refs)
            else:
                ref_lookup[qid] = str(refs)
    return ref_lookup


# ── Cold/Warm Measurement Protocol ──


def run_cold_warm_pair(
    client: VLLMClient,
    collector: VLLMMetricsCollector,
    exp_logger: ExperimentLogger,
    question_id: str,
    question: str,
    condition: str,
    context: Optional[str],
    max_tokens: int,
    lmcache_wait_sec: float,
) -> Dict[str, Any]:
    """Run cold+warm pair for a single (question, condition).

    Protocol:
    1. pre_cold = scrape /metrics → raw dump + parsed
    2. cold_resp = API call
    3. post_cold = scrape /metrics → raw dump + parsed + cold_delta
    4. sleep(lmcache_wait_sec) — LMCache store 대기
    5. pre_warm = scrape /metrics → raw dump + parsed
    6. warm_resp = API call (identical prompt)
    7. post_warm = scrape /metrics → raw dump + parsed + warm_delta
    """
    result: Dict[str, Any] = {
        "question_id": question_id,
        "condition": condition,
    }

    try:
        # ── Cold run ──
        pre_cold = collector.snapshot()
        exp_logger.save_raw_metrics(question_id, condition, "pre_cold", pre_cold["raw"])

        cold_resp = client.generate(
            question=question, context=context, max_tokens=max_tokens,
        )

        post_cold = collector.snapshot()
        exp_logger.save_raw_metrics(question_id, condition, "post_cold", post_cold["raw"])
        cold_delta = collector.delta(pre_cold, post_cold)

        # Log cold metrics
        exp_logger.log_metrics({
            "question_id": question_id,
            "condition": condition,
            "phase": "cold",
            "pre_cold": pre_cold,
            "post_cold": post_cold,
            "delta": cold_delta,
        })

        # ── Wait for LMCache store ──
        time.sleep(lmcache_wait_sec)

        # ── Warm run ──
        pre_warm = collector.snapshot()
        exp_logger.save_raw_metrics(question_id, condition, "pre_warm", pre_warm["raw"])

        warm_resp = client.generate(
            question=question, context=context, max_tokens=max_tokens,
        )

        post_warm = collector.snapshot()
        exp_logger.save_raw_metrics(question_id, condition, "post_warm", post_warm["raw"])
        warm_delta = collector.delta(pre_warm, post_warm)

        # Log warm metrics
        exp_logger.log_metrics({
            "question_id": question_id,
            "condition": condition,
            "phase": "warm",
            "pre_warm": pre_warm,
            "post_warm": post_warm,
            "delta": warm_delta,
        })

        # ── Assemble result ──
        cold_latency = cold_resp["latency_sec"]
        warm_latency = warm_resp["latency_sec"]
        speedup = cold_latency / max(warm_latency, 0.001)

        result.update({
            "status": "success",
            # Latency
            "cold_latency": cold_latency,
            "warm_latency": warm_latency,
            "speedup": speedup,
            # Tokens
            "prompt_tokens": cold_resp["prompt_tokens"],
            "cold_completion_tokens": cold_resp["completion_tokens"],
            "warm_completion_tokens": warm_resp["completion_tokens"],
            # Responses
            "cold_response_text": cold_resp["response_text"],
            "warm_response_text": warm_resp["response_text"],
            "cold_reasoning_text": cold_resp.get("reasoning_text", ""),
            "warm_reasoning_text": warm_resp.get("reasoning_text", ""),
            "cold_finish_reason": cold_resp["finish_reason"],
            "warm_finish_reason": warm_resp["finish_reason"],
            # Cache deltas
            "cold_delta": cold_delta,
            "warm_delta": warm_delta,
            # Context length (chars)
            "context_length": len(context) if context else 0,
        })

    except Exception as e:
        logger.error(f"Error on {question_id}/{condition}: {e}")
        result.update({
            "status": "error",
            "error": str(e),
        })

    return result


# ── Full Experiment ──


def run_full_experiment(args: KVCacheExperimentArgs) -> None:
    """Run the full KV cache experiment."""
    set_seed(args.seed)

    # ── Output setup ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    exp_logger = ExperimentLogger(output_dir)
    exp_logger.save_config(asdict(args))

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Conditions: {args.conditions}")
    logger.info(f"Sample size: {args.sample_size}")

    # ── Clients ──
    vllm_client = VLLMClient()
    collector = VLLMMetricsCollector()
    evaluator = Evaluator(use_bertscore=False, use_rouge=True)

    neo4j_cfg = Neo4jConfig(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )
    neo4j_client = Neo4jClient(neo4j_cfg)
    neo4j_client.connect()

    # ── Load data ──
    logger.info("Loading common question IDs...")
    common_ids = load_common_question_ids(args.common_ids_path)
    logger.info(f"Loaded {len(common_ids)} common question IDs")

    logger.info("Sampling questions...")
    questions = sample_questions(neo4j_client, common_ids, args.sample_size, args.seed)
    logger.info(f"Sampled {len(questions)} questions")

    # Text references
    ref_lookup: Dict[str, str] = {}
    if "text" in args.conditions or "lpg_rdf" in args.conditions:
        logger.info("Loading text references from parquet...")
        ref_lookup = load_text_references(args.parquet_path)
        logger.info(f"Loaded references for {len(ref_lookup)} questions")

    # ── Resume support ──
    completed_pairs: set = set()  # (question_id, condition)
    if args.resume_dir:
        resume_results = Path(args.resume_dir) / "results_log.jsonl"
        if resume_results.exists():
            with open(resume_results) as f:
                for line in f:
                    entry = json.loads(line)
                    qid = entry.get("question_id")
                    cond = entry.get("condition")
                    status = entry.get("status")
                    # Only skip truly completed (success) pairs, not skipped ones
                    if qid and cond and status == "success":
                        completed_pairs.add((qid, cond))
            logger.info(f"Resuming: {len(completed_pairs)} pairs already completed")
            # Use the resume dir instead of creating new
            output_dir = Path(args.resume_dir)
            exp_logger = ExperimentLogger(output_dir)

    # ── Main loop ──
    all_results: List[Dict[str, Any]] = []
    n_success = 0
    n_error = 0
    n_skipped = 0

    for q_idx, q in enumerate(questions):
        qid = q["id"]
        question_text = q["text"]
        answer = q.get("answer", "")

        logger.info(
            f"[{q_idx + 1}/{len(questions)}] Question {qid}: "
            f"{question_text[:60]}..."
        )

        # Fetch graph data once per question
        lpg_data = None
        rdf_data = None
        text_refs = ref_lookup.get(qid)

        needs_lpg = any(c in args.conditions for c in ["lpg", "lpg_rdf"])
        needs_rdf = any(c in args.conditions for c in ["rdf", "lpg_rdf"])

        if needs_lpg:
            try:
                lpg_data = fetch_lpg_subgraph(neo4j_client, qid)
            except Exception as e:
                logger.warning(f"Failed to fetch LPG for {qid}: {e}")

        if needs_rdf:
            try:
                rdf_data = fetch_rdf_subgraph(neo4j_client, qid)
            except Exception as e:
                logger.warning(f"Failed to fetch RDF for {qid}: {e}")

        for condition in args.conditions:
            if (qid, condition) in completed_pairs:
                logger.info(f"  Skipping {condition} (already completed)")
                n_skipped += 1
                continue

            context = build_context(condition, lpg_data, rdf_data, text_refs)

            # Skip if required context is missing (except vanilla)
            if condition != "vanilla" and context is None:
                logger.warning(f"  Skipping {condition}: no context available")
                exp_logger.log_result({
                    "question_id": qid,
                    "condition": condition,
                    "status": "skipped",
                    "reason": "no_context",
                })
                n_skipped += 1
                continue

            logger.info(
                f"  Condition: {condition} "
                f"(context: {len(context) if context else 0} chars)"
            )

            result = run_cold_warm_pair(
                client=vllm_client,
                collector=collector,
                exp_logger=exp_logger,
                question_id=qid,
                question=question_text,
                condition=condition,
                context=context,
                max_tokens=args.max_tokens,
                lmcache_wait_sec=args.lmcache_wait_sec,
            )

            # Evaluate answer quality (cold response only)
            if result.get("status") == "success" and answer:
                try:
                    eval_result = evaluator.evaluate_single(
                        result["cold_response_text"], answer,
                    )
                    result["eval"] = asdict(eval_result)
                except Exception as e:
                    logger.warning(f"  Evaluation failed: {e}")

            # Log
            result["question_text"] = question_text
            result["answer"] = answer
            exp_logger.log_result(result)
            all_results.append(result)

            if result.get("status") == "success":
                n_success += 1
                logger.info(
                    f"    Cold: {result['cold_latency']:.2f}s, "
                    f"Warm: {result['warm_latency']:.2f}s, "
                    f"Speedup: {result['speedup']:.2f}x"
                )
            else:
                n_error += 1
                logger.error(f"    Error: {result.get('error', 'unknown')}")

            # Inter-condition wait
            time.sleep(args.inter_question_wait)

        # Checkpoint
        if (q_idx + 1) % args.checkpoint_every == 0:
            logger.info(f"  Saving checkpoint at question {q_idx + 1}...")
            exp_logger.save_checkpoint(
                {
                    "question_idx": q_idx + 1,
                    "n_success": n_success,
                    "n_error": n_error,
                    "n_skipped": n_skipped,
                    "total_results": len(all_results),
                },
                q_idx + 1,
            )

    # ── Summary ──
    logger.info("Computing summary...")
    summary = _compute_summary(all_results, args)
    summary["n_success"] = n_success
    summary["n_error"] = n_error
    summary["n_skipped"] = n_skipped
    exp_logger.save_summary(summary)

    logger.info(f"Experiment complete: {n_success} success, {n_error} error, {n_skipped} skipped")
    logger.info(f"Results saved to: {output_dir}/")

    neo4j_client.close()


def _compute_summary(
    results: List[Dict[str, Any]],
    args: KVCacheExperimentArgs,
) -> Dict[str, Any]:
    """Compute per-condition aggregate statistics."""
    per_condition: Dict[str, Dict[str, Any]] = {}

    for condition in args.conditions:
        cond_results = [
            r for r in results
            if r.get("condition") == condition and r.get("status") == "success"
        ]
        if not cond_results:
            per_condition[condition] = {"n_success": 0}
            continue

        cold_lats = [r["cold_latency"] for r in cond_results]
        warm_lats = [r["warm_latency"] for r in cond_results]
        speedups = [r["speedup"] for r in cond_results]
        prompt_toks = [r["prompt_tokens"] for r in cond_results]
        ctx_lens = [r["context_length"] for r in cond_results]

        stats: Dict[str, Any] = {
            "n_success": len(cond_results),
            # Latency
            "cold_latency_mean": float(np.mean(cold_lats)),
            "cold_latency_std": float(np.std(cold_lats)),
            "warm_latency_mean": float(np.mean(warm_lats)),
            "warm_latency_std": float(np.std(warm_lats)),
            "speedup_mean": float(np.mean(speedups)),
            "speedup_std": float(np.std(speedups)),
            # Tokens
            "prompt_tokens_mean": float(np.mean(prompt_toks)),
            "context_length_mean": float(np.mean(ctx_lens)),
        }

        # Cache deltas (warm run)
        warm_cache_hits = []
        warm_cache_queries = []
        for r in cond_results:
            wd = r.get("warm_delta", {})
            hit_key = "delta_vllm:gpu_prefix_cache_hits_total"
            query_key = "delta_vllm:gpu_prefix_cache_queries_total"
            if hit_key in wd:
                warm_cache_hits.append(wd[hit_key])
            if query_key in wd:
                warm_cache_queries.append(wd[query_key])

        if warm_cache_hits:
            stats["warm_cache_hit_delta_mean"] = float(np.mean(warm_cache_hits))
        if warm_cache_queries:
            stats["warm_cache_queries_delta_mean"] = float(np.mean(warm_cache_queries))

        # Quality metrics
        eval_results = [r["eval"] for r in cond_results if "eval" in r]
        if eval_results:
            stats["em_mean"] = float(np.mean([e["exact_match"] for e in eval_results]))
            stats["f1_mean"] = float(np.mean([e["token_f1"] for e in eval_results]))
            stats["rouge_l_mean"] = float(np.mean([e["rouge_l"] for e in eval_results]))

        per_condition[condition] = stats

    return {
        "experiment_time": datetime.now().isoformat(),
        "args": asdict(args),
        "per_condition": per_condition,
    }


# ── Test Functions ──


def test_metrics_logging():
    """Test API connectivity and metrics logging with cold/warm comparison."""
    print("=" * 70)
    print("KV Cache Metrics Logging Test")
    print("=" * 70)

    collector = VLLMMetricsCollector()
    client = VLLMClient()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/kvcache_test/{timestamp}")
    exp_logger = ExperimentLogger(output_dir)

    # 1. Test /metrics endpoint
    print("\n[1/5] Testing /metrics endpoint...")
    try:
        snap = collector.snapshot()
        print(f"  Timestamp: {snap['timestamp']}")
        print(f"  Server: {snap['server']}")
        print(f"  Metrics found: {len(snap['metrics'])}")
        for k, v in sorted(snap["metrics"].items())[:20]:
            print(f"    {k}: {v}")
        if len(snap["metrics"]) > 20:
            print(f"    ... and {len(snap['metrics']) - 20} more")
        exp_logger.log_metrics({"phase": "initial", **snap})
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # 2. Test API - simple short prompt
    print("\n[2/5] Testing API (short prompt)...")
    try:
        resp = client.generate(
            question="What is a knowledge graph?",
            max_tokens=128,
        )
        print(f"  Response: {resp['response_text'][:100]}")
        if resp.get("reasoning_text"):
            print(f"  Reasoning: {resp['reasoning_text'][:100]}...")
        print(f"  Latency: {resp['latency_sec']:.3f}s")
        print(f"  Tokens: prompt={resp['prompt_tokens']}, "
              f"completion={resp['completion_tokens']}")
        print(f"  Finish reason: {resp['finish_reason']}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # 3. Cold run with longer context
    print("\n[3/5] Cold run (with context, first request)...")
    test_context = (
        "The Federal Reserve (Fed) is the central banking system of the "
        "United States. It was created on December 23, 1913. The Fed's "
        "main duties include managing monetary policy, supervising banks, "
        "maintaining financial stability, and providing banking services. "
        "The current chair is Jerome Powell, appointed in 2018."
    )
    test_question = "Who is the current chair of the Federal Reserve?"

    snap_before_cold = collector.snapshot()
    cold_resp = client.generate(
        question=test_question,
        context=test_context,
        max_tokens=128,
    )
    snap_after_cold = collector.snapshot()
    cold_delta = collector.delta(snap_before_cold, snap_after_cold)

    print(f"  Response: {cold_resp['response_text'][:100]}")
    if cold_resp.get("reasoning_text"):
        print(f"  Reasoning: {cold_resp['reasoning_text'][:100]}...")
    print(f"  Latency: {cold_resp['latency_sec']:.3f}s")
    print(f"  Tokens: prompt={cold_resp['prompt_tokens']}, "
          f"completion={cold_resp['completion_tokens']}")
    print(f"  Metrics delta:")
    for k, v in sorted(cold_delta.items()):
        if v != 0:
            print(f"    {k}: {v}")

    exp_logger.log_metrics({
        "phase": "cold_run",
        "before": snap_before_cold,
        "after": snap_after_cold,
        "delta": cold_delta,
    })
    exp_logger.log_result({"phase": "cold_run", **cold_resp})

    # 4. Warm run (same prompt → should hit LMCache)
    print("\n[4/5] Warm run (same prompt, should hit KV cache)...")
    time.sleep(3)  # Give LMCache time to store

    snap_before_warm = collector.snapshot()
    warm_resp = client.generate(
        question=test_question,
        context=test_context,
        max_tokens=128,
    )
    snap_after_warm = collector.snapshot()
    warm_delta = collector.delta(snap_before_warm, snap_after_warm)

    print(f"  Response: {warm_resp['response_text'][:100]}")
    if warm_resp.get("reasoning_text"):
        print(f"  Reasoning: {warm_resp['reasoning_text'][:100]}...")
    print(f"  Latency: {warm_resp['latency_sec']:.3f}s")
    print(f"  Tokens: prompt={warm_resp['prompt_tokens']}, "
          f"completion={warm_resp['completion_tokens']}")
    print(f"  Metrics delta:")
    for k, v in sorted(warm_delta.items()):
        if v != 0:
            print(f"    {k}: {v}")

    exp_logger.log_metrics({
        "phase": "warm_run",
        "before": snap_before_warm,
        "after": snap_after_warm,
        "delta": warm_delta,
    })
    exp_logger.log_result({"phase": "warm_run", **warm_resp})

    # 5. Summary
    print("\n[5/5] Summary")
    print("-" * 40)
    speedup = cold_resp["latency_sec"] / max(warm_resp["latency_sec"], 0.001)
    print(f"  Cold latency:  {cold_resp['latency_sec']:.3f}s")
    print(f"  Warm latency:  {warm_resp['latency_sec']:.3f}s")
    print(f"  Speedup:       {speedup:.2f}x")

    cache_hit_before = snap_before_warm["metrics"].get(
        "vllm:gpu_prefix_cache_hit_rate", 0,
    )
    cache_hit_after = snap_after_warm["metrics"].get(
        "vllm:gpu_prefix_cache_hit_rate", 0,
    )
    print(f"  Cache hit rate: {cache_hit_before:.4f} → {cache_hit_after:.4f}")

    if speedup >= 1.5:
        print("\n  [OK] KV Cache offloading appears to be working!")
    else:
        print("\n  [WARN] Speedup < 1.5x — cache may not be effective "
              "for this prompt length, or load balancer routed to different pods.")

    summary = {
        "test_time": timestamp,
        "cold_latency": cold_resp["latency_sec"],
        "warm_latency": warm_resp["latency_sec"],
        "speedup": speedup,
        "cold_prompt_tokens": cold_resp["prompt_tokens"],
        "warm_prompt_tokens": warm_resp["prompt_tokens"],
        "cache_hit_rate_before": cache_hit_before,
        "cache_hit_rate_after": cache_hit_after,
    }
    exp_logger.save_summary(summary)

    print(f"\n  Logs saved to: {output_dir}/")
    print(f"    - metrics_log.jsonl")
    print(f"    - results_log.jsonl")
    print(f"    - summary.json")

    return True


# ── Entry Point ──


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KV Cache Offloading Experiment",
    )
    parser.add_argument(
        "--test-metrics", action="store_true",
        help="Test API + metrics logging (no Neo4j needed)",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS,
        help="Context conditions to test",
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--lmcache-wait", type=float, default=3.0,
        help="Seconds to wait between cold and warm runs for LMCache",
    )
    parser.add_argument(
        "--inter-question-wait", type=float, default=1.0,
        help="Seconds to wait between conditions",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Checkpoint every N questions",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="results/kvcache_experiment",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--parquet-path", type=str,
        default=DEFAULT_PARQUET_PATH,
    )
    parser.add_argument(
        "--neo4j-uri", type=str, default="bolt://localhost:7687",
    )
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="password")

    args = parser.parse_args()

    setup_logging()

    if args.test_metrics:
        success = test_metrics_logging()
        exit(0 if success else 1)

    # Build experiment args
    exp_args = KVCacheExperimentArgs(
        conditions=args.conditions,
        sample_size=args.sample_size,
        seed=args.seed,
        max_tokens=args.max_tokens,
        lmcache_wait_sec=args.lmcache_wait,
        inter_question_wait=args.inter_question_wait,
        output_dir=args.output_dir,
        checkpoint_every=args.checkpoint_every,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        parquet_path=args.parquet_path,
        resume_dir=args.resume,
    )

    run_full_experiment(exp_args)
