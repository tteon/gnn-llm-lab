"""
Opik-integrated experiment driver for soft-prompt Graph RAG evaluation.

Runs the 3-axis experiment matrix (models x contexts x few-shot) with
Opik tracing, evaluation, and dashboard integration.

Usage:
    # Basic run
    uv run python src/opik_experiment.py \
        --models llama8b --contexts none lpg rdf --sample-size 50

    # Full matrix + LLM-as-Judge
    uv run python src/opik_experiment.py \
        --models llama8b mixtral --contexts none lpg rdf lpg_rdf \
        --few-shot --sample-size 100 --judge-model gpt-4o-mini

    # Heuristic only (no judge, no BERTScore)
    uv run python src/opik_experiment.py \
        --models llama8b --contexts none lpg --sample-size 20 \
        --no-judge --no-bertscore
"""

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import opik
from opik import Opik, opik_context
from opik.evaluation import evaluate
from opik.evaluation.metrics import base_metric
from opik.evaluation.metrics.score_result import ScoreResult

from src.utils import (
    ExperimentConfig,
    FewShotConfig,
    FewShotSelector,
    GraphFormatter,
    LocalLLMManager,
    Neo4jClient,
    Neo4jConfig,
    get_logger,
    set_seed,
    setup_logging,
)
from src.utils.evaluation import Evaluator

logger = get_logger("opik_experiment")

VALID_CONTEXTS = {"none", "text", "lpg", "rdf", "lpg_rdf"}

# ---------------------------------------------------------------------------
# Custom Opik Metrics — wrap existing Evaluator methods
# ---------------------------------------------------------------------------


class ExactMatchMetric(base_metric.BaseMetric):
    """Normalized exact match via Evaluator.compute_exact_match()."""

    def __init__(self):
        super().__init__(name="exact_match", track=True)

    def score(self, output: str, expected_output: str, **ignored) -> ScoreResult:
        value = Evaluator.compute_exact_match(output, expected_output)
        return ScoreResult(value=value, name=self.name)


class TokenF1Metric(base_metric.BaseMetric):
    """Token-level F1 via Evaluator.compute_token_f1()."""

    def __init__(self):
        super().__init__(name="token_f1", track=True)

    def score(self, output: str, expected_output: str, **ignored) -> ScoreResult:
        value = Evaluator.compute_token_f1(output, expected_output)
        return ScoreResult(value=value, name=self.name)


class RougeMetric(base_metric.BaseMetric):
    """ROUGE-1/2/L via Evaluator.compute_rouge(). Returns list of ScoreResult."""

    def __init__(self):
        super().__init__(name="rouge", track=True)
        self._evaluator: Optional[Evaluator] = None

    def score(self, output: str, expected_output: str, **ignored) -> List[ScoreResult]:
        if self._evaluator is None:
            self._evaluator = Evaluator(use_bertscore=False, use_rouge=True)
        scores = self._evaluator.compute_rouge(output, expected_output)
        return [
            ScoreResult(value=scores["rouge1"], name="rouge1"),
            ScoreResult(value=scores["rouge2"], name="rouge2"),
            ScoreResult(value=scores["rougeL"], name="rougeL"),
        ]


class BertScoreMetric(base_metric.BaseMetric):
    """BERTScore via Evaluator._compute_bertscore_batch(). Lazy init."""

    def __init__(self):
        super().__init__(name="bertscore_f1", track=True)
        self._evaluator: Optional[Evaluator] = None

    def score(self, output: str, expected_output: str, **ignored) -> ScoreResult:
        if self._evaluator is None:
            self._evaluator = Evaluator(use_bertscore=True, use_rouge=False)
        results = self._evaluator._compute_bertscore_batch([output], [expected_output])
        value = results[0] if results else 0.0
        return ScoreResult(value=value, name=self.name)


# ---------------------------------------------------------------------------
# OpikGraphRAGExperiment
# ---------------------------------------------------------------------------


class OpikGraphRAGExperiment:
    """Opik-integrated soft-prompt Graph RAG experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        opik_project: str = "FinDER_GraphRAG",
        judge_model: Optional[str] = "gpt-4o-mini",
        use_judge: bool = True,
        use_bertscore: bool = True,
    ):
        self.config = config
        self.opik_project = opik_project
        self.judge_model = judge_model
        self.use_judge = use_judge
        self.use_bertscore = use_bertscore

        self.local_llm: Optional[LocalLLMManager] = None
        self.neo4j_lpg: Optional[Neo4jClient] = None
        self.neo4j_rdf: Optional[Neo4jClient] = None
        self.few_shot_selector: Optional[FewShotSelector] = None
        self.opik_client: Optional[Opik] = None

    def setup(self, contexts: Set[str]) -> None:
        """Initialize Neo4j connections, LLM manager, and Opik client."""
        mc = self.config.model

        # LocalLLMManager (models loaded/unloaded per-alias in run_all)
        self.local_llm = LocalLLMManager(mc)
        logger.info("LocalLLMManager initialized")

        # Neo4j connections
        needs_lpg = contexts & {"lpg", "lpg_rdf"}
        if needs_lpg:
            logger.info("Connecting to Neo4j (finderlpg)...")
            self.neo4j_lpg = Neo4jClient(Neo4jConfig(database="finderlpg"))
            self.neo4j_lpg.connect()
            info = self.neo4j_lpg.get_database_info()
            logger.info(f"finderlpg: {info['node_count']:,} nodes, {info['edge_count']:,} edges")

        needs_rdf = contexts & {"rdf", "lpg_rdf"}
        if needs_rdf:
            logger.info("Connecting to Neo4j (finderrdf)...")
            self.neo4j_rdf = Neo4jClient(Neo4jConfig(database="finderrdf"))
            self.neo4j_rdf.connect()
            info = self.neo4j_rdf.get_database_info()
            logger.info(f"finderrdf: {info['node_count']:,} nodes, {info['edge_count']:,} edges")

        # Few-shot selector
        if self.config.few_shot.enabled:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedder for few-shot: {mc.embedding_model_id}")
            self.few_shot_selector = FewShotSelector(self.config.few_shot)
            if not self.few_shot_selector.load():
                logger.info("Few-shot cache not found, will build during run_all")

        # Opik client
        self.opik_client = Opik(project_name=self.opik_project)
        logger.info(f"Opik client initialized, project: {self.opik_project}")

    def cleanup(self) -> None:
        """Release resources."""
        if self.local_llm:
            self.local_llm.unload_model()
        if self.neo4j_lpg:
            self.neo4j_lpg.close()
        if self.neo4j_rdf:
            self.neo4j_rdf.close()
        logger.info("Resources cleaned up")

    # ------------------------------------------------------------------
    # Context building (replicated from run_experiment.py)
    # ------------------------------------------------------------------

    def _load_subgraph_rdf(self, question_id: str) -> dict:
        """Load RDF subgraph from finderrdf database."""
        query = """
        MATCH (r:Resource)-[rel]->(r2:Resource)
        WHERE rel.question_id = $qid
        RETURN collect(DISTINCT {id: r.uri, uri: r.uri})
             + collect(DISTINCT {id: r2.uri, uri: r2.uri}) as nodes,
               collect(DISTINCT {source: r.uri, target: r2.uri, type: type(rel)}) as edges
        """
        try:
            result = self.neo4j_rdf.query_single(query, {"qid": question_id})
            if result:
                seen = set()
                unique_nodes = []
                for n in result["nodes"]:
                    if n and n.get("id") and n["id"] not in seen:
                        seen.add(n["id"])
                        unique_nodes.append(n)
                valid_edges = [
                    e for e in result["edges"]
                    if e.get("source") and e.get("target")
                ]
                return {
                    "nodes": unique_nodes[:50],
                    "edges": valid_edges[:100],
                }
        except Exception as e:
            logger.warning(f"RDF subgraph load failed for {question_id}: {e}")
        return {"nodes": [], "edges": []}

    def _build_context(
        self, context_mode: str, question_id: str, references: str = ""
    ) -> tuple:
        """
        Build text context string for a given mode.

        Returns:
            (context_str, subgraph_info_dict, entity_names)
        """
        info = {
            "lpg_nodes": 0, "lpg_edges": 0,
            "rdf_nodes": 0, "rdf_edges": 0,
        }
        entity_names = []

        if context_mode == "none":
            return None, info, entity_names

        if context_mode == "text":
            try:
                parsed = ast.literal_eval(references) if isinstance(references, str) else references
                if hasattr(parsed, '__iter__') and not isinstance(parsed, str):
                    context = "\n".join(str(s) for s in parsed)
                else:
                    context = str(references) if references else ""
            except (ValueError, SyntaxError):
                context = str(references) if references else ""
            return context if context else None, info, entity_names

        lpg_context = None
        lpg_subgraph = None
        rdf_context = None
        rdf_subgraph = None

        # LPG: fetch subgraph -> format as text
        if context_mode in ("lpg", "lpg_rdf") and self.neo4j_lpg:
            lpg_subgraph = self.neo4j_lpg.get_subgraph(
                question_id, max_hops=self.config.max_hops
            )
            nodes = lpg_subgraph.get("nodes", [])
            edges = lpg_subgraph.get("edges", [])
            if nodes:
                info["lpg_nodes"] = len(nodes)
                info["lpg_edges"] = len(edges)
                entity_names.extend(
                    n.get("name", n.get("id", ""))
                    for n in nodes
                )

        # RDF: fetch subgraph
        if context_mode in ("rdf", "lpg_rdf") and self.neo4j_rdf:
            rdf_subgraph = self._load_subgraph_rdf(question_id)
            rdf_nodes = rdf_subgraph.get("nodes", [])
            rdf_edges = rdf_subgraph.get("edges", [])
            if rdf_edges:
                info["rdf_nodes"] = len(rdf_nodes)
                info["rdf_edges"] = len(rdf_edges)

        # Format context as text
        if context_mode == "lpg":
            if lpg_subgraph and info["lpg_nodes"] > 0:
                lpg_context = GraphFormatter.format(
                    nodes=lpg_subgraph["nodes"],
                    edges=lpg_subgraph["edges"],
                    style=self.config.soft_prompt_format,
                    max_nodes=self.config.max_context_nodes,
                    max_edges=self.config.max_context_edges,
                )
            return lpg_context, info, entity_names

        elif context_mode == "rdf":
            if rdf_subgraph and info["rdf_edges"] > 0:
                rdf_edges = rdf_subgraph.get("edges", [])
                rdf_context = GraphFormatter.format_rdf_cleaned(rdf_edges[:40])
            return rdf_context, info, entity_names

        elif context_mode == "lpg_rdf":
            if (lpg_subgraph and info["lpg_nodes"] > 0) or (rdf_subgraph and info["rdf_edges"] > 0):
                combined = GraphFormatter.format_combined(
                    lpg_nodes=lpg_subgraph.get("nodes", []) if lpg_subgraph else [],
                    lpg_edges=lpg_subgraph.get("edges", []) if lpg_subgraph else [],
                    rdf_edges=rdf_subgraph.get("edges", []) if rdf_subgraph else [],
                    style=self.config.soft_prompt_format,
                )
                return combined, info, entity_names
            return None, info, entity_names

        return None, info, entity_names

    # ------------------------------------------------------------------
    # Opik dataset
    # ------------------------------------------------------------------

    def create_dataset(self, df, dataset_name: str = "FinDER_RAG_v1") -> opik.Dataset:
        """Create or update an Opik dataset from a DataFrame."""
        dataset = self.opik_client.get_or_create_dataset(name=dataset_name)

        items = []
        for _, row in df.iterrows():
            items.append({
                "question": row["text"],
                "question_id": str(row["_id"]),
                "category": str(row.get("category", "")),
                "references": str(row.get("references", "")),
                "expected_output": str(row.get("answer", "")),
            })

        dataset.insert(items)
        logger.info(f"Opik dataset '{dataset_name}': {len(items)} items")
        return dataset

    # ------------------------------------------------------------------
    # Evaluation task builder
    # ------------------------------------------------------------------

    def build_evaluation_task(
        self, context_mode: str, few_shot: bool, model_alias: str
    ):
        """
        Build an Opik evaluation task closure.

        The returned function processes one dataset item:
        question -> context building -> LLM generation -> traced output.
        """
        experiment = self

        @opik.track(name=f"rag_{context_mode}", project_name=self.opik_project)
        def task(item: dict) -> dict:
            question = item["question"]
            question_id = item.get("question_id", "")
            category = item.get("category", "")
            references = item.get("references", "")

            # Build context
            context, subgraph_info, entity_names = experiment._build_context(
                context_mode, question_id, references
            )

            # Build few-shot examples
            few_shot_examples = None
            if few_shot and experiment.few_shot_selector:
                examples = experiment.few_shot_selector.get_examples(
                    category, exclude_question_id=question_id
                )
                if examples:
                    few_shot_examples = experiment.few_shot_selector.format_for_prompt(examples)

            # Generate
            resp = experiment.local_llm.generate(
                question=question,
                context=context,
                few_shot_examples=few_shot_examples,
                max_new_tokens=experiment.config.max_new_tokens,
                temperature=experiment.config.temperature,
            )

            # Log metadata to Opik trace
            opik_context.update_current_trace(
                metadata={
                    "model": model_alias,
                    "context_mode": context_mode,
                    "few_shot": few_shot,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "generation_time": resp.generation_time,
                    "question_id": question_id,
                    "category": category,
                    **subgraph_info,
                },
                tags=[
                    model_alias,
                    f"ctx:{context_mode}",
                    "fewshot" if few_shot else "zeroshot",
                ],
            )

            return {
                "output": resp.text,
                "context": [context] if context else [],
                "input": question,
            }

        return task

    # ------------------------------------------------------------------
    # Main experiment loop
    # ------------------------------------------------------------------

    def run_all(
        self,
        df,
        models: List[str],
        contexts: List[str],
        run_few_shot: bool = False,
        dataset_name: str = "FinDER_RAG_v1",
    ) -> None:
        """
        Run the full 3-axis experiment matrix with Opik evaluate().

        Each (model, context, few_shot) combination = 1 Opik experiment.
        """
        # Create Opik dataset
        dataset = self.create_dataset(df, dataset_name=dataset_name)

        # Build scoring metrics
        heuristic_metrics: List[base_metric.BaseMetric] = [
            ExactMatchMetric(),
            TokenF1Metric(),
            RougeMetric(),
        ]
        if self.use_bertscore:
            heuristic_metrics.append(BertScoreMetric())

        judge_metrics = []
        if self.use_judge and self.judge_model:
            try:
                from opik.evaluation.metrics import AnswerRelevance, ContextPrecision, Hallucination
                judge_metrics = [
                    AnswerRelevance(model=self.judge_model),
                    Hallucination(model=self.judge_model),
                    ContextPrecision(model=self.judge_model),
                ]
                logger.info(f"LLM-as-Judge enabled: {self.judge_model} (3 metrics)")
            except ImportError:
                logger.warning("Opik LLM-as-Judge metrics not available, using heuristic only")

        all_metrics = heuristic_metrics + judge_metrics

        # Few-shot variations
        few_shot_flags = [False]
        if run_few_shot:
            few_shot_flags.append(True)

        total = len(models) * len(contexts) * len(few_shot_flags)
        logger.info(
            f"Starting {total} experiments: "
            f"{len(models)} models x {len(contexts)} contexts x {len(few_shot_flags)} shot"
        )

        run_idx = 0
        for model_alias in models:
            logger.info(f"Loading model: {model_alias}")
            self.local_llm.load_model(model_alias)

            for context_mode in contexts:
                for few_shot in few_shot_flags:
                    run_idx += 1
                    shot_label = "fewshot" if few_shot else "zeroshot"
                    experiment_name = f"{model_alias}_{context_mode}_{shot_label}"

                    logger.info(
                        f"[{run_idx}/{total}] Running: {experiment_name}"
                    )

                    task_fn = self.build_evaluation_task(
                        context_mode, few_shot, model_alias
                    )

                    try:
                        evaluate(
                            dataset=dataset,
                            task=task_fn,
                            scoring_metrics=all_metrics,
                            experiment_name=experiment_name,
                            project_name=self.opik_project,
                            task_threads=1,  # GPU serial execution
                        )
                        logger.info(f"Completed: {experiment_name}")
                    except Exception as e:
                        logger.error(f"Failed: {experiment_name} — {e}")

            self.local_llm.unload_model()
            logger.info(f"Model unloaded: {model_alias}")

        logger.info(f"All {total} experiments complete. View at Opik dashboard.")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(config: ExperimentConfig):
    """Load the FinDER KG Merged parquet dataset."""
    import pandas as pd

    parquet_path = Path(config.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded dataset: {len(df)} rows from {parquet_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Opik-integrated soft-prompt Graph RAG experiment"
    )
    parser.add_argument(
        "--models", nargs="+", default=["llama8b"],
        help="Model aliases from MODEL_REGISTRY (default: llama8b)",
    )
    parser.add_argument(
        "--contexts", nargs="+", default=["none", "lpg", "rdf"],
        help=f"Context conditions (choices: {sorted(VALID_CONTEXTS)})",
    )
    parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Number of questions to evaluate (default: 50)",
    )
    parser.add_argument(
        "--few-shot", action="store_true",
        help="Include few-shot experiments alongside zero-shot",
    )
    parser.add_argument(
        "--judge-model", type=str, default="gpt-4o-mini",
        help="LLM-as-Judge model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Disable LLM-as-Judge metrics (heuristic only)",
    )
    parser.add_argument(
        "--no-bertscore", action="store_true",
        help="Disable BERTScore metric",
    )
    parser.add_argument(
        "--project", type=str, default="FinDER_GraphRAG",
        help="Opik project name (default: FinDER_GraphRAG)",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="FinDER_RAG_v1",
        help="Opik dataset name (default: FinDER_RAG_v1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Validate contexts
    for ctx in args.contexts:
        if ctx not in VALID_CONTEXTS:
            parser.error(f"Invalid context '{ctx}'. Choose from: {sorted(VALID_CONTEXTS)}")

    # Setup logging
    setup_logging(level="INFO")
    logger.info("Opik Graph RAG Experiment starting")

    # Set seed
    set_seed(args.seed)

    # Build config
    config = ExperimentConfig(
        sample_size=args.sample_size,
        model_aliases=args.models,
        context_conditions=args.contexts,
        few_shot=FewShotConfig(enabled=args.few_shot),
        eval_bertscore=not args.no_bertscore,
        eval_rouge=True,
    )

    # Configure Opik for self-hosted
    opik.configure(use_local=True)

    # Load dataset
    df = load_dataset(config)
    if config.sample_size and config.sample_size < len(df):
        df = df.sample(n=config.sample_size, random_state=args.seed).reset_index(drop=True)
        logger.info(f"Sampled {len(df)} questions")

    # Create experiment
    experiment = OpikGraphRAGExperiment(
        config=config,
        opik_project=args.project,
        judge_model=args.judge_model,
        use_judge=not args.no_judge,
        use_bertscore=not args.no_bertscore,
    )

    try:
        experiment.setup(set(args.contexts))
        experiment.run_all(
            df=df,
            models=args.models,
            contexts=args.contexts,
            run_few_shot=args.few_shot,
            dataset_name=args.dataset_name,
        )
    finally:
        experiment.cleanup()


if __name__ == "__main__":
    main()
