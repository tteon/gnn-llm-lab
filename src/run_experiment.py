"""
3-Axis GraphRAG comparison experiment runner.

Runs a full experiment matrix across:
  - Model axis: Dense/Sparse(MoE) models via local HuggingFace loading
  - Context axis: none / LPG / RDF / LPG+RDF
  - Few-shot axis: zero-shot / category-representative few-shot

With attention score extraction and comprehensive evaluation metrics.

Usage:
    # Full matrix
    uv run python src/run_experiment.py \\
        --models llama8b llama70b mixtral qwen_moe \\
        --contexts none lpg rdf lpg_rdf \\
        --few-shot --sample-size 50

    # Quick smoke test
    uv run python src/run_experiment.py \\
        --models llama8b --contexts none lpg --sample-size 2 --no-bertscore

    # Legacy compatibility
    uv run python src/run_experiment.py \\
        --experiments llm text_rag graph_lpg graph_rdf --sample-size 50
"""

import argparse
import ast
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.models import MessagePassingGNN, TransEEncoder
from src.utils import (
    AttentionConfig,
    ExperimentConfig,
    FewShotConfig,
    GraphFormatter,
    Neo4jClient,
    Neo4jConfig,
    setup_logging,
    get_logger,
    set_seed,
)
from src.utils.attention import AttentionExtractor
from src.utils.evaluation import Evaluator
from src.utils.few_shot import FewShotSelector
from src.utils.llm_client import LLMClient, LLMResponse
from src.utils.local_llm import LocalLLMManager, LocalLLMResponse, MODEL_REGISTRY

logger = get_logger("experiment")

# Legacy experiment name → context mapping
LEGACY_EXPERIMENT_MAP = {
    "llm": "none",
    "text_rag": "none",  # text_rag uses references, handled specially
    "graph_lpg": "lpg",
    "graph_rdf": "rdf",
}

VALID_CONTEXTS = {"none", "lpg", "rdf", "lpg_rdf"}


class LocalExperiment:
    """Orchestrates 3-axis comparison experiment with local HF models."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.local_llm: Optional[LocalLLMManager] = None
        self.llm_api: Optional[LLMClient] = None  # Legacy API fallback
        self.embedder = None
        self.neo4j_lpg: Optional[Neo4jClient] = None
        self.neo4j_rdf: Optional[Neo4jClient] = None
        self.gnn_lpg: Optional[MessagePassingGNN] = None
        self.kge_rdf: Optional[TransEEncoder] = None
        self.device = torch.device(config.model.device)
        self.few_shot_selector: Optional[FewShotSelector] = None
        self.evaluator: Optional[Evaluator] = None

        # Entity embedding cache
        self._entity_embeddings: Dict[str, torch.Tensor] = {}
        # RDF index maps
        self._node_to_idx: Dict[str, int] = {}
        self._rel_to_idx: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup(self, contexts: Set[str], use_api: bool = False) -> None:
        """
        Initialize components needed for the requested contexts.

        Args:
            contexts: Set of context modes to prepare for
            use_api: Use remote LLM API instead of local models
        """
        mc = self.config.model

        # LocalLLMManager (models loaded/unloaded per-alias in run_all)
        if not use_api:
            self.local_llm = LocalLLMManager(mc)
            logger.info("LocalLLMManager initialized (models loaded per-alias)")
        else:
            # Legacy API mode
            logger.info(f"Connecting to LLM API: {mc.llm_api_base_url}")
            self.llm_api = LLMClient(
                base_url=mc.llm_api_base_url,
                api_key=mc.llm_api_key,
                model=mc.llm_api_model,
            )
            if self.llm_api.health_check():
                logger.info(f"LLM API OK — model: {mc.llm_api_model}")
            else:
                logger.warning("LLM API health check failed — proceeding anyway")

        # Embedder + GNN models (for graph contexts)
        needs_graph = contexts & {"lpg", "rdf", "lpg_rdf"}
        if needs_graph:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedder: {mc.embedding_model_id}")
            self.embedder = SentenceTransformer(mc.embedding_model_id)
            logger.info(f"Embedder loaded ({mc.embedding_dim}-dim)")

        needs_lpg = contexts & {"lpg", "lpg_rdf"}
        if needs_lpg:
            logger.info("Connecting to Neo4j (finderlpg)...")
            self.neo4j_lpg = Neo4jClient(Neo4jConfig(database="finderlpg"))
            self.neo4j_lpg.connect()
            info = self.neo4j_lpg.get_database_info()
            logger.info(f"finderlpg: {info['node_count']:,} nodes, {info['edge_count']:,} edges")

            self.gnn_lpg = MessagePassingGNN(
                input_dim=mc.embedding_dim,
                hidden_dim=mc.gnn_hidden_dim,
                output_dim=mc.gnn_output_dim,
                num_layers=mc.gnn_num_layers,
                heads=mc.gnn_heads,
                dropout=mc.gnn_dropout,
            ).to(self.device)
            self.gnn_lpg.eval()
            logger.info(
                f"MessagePassingGNN: {sum(p.numel() for p in self.gnn_lpg.parameters()):,} params"
            )

        needs_rdf = contexts & {"rdf", "lpg_rdf"}
        if needs_rdf:
            logger.info("Connecting to Neo4j (finderrdf)...")
            self.neo4j_rdf = Neo4jClient(Neo4jConfig(database="finderrdf"))
            self.neo4j_rdf.connect()
            info = self.neo4j_rdf.get_database_info()
            logger.info(f"finderrdf: {info['node_count']:,} nodes, {info['edge_count']:,} edges")

            num_nodes_rdf = 15000
            num_relations_rdf = 100
            self.kge_rdf = TransEEncoder(
                num_nodes=num_nodes_rdf,
                num_relations=num_relations_rdf,
                hidden_dim=mc.gnn_hidden_dim,
                output_dim=mc.embedding_dim,
            ).to(self.device)
            self.kge_rdf.eval()
            logger.info(
                f"TransEEncoder: {sum(p.numel() for p in self.kge_rdf.parameters()):,} params"
            )

        # Few-shot selector
        if self.config.few_shot.enabled:
            self.few_shot_selector = FewShotSelector(self.config.few_shot)
            if not self.few_shot_selector.load():
                logger.info("Few-shot cache not found, will build during run_all")

        # Evaluator
        self.evaluator = Evaluator(
            use_bertscore=self.config.eval_bertscore,
            use_rouge=self.config.eval_rouge,
        )

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
    # Entity embedding helpers
    # ------------------------------------------------------------------

    def _get_entity_embedding(self, text: str) -> torch.Tensor:
        if text not in self._entity_embeddings:
            self._entity_embeddings[text] = self.embedder.encode(
                text, convert_to_tensor=True
            )
        return self._entity_embeddings[text]

    # ------------------------------------------------------------------
    # Graph building helpers
    # ------------------------------------------------------------------

    def _build_lpg_graph(self, subgraph: dict) -> Optional[Data]:
        """Build a PyG Data object from an LPG subgraph dict."""
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        if not nodes:
            return None

        node_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
        node_texts = [
            f"{n.get('label', '')}: {n.get('name', n['id'])}" for n in nodes
        ]
        x = torch.stack([self._get_entity_embedding(t) for t in node_texts])

        edge_index_list = []
        edge_descs = []
        for e in edges:
            src, tgt = e.get("source"), e.get("target")
            if src in node_to_idx and tgt in node_to_idx:
                edge_index_list.append([node_to_idx[src], node_to_idx[tgt]])
                edge_descs.append(f"{src} --{e.get('type', 'rel')}--> {tgt}")

        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data.node_descs = node_texts
        data.edge_descs = edge_descs
        return data

    def _load_subgraph_rdf(self, question_id: str) -> dict:
        """Load RDF subgraph from finderrdf database."""
        query = """
        MATCH (r:Resource)-[rel]-(r2:Resource)
        WITH r, r2, rel LIMIT 100
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

    def _process_rdf_triples(self, edges: list) -> tuple:
        """Process RDF edges into TransE inputs and descriptions."""
        head_indices, rel_types, tail_indices, edge_descs = [], [], [], []
        for e in edges:
            src = e.get("source")
            tgt = e.get("target")
            rel = e.get("type") or "related"
            if src and tgt:
                if src not in self._node_to_idx:
                    self._node_to_idx[src] = len(self._node_to_idx) % self.kge_rdf.transe.num_nodes
                if tgt not in self._node_to_idx:
                    self._node_to_idx[tgt] = len(self._node_to_idx) % self.kge_rdf.transe.num_nodes
                if rel not in self._rel_to_idx:
                    self._rel_to_idx[rel] = len(self._rel_to_idx) % self.kge_rdf.transe.num_relations
                head_indices.append(self._node_to_idx[src])
                rel_types.append(self._rel_to_idx[rel])
                tail_indices.append(self._node_to_idx[tgt])
                edge_descs.append(f"{src} --{rel}--> {tgt}")
        return head_indices, rel_types, tail_indices, edge_descs

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_context(
        self, context_mode: str, question_id: str, references: str = ""
    ) -> tuple:
        """
        Build context string for a given mode.

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

        lpg_context = None
        lpg_subgraph = None
        rdf_context = None
        rdf_subgraph = None

        # LPG processing
        if context_mode in ("lpg", "lpg_rdf") and self.neo4j_lpg:
            lpg_subgraph = self.neo4j_lpg.get_subgraph(
                question_id, max_hops=self.config.max_hops
            )
            pyg_data = self._build_lpg_graph(lpg_subgraph)
            if pyg_data is not None and pyg_data.x.shape[0] > 0:
                pyg_data = pyg_data.to(self.device)
                with torch.no_grad():
                    _ = self.gnn_lpg(pyg_data.x, pyg_data.edge_index)
                info["lpg_nodes"] = int(pyg_data.x.shape[0])
                info["lpg_edges"] = int(pyg_data.edge_index.shape[1])
                entity_names.extend(
                    n.get("name", n.get("id", ""))
                    for n in lpg_subgraph.get("nodes", [])
                )

        # RDF processing
        if context_mode in ("rdf", "lpg_rdf") and self.neo4j_rdf:
            rdf_subgraph = self._load_subgraph_rdf(question_id)
            edges = rdf_subgraph.get("edges", [])
            if edges:
                head_idx, rel_types, tail_idx, edge_descs = self._process_rdf_triples(edges)
                if head_idx:
                    with torch.no_grad():
                        _ = self.kge_rdf(
                            torch.tensor(head_idx, device=self.device),
                            torch.tensor(rel_types, device=self.device),
                            torch.tensor(tail_idx, device=self.device),
                        )
                    info["rdf_nodes"] = len(set(head_idx + tail_idx))
                    info["rdf_edges"] = len(head_idx)

        # Format context
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
                rdf_context = "=== [RDF/TransE] Triples ===\n" + "\n".join(
                    f"- {e.get('source', '?')} --{e.get('type', 'rel')}--> {e.get('target', '?')}"
                    for e in rdf_edges[:40]
                )
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
    # Unified condition runner
    # ------------------------------------------------------------------

    def run_condition(
        self,
        question: str,
        question_id: str,
        category: str,
        references: str,
        context_mode: str,
        few_shot: bool,
        model_alias: str,
    ) -> dict:
        """
        Run a single experimental condition.

        Args:
            question: Question text
            question_id: Question identifier
            category: Question category
            references: Text references (for text_rag fallback)
            context_mode: "none" | "lpg" | "rdf" | "lpg_rdf"
            few_shot: Whether to include few-shot examples
            model_alias: Current model alias

        Returns:
            Dict with response, timing, subgraph info
        """
        # Build context
        context, subgraph_info, entity_names = self._build_context(
            context_mode, question_id, references
        )

        # Build few-shot examples
        few_shot_examples = None
        if few_shot and self.few_shot_selector:
            examples = self.few_shot_selector.get_examples(
                category, exclude_question_id=question_id
            )
            if examples:
                few_shot_examples = self.few_shot_selector.format_for_prompt(examples)

        # Generate
        extract_attention = (
            self.config.attention.enabled
            and context is not None
            and self.local_llm is not None
        )

        result = {
            "model": model_alias,
            "context_mode": context_mode,
            "few_shot": few_shot,
            **subgraph_info,
        }

        if self.local_llm and self.local_llm.model is not None:
            resp = self.local_llm.generate(
                question=question,
                context=context,
                few_shot_examples=few_shot_examples,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                extract_attention=extract_attention,
                attention_config=self.config.attention if extract_attention else None,
                entity_names=entity_names if extract_attention else None,
            )
            result["response"] = resp.text
            result["generation_time"] = resp.generation_time
            result["input_tokens"] = resp.input_tokens
            result["output_tokens"] = resp.output_tokens

            # Save attention data
            if resp.attention_data and self.config.attention.enabled:
                self._save_attention_result(
                    resp.attention_data, model_alias, context_mode,
                    few_shot, question_id,
                )

        elif self.llm_api:
            # Legacy API path
            resp = self.llm_api.generate(
                question=question,
                context=context,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            result["response"] = resp.text
            result["generation_time"] = resp.generation_time
            result["input_tokens"] = resp.input_tokens
            result["output_tokens"] = resp.output_tokens

        return result

    def _save_attention_result(
        self, attention_data, model_alias: str, context_mode: str,
        few_shot: bool, question_id: str,
    ) -> None:
        """Save attention result to disk."""
        timestamp = getattr(self, "_run_timestamp", "unknown")
        fs_label = "fewshot" if few_shot else "zeroshot"
        attn_dir = Path(self.config.attention.output_dir) / timestamp / model_alias
        attn_dir = attn_dir / f"{context_mode}_{fs_label}"
        save_path = attn_dir / f"{question_id}.npz"
        try:
            AttentionExtractor.save_attention(attention_data, str(save_path))
        except Exception as e:
            logger.warning(f"Failed to save attention for {question_id}: {e}")

    # ------------------------------------------------------------------
    # Main runner: 3-axis matrix
    # ------------------------------------------------------------------

    def run_all(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run the full 3-axis experiment matrix.

        Outer loop: models (load/unload)
        Middle loop: samples
        Inner loops: contexts × few-shot

        Args:
            df: DataFrame with _id, text, answer, references, category
            sample_size: Limit samples (None = all)

        Returns:
            Results DataFrame
        """
        if sample_size:
            df = df.head(sample_size)

        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.results_dir) / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Build few-shot examples if needed and not cached
        if self.config.few_shot.enabled and self.few_shot_selector and not self.few_shot_selector._loaded:
            logger.info("Building few-shot examples from dataset...")
            self.few_shot_selector.build_examples(df, self.embedder)
            self.few_shot_selector.save()

        # Determine few-shot conditions
        few_shot_conditions = [False]
        if self.config.few_shot.enabled:
            few_shot_conditions.append(True)

        results: List[dict] = []
        total_conditions = (
            len(self.config.model_aliases)
            * len(df)
            * len(self.config.context_conditions)
            * len(few_shot_conditions)
        )
        logger.info(
            f"Experiment matrix: {len(self.config.model_aliases)} models × "
            f"{len(df)} samples × {len(self.config.context_conditions)} contexts × "
            f"{len(few_shot_conditions)} few-shot = {total_conditions} conditions"
        )

        for model_alias in self.config.model_aliases:
            logger.info(f"\n{'='*60}")
            logger.info(f"MODEL: {model_alias}")
            logger.info(f"{'='*60}")

            # Load model
            if self.local_llm:
                self.local_llm.load_model(
                    model_alias,
                    attention_config=self.config.attention if self.config.attention.enabled else None,
                )

            model_results: List[dict] = []

            for i, (_, row) in enumerate(tqdm(
                df.iterrows(), total=len(df),
                desc=f"{model_alias}"
            )):
                question_id = row["_id"]
                question = row["text"]
                ground_truth = row["answer"]
                references = row.get("references", "")
                category = row.get("category", "")

                for context_mode in self.config.context_conditions:
                    for few_shot in few_shot_conditions:
                        try:
                            result = self.run_condition(
                                question=question,
                                question_id=question_id,
                                category=category,
                                references=references,
                                context_mode=context_mode,
                                few_shot=few_shot,
                                model_alias=model_alias,
                            )
                            result["question_id"] = question_id
                            result["question"] = question
                            result["ground_truth"] = str(ground_truth)
                            result["category"] = category
                            result["error"] = ""
                        except Exception as e:
                            logger.error(
                                f"Error: {model_alias}/{context_mode}/"
                                f"{'fs' if few_shot else 'zs'} sample {i} ({question_id}): {e}"
                            )
                            result = {
                                "question_id": question_id,
                                "question": question,
                                "ground_truth": str(ground_truth),
                                "category": category,
                                "model": model_alias,
                                "context_mode": context_mode,
                                "few_shot": few_shot,
                                "response": "",
                                "generation_time": 0.0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "lpg_nodes": 0,
                                "lpg_edges": 0,
                                "rdf_nodes": 0,
                                "rdf_edges": 0,
                                "error": str(e),
                            }

                        model_results.append(result)

                # Checkpoint per model
                if (i + 1) % self.config.checkpoint_interval == 0:
                    ckpt_path = results_dir / f"{self._run_timestamp}_checkpoint_{model_alias}_{i+1}.csv"
                    pd.DataFrame(model_results).to_csv(ckpt_path, index=False)
                    logger.info(f"Checkpoint: {ckpt_path} ({i+1}/{len(df)})")

                # Memory cleanup between samples
                torch.cuda.empty_cache()

            results.extend(model_results)

            # Unload model before loading next
            if self.local_llm:
                self.local_llm.unload_model()

        # Final results
        results_df = pd.DataFrame(results)
        final_path = results_dir / f"{self._run_timestamp}_results_final.csv"
        results_df.to_csv(final_path, index=False)
        logger.info(f"Final results saved: {final_path}")

        # Evaluate and summarize
        self._evaluate_and_summarize(results_df, results_dir)

        return results_df

    # ------------------------------------------------------------------
    # Legacy compatibility: run with old-style experiments
    # ------------------------------------------------------------------

    def run_legacy(
        self,
        df: pd.DataFrame,
        experiments: Set[str],
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Legacy runner: 4-experiment style (API-based).
        Maps old experiment names to context modes and runs via run_condition.
        """
        if sample_size:
            df = df.head(sample_size)

        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.results_dir) / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        results: List[dict] = []
        total = len(df)

        logger.info(f"Running legacy experiments {experiments} on {total} samples")

        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=total, desc="Experiments")):
            question_id = row["_id"]
            question = row["text"]
            ground_truth = row["answer"]
            references = row.get("references", "")
            category = row.get("category", "")

            base_result = {
                "question_id": question_id,
                "question": question,
                "ground_truth": ground_truth,
                "category": category,
            }

            try:
                if "llm" in experiments:
                    r = self.run_condition(
                        question, question_id, category, references,
                        context_mode="none", few_shot=False,
                        model_alias=self.config.model_aliases[0],
                    )
                    base_result.update({
                        "response_llm_only": r.get("response", ""),
                        "llm_only_time": r.get("generation_time", 0),
                        "llm_only_input_tokens": r.get("input_tokens", 0),
                        "llm_only_output_tokens": r.get("output_tokens", 0),
                    })

                if "text_rag" in experiments:
                    # Text RAG: parse references as context
                    try:
                        parsed = ast.literal_eval(references) if isinstance(references, str) else references
                        context = "\n".join(parsed) if isinstance(parsed, list) else str(references)
                    except (ValueError, SyntaxError):
                        context = str(references) if references else ""

                    if self.llm_api:
                        resp = self.llm_api.generate(
                            question, context=context if context else None,
                            max_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                        )
                        base_result.update({
                            "response_text_rag": resp.text,
                            "text_rag_time": resp.generation_time,
                            "text_rag_input_tokens": resp.input_tokens,
                            "text_rag_output_tokens": resp.output_tokens,
                        })

                if "graph_lpg" in experiments:
                    r = self.run_condition(
                        question, question_id, category, references,
                        context_mode="lpg", few_shot=False,
                        model_alias=self.config.model_aliases[0],
                    )
                    base_result.update({
                        "response_graph_lpg": r.get("response", ""),
                        "graph_lpg_time": r.get("generation_time", 0),
                        "graph_lpg_input_tokens": r.get("input_tokens", 0),
                        "graph_lpg_output_tokens": r.get("output_tokens", 0),
                        "lpg_nodes": r.get("lpg_nodes", 0),
                        "lpg_edges": r.get("lpg_edges", 0),
                    })

                if "graph_rdf" in experiments:
                    r = self.run_condition(
                        question, question_id, category, references,
                        context_mode="rdf", few_shot=False,
                        model_alias=self.config.model_aliases[0],
                    )
                    base_result.update({
                        "response_graph_rdf": r.get("response", ""),
                        "graph_rdf_time": r.get("generation_time", 0),
                        "graph_rdf_input_tokens": r.get("input_tokens", 0),
                        "graph_rdf_output_tokens": r.get("output_tokens", 0),
                        "rdf_nodes": r.get("rdf_nodes", 0),
                        "rdf_edges": r.get("rdf_edges", 0),
                    })

            except Exception as e:
                logger.error(f"Error at sample {i} ({question_id}): {e}")
                base_result["error"] = str(e)

            results.append(base_result)

            if (i + 1) % self.config.checkpoint_interval == 0:
                ckpt_path = results_dir / f"{self._run_timestamp}_checkpoint_{i+1}.csv"
                pd.DataFrame(results).to_csv(ckpt_path, index=False)
                logger.info(f"Checkpoint: {ckpt_path} ({i+1}/{total})")

        results_df = pd.DataFrame(results)
        final_path = results_dir / f"{self._run_timestamp}_results_final.csv"
        results_df.to_csv(final_path, index=False)
        logger.info(f"Final results saved: {final_path}")

        self._print_legacy_summary(results_df, experiments)
        return results_df

    # ------------------------------------------------------------------
    # Evaluation and summary
    # ------------------------------------------------------------------

    def _evaluate_and_summarize(
        self, df: pd.DataFrame, results_dir: Path
    ) -> None:
        """Evaluate all conditions and print/save summary."""
        if self.evaluator is None or "response" not in df.columns:
            return

        print("\n" + "=" * 80)
        print("3-AXIS EXPERIMENT RESULTS")
        print("=" * 80)
        print(f"Total conditions: {len(df)}")
        print(f"Models: {df['model'].unique().tolist()}")
        print(f"Contexts: {df['context_mode'].unique().tolist()}")
        print(f"Few-shot: {df['few_shot'].unique().tolist()}")

        # Group by (model, context_mode, few_shot)
        summary_rows = []
        groups = df.groupby(["model", "context_mode", "few_shot"])

        for (model, ctx, fs), group in groups:
            valid = group.dropna(subset=["response", "ground_truth"])
            valid = valid[valid["response"] != ""]

            if len(valid) == 0:
                continue

            predictions = valid["response"].tolist()
            references = valid["ground_truth"].tolist()

            metrics = self.evaluator.evaluate_batch(predictions, references)

            row = {
                "model": model,
                "context_mode": ctx,
                "few_shot": fs,
                "n_samples": len(valid),
                "avg_gen_time": valid["generation_time"].mean(),
                "avg_input_tokens": valid["input_tokens"].mean(),
                "avg_output_tokens": valid["output_tokens"].mean(),
                **metrics,
            }

            if "lpg_nodes" in valid.columns:
                row["avg_lpg_nodes"] = valid["lpg_nodes"].mean()
                row["avg_lpg_edges"] = valid["lpg_edges"].mean()
            if "rdf_nodes" in valid.columns:
                row["avg_rdf_nodes"] = valid["rdf_nodes"].mean()
                row["avg_rdf_edges"] = valid["rdf_edges"].mean()

            summary_rows.append(row)

            fs_label = "few-shot" if fs else "zero-shot"
            print(f"\n--- {model} | {ctx} | {fs_label} (n={len(valid)}) ---")
            print(f"  EM: {metrics.get('exact_match', 0):.1%}")
            print(f"  Substring: {metrics.get('substring_match', 0):.1%}")
            print(f"  Token F1: {metrics.get('token_f1', 0):.3f}")
            if "rouge_l" in metrics:
                print(f"  ROUGE-L: {metrics.get('rouge_l', 0):.3f}")
            if "bert_score_f1" in metrics:
                print(f"  BERTScore: {metrics.get('bert_score_f1', 0):.3f}")
            print(f"  Avg time: {valid['generation_time'].mean():.2f}s")

        # Save summary
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = results_dir / f"{self._run_timestamp}_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary saved: {summary_path}")

            print(f"\n{'='*80}")
            print(f"Summary saved: {summary_path}")

        # Save metadata
        meta = {
            "timestamp": self._run_timestamp,
            "config": self.config.to_dict(),
            "total_samples": len(df),
            "models": self.config.model_aliases,
            "contexts": self.config.context_conditions,
            "few_shot_enabled": self.config.few_shot.enabled,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        meta_path = results_dir / f"{self._run_timestamp}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info(f"Metadata saved: {meta_path}")

        print("=" * 80)

    def _print_legacy_summary(self, df: pd.DataFrame, experiments: Set[str]) -> None:
        """Print legacy-format experiment summary."""
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"Total samples: {len(df)}")

        if "category" in df.columns:
            print(f"\nCategories:\n{df['category'].value_counts().to_string()}")

        print("\n--- Response Lengths (avg chars) ---")
        for exp, col in [
            ("llm", "response_llm_only"),
            ("text_rag", "response_text_rag"),
            ("graph_lpg", "response_graph_lpg"),
            ("graph_rdf", "response_graph_rdf"),
        ]:
            if exp in experiments and col in df.columns:
                avg_len = df[col].dropna().str.len().mean()
                print(f"  {exp}: {avg_len:.0f}")

        print("\n--- Generation Time (avg sec) ---")
        for exp, col in [
            ("llm", "llm_only_time"),
            ("text_rag", "text_rag_time"),
            ("graph_lpg", "graph_lpg_time"),
            ("graph_rdf", "graph_rdf_time"),
        ]:
            if exp in experiments and col in df.columns:
                avg_time = df[col].dropna().mean()
                print(f"  {exp}: {avg_time:.2f}s")

        print("\n--- Token Usage (avg) ---")
        for exp, in_col, out_col in [
            ("llm", "llm_only_input_tokens", "llm_only_output_tokens"),
            ("text_rag", "text_rag_input_tokens", "text_rag_output_tokens"),
            ("graph_lpg", "graph_lpg_input_tokens", "graph_lpg_output_tokens"),
            ("graph_rdf", "graph_rdf_input_tokens", "graph_rdf_output_tokens"),
        ]:
            if exp in experiments and in_col in df.columns:
                avg_in = df[in_col].dropna().mean()
                avg_out = df[out_col].dropna().mean()
                print(f"  {exp}: in={avg_in:.0f}, out={avg_out:.0f}")

        if "ground_truth" in df.columns:
            print("\n--- Accuracy (simple) ---")
            for exp, col in [
                ("llm", "response_llm_only"),
                ("text_rag", "response_text_rag"),
                ("graph_lpg", "response_graph_lpg"),
                ("graph_rdf", "response_graph_rdf"),
            ]:
                if exp in experiments and col in df.columns:
                    valid = df.dropna(subset=[col, "ground_truth"])
                    if len(valid) == 0:
                        continue
                    exact = (
                        valid[col].str.strip().str.lower()
                        == valid["ground_truth"].str.strip().str.lower()
                    ).mean()
                    substr = valid.apply(
                        lambda r: str(r["ground_truth"]).strip().lower()
                        in str(r[col]).strip().lower(),
                        axis=1,
                    ).mean()
                    print(f"  {exp}: exact={exact:.1%}, substr_match={substr:.1%}")

        print("=" * 70)


def load_dataset(config: ExperimentConfig) -> pd.DataFrame:
    """Load the FinDER KG Merged parquet dataset."""
    parquet_path = Path(config.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded dataset: {len(df)} rows from {parquet_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run 3-axis GNN+LLM comparison experiments"
    )

    # New 3-axis arguments
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model aliases to evaluate (default: llama8b)",
    )
    parser.add_argument(
        "--contexts", nargs="+", default=None,
        choices=sorted(VALID_CONTEXTS),
        help="Context conditions (default: none lpg rdf lpg_rdf)",
    )
    parser.add_argument(
        "--few-shot", action="store_true", default=False,
        help="Enable few-shot examples",
    )
    parser.add_argument(
        "--few-shot-n", type=int, default=1,
        help="Number of few-shot examples per category",
    )
    parser.add_argument(
        "--extract-attention", action="store_true", default=False,
        help="Extract attention scores",
    )
    parser.add_argument(
        "--attention-layers", nargs="+", type=int, default=None,
        help="Layer indices for attention extraction (e.g. -1 -2 -3)",
    )
    parser.add_argument(
        "--no-bertscore", action="store_true", default=False,
        help="Disable BERTScore evaluation",
    )
    parser.add_argument(
        "--no-rouge", action="store_true", default=False,
        help="Disable ROUGE evaluation",
    )

    # Legacy arguments
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        choices=sorted({"llm", "text_rag", "graph_lpg", "graph_rdf"}),
        help="(Legacy) Which experiments to run",
    )

    # Common arguments
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--format-style", default="structured",
        choices=["structured", "natural", "triple", "csv"],
        help="Graph context formatting style",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--parquet-path", type=str, default=None,
        help="Override path to FinDER_KG_Merged.parquet",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    # Determine run mode
    legacy_mode = args.experiments is not None and args.models is None

    # Setup logging
    setup_logging(level=args.log_level)

    # Config
    config = ExperimentConfig()
    config.soft_prompt_format = args.format_style
    config.seed = args.seed
    config.sample_size = args.sample_size
    config.eval_bertscore = not args.no_bertscore
    config.eval_rouge = not args.no_rouge

    if args.parquet_path:
        config.parquet_path = args.parquet_path

    # 3-axis config
    if args.models:
        config.model_aliases = args.models
    elif not legacy_mode:
        config.model_aliases = ["llama8b"]

    if args.contexts:
        config.context_conditions = args.contexts
    elif not legacy_mode:
        config.context_conditions = ["none", "lpg", "rdf", "lpg_rdf"]

    # Few-shot config
    config.few_shot.enabled = args.few_shot
    config.few_shot.num_examples_per_category = args.few_shot_n

    # Attention config
    config.attention.enabled = args.extract_attention
    if args.attention_layers:
        config.attention.layers_to_extract = args.attention_layers

    config.validate()

    # Reproducibility
    set_seed(args.seed)

    # Load data
    df = load_dataset(config)

    # Run
    exp = LocalExperiment(config)
    try:
        if legacy_mode:
            # Legacy API mode
            experiments = set(args.experiments)
            contexts_needed = set()
            for e in experiments:
                ctx = LEGACY_EXPERIMENT_MAP.get(e, "none")
                if ctx != "none":
                    contexts_needed.add(ctx)
            exp.setup(contexts_needed, use_api=True)
            exp.run_legacy(df, experiments, sample_size=args.sample_size)
        else:
            # New 3-axis mode
            contexts = set(config.context_conditions)
            exp.setup(contexts, use_api=False)
            exp.run_all(df, sample_size=args.sample_size)
    finally:
        exp.cleanup()


if __name__ == "__main__":
    main()
