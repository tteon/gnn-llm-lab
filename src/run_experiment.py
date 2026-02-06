"""
Local experiment runner for GNN+LLM Graph RAG comparison.

Runs 4 experiments using remote LLM API + local Neo4j + local GNN models:
  [A] LLM Only      — question → LLM API → answer
  [B] Text RAG      — question + references → LLM API → answer
  [C] Graph LPG     — subgraph → GAT → text context → LLM API → answer
  [D] Graph RDF     — triples → TransE → text context → LLM API → answer

Usage:
    uv run python src/run_experiment.py --sample-size 50 --experiments llm text_rag graph_lpg graph_rdf
"""

import argparse
import ast
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.models import MessagePassingGNN, TransEEncoder
from src.utils import (
    ExperimentConfig,
    GraphFormatter,
    Neo4jClient,
    Neo4jConfig,
    setup_logging,
    get_logger,
    set_seed,
)
from src.utils.llm_client import LLMClient, LLMResponse

logger = get_logger("experiment")

VALID_EXPERIMENTS = {"llm", "text_rag", "graph_lpg", "graph_rdf"}


class LocalExperiment:
    """Orchestrates 4-way comparison experiment using remote LLM API."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm: Optional[LLMClient] = None
        self.embedder = None
        self.neo4j_lpg: Optional[Neo4jClient] = None
        self.neo4j_rdf: Optional[Neo4jClient] = None
        self.gnn_lpg: Optional[MessagePassingGNN] = None
        self.kge_rdf: Optional[TransEEncoder] = None
        self.device = torch.device(config.model.device)

        # Entity embedding cache
        self._entity_embeddings: Dict[str, torch.Tensor] = {}
        # RDF index maps (persisted across samples)
        self._node_to_idx: Dict[str, int] = {}
        self._rel_to_idx: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup(self, experiments: set[str]) -> None:
        """Initialize only the components needed for the requested experiments."""
        mc = self.config.model

        # LLM client (always needed)
        logger.info(f"Connecting to LLM API: {mc.llm_api_base_url}")
        self.llm = LLMClient(
            base_url=mc.llm_api_base_url,
            api_key=mc.llm_api_key,
            model=mc.llm_api_model,
        )
        if self.llm.health_check():
            logger.info(f"LLM API OK — model: {mc.llm_api_model}")
        else:
            logger.warning("LLM API health check failed — proceeding anyway")

        # Embedder + GNN models (only for graph experiments)
        needs_graph = experiments & {"graph_lpg", "graph_rdf"}
        if needs_graph:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedder: {mc.embedding_model_id}")
            self.embedder = SentenceTransformer(mc.embedding_model_id)
            logger.info(f"Embedder loaded ({mc.embedding_dim}-dim)")

        if "graph_lpg" in experiments:
            logger.info("Connecting to Neo4j (finderlpg)...")
            self.neo4j_lpg = Neo4jClient(
                Neo4jConfig(database="finderlpg")
            )
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

        if "graph_rdf" in experiments:
            logger.info("Connecting to Neo4j (finderrdf)...")
            self.neo4j_rdf = Neo4jClient(
                Neo4jConfig(database="finderrdf")
            )
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

    def cleanup(self) -> None:
        """Release resources."""
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

    # ------------------------------------------------------------------
    # Individual experiment methods
    # ------------------------------------------------------------------

    def run_llm_only(self, question: str) -> dict:
        """[A] LLM Only — no context."""
        resp = self.llm.generate(
            question,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return {
            "response_llm_only": resp.text,
            "llm_only_time": resp.generation_time,
            "llm_only_input_tokens": resp.input_tokens,
            "llm_only_output_tokens": resp.output_tokens,
        }

    def run_text_rag(self, question: str, references) -> dict:
        """[B] Text RAG — references as context."""
        try:
            parsed = ast.literal_eval(references) if isinstance(references, str) else references
            context = "\n".join(parsed) if isinstance(parsed, list) else str(references)
        except (ValueError, SyntaxError):
            context = str(references) if references else ""

        if not context:
            resp = self.llm.generate(
                question,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
        else:
            resp = self.llm.generate(
                question,
                context=context,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
        return {
            "response_text_rag": resp.text,
            "text_rag_time": resp.generation_time,
            "text_rag_input_tokens": resp.input_tokens,
            "text_rag_output_tokens": resp.output_tokens,
        }

    def run_graph_lpg(self, question: str, question_id: str) -> dict:
        """[C] Graph RAG (LPG) — GAT over subgraph, text context to LLM."""
        subgraph = self.neo4j_lpg.get_subgraph(
            question_id, max_hops=self.config.max_hops
        )
        pyg_data = self._build_lpg_graph(subgraph)

        if pyg_data is None or pyg_data.x.shape[0] == 0:
            resp = self.llm.generate(
                question,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            return {
                "response_graph_lpg": resp.text,
                "graph_lpg_time": resp.generation_time,
                "graph_lpg_input_tokens": resp.input_tokens,
                "graph_lpg_output_tokens": resp.output_tokens,
                "lpg_nodes": 0,
                "lpg_edges": 0,
            }

        pyg_data = pyg_data.to(self.device)
        with torch.no_grad():
            _ = self.gnn_lpg(pyg_data.x, pyg_data.edge_index)

        # Format graph as text context (soft prompt)
        context = GraphFormatter.format(
            nodes=subgraph["nodes"],
            edges=subgraph["edges"],
            style=self.config.soft_prompt_format,
            max_nodes=self.config.max_context_nodes,
            max_edges=self.config.max_context_edges,
        )

        resp = self.llm.generate(
            question,
            context=context,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return {
            "response_graph_lpg": resp.text,
            "graph_lpg_time": resp.generation_time,
            "graph_lpg_input_tokens": resp.input_tokens,
            "graph_lpg_output_tokens": resp.output_tokens,
            "lpg_nodes": int(pyg_data.x.shape[0]),
            "lpg_edges": int(pyg_data.edge_index.shape[1]),
        }

    def run_graph_rdf(self, question: str, question_id: str) -> dict:
        """[D] Graph RAG (RDF) — TransE over triples, text context to LLM."""
        subgraph = self._load_subgraph_rdf(question_id)
        edges = subgraph.get("edges", [])

        if not edges:
            resp = self.llm.generate(
                question,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            return {
                "response_graph_rdf": resp.text,
                "graph_rdf_time": resp.generation_time,
                "graph_rdf_input_tokens": resp.input_tokens,
                "graph_rdf_output_tokens": resp.output_tokens,
                "rdf_nodes": 0,
                "rdf_edges": 0,
            }

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

        if not head_indices:
            resp = self.llm.generate(
                question,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            return {
                "response_graph_rdf": resp.text,
                "graph_rdf_time": resp.generation_time,
                "graph_rdf_input_tokens": resp.input_tokens,
                "graph_rdf_output_tokens": resp.output_tokens,
                "rdf_nodes": 0,
                "rdf_edges": 0,
            }

        with torch.no_grad():
            _ = self.kge_rdf(
                torch.tensor(head_indices, device=self.device),
                torch.tensor(rel_types, device=self.device),
                torch.tensor(tail_indices, device=self.device),
            )

        # Format triples as text context (soft prompt)
        context = "=== [RDF/TransE] Triples ===\n" + "\n".join(
            f"- {d}" for d in edge_descs[:40]
        )

        resp = self.llm.generate(
            question,
            context=context,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return {
            "response_graph_rdf": resp.text,
            "graph_rdf_time": resp.generation_time,
            "graph_rdf_input_tokens": resp.input_tokens,
            "graph_rdf_output_tokens": resp.output_tokens,
            "rdf_nodes": len(set(head_indices + tail_indices)),
            "rdf_edges": len(head_indices),
        }

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run_all(
        self,
        df: pd.DataFrame,
        experiments: set[str],
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run selected experiments on the dataset.

        Args:
            df: DataFrame with columns _id, text, answer, references, category, ...
            experiments: Set of experiment names to run
            sample_size: Limit number of samples (None = all)

        Returns:
            Results DataFrame
        """
        if sample_size:
            df = df.head(sample_size)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.results_dir) / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        results: List[dict] = []
        total = len(df)

        logger.info(f"Running experiments {experiments} on {total} samples")

        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=total, desc="Experiments")):
            question_id = row["_id"]
            question = row["text"]
            ground_truth = row["answer"]
            references = row.get("references", "")
            category = row.get("category", "")

            result = {
                "question_id": question_id,
                "question": question,
                "ground_truth": ground_truth,
                "category": category,
            }

            try:
                if "llm" in experiments:
                    result.update(self.run_llm_only(question))

                if "text_rag" in experiments:
                    result.update(self.run_text_rag(question, references))

                if "graph_lpg" in experiments:
                    result.update(self.run_graph_lpg(question, question_id))

                if "graph_rdf" in experiments:
                    result.update(self.run_graph_rdf(question, question_id))

            except Exception as e:
                logger.error(f"Error at sample {i} ({question_id}): {e}")
                # Keep partial results
                result["error"] = str(e)

            results.append(result)

            # Checkpoint
            if (i + 1) % self.config.checkpoint_interval == 0:
                ckpt_path = results_dir / f"{timestamp}_checkpoint_{i + 1}.csv"
                pd.DataFrame(results).to_csv(ckpt_path, index=False)
                logger.info(f"Checkpoint saved: {ckpt_path} ({i + 1}/{total})")

        # Final save
        results_df = pd.DataFrame(results)
        final_path = results_dir / f"{timestamp}_results_final.csv"
        results_df.to_csv(final_path, index=False)
        logger.info(f"Final results saved: {final_path}")

        # Print summary
        self._print_summary(results_df, experiments)

        return results_df

    def _print_summary(self, df: pd.DataFrame, experiments: set[str]) -> None:
        """Print experiment summary statistics."""
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

        if "graph_lpg" in experiments and "lpg_nodes" in df.columns:
            print(f"\n--- Graph Stats (LPG) ---")
            print(f"  Avg nodes: {df['lpg_nodes'].dropna().mean():.1f}")
            print(f"  Avg edges: {df['lpg_edges'].dropna().mean():.1f}")

        if "graph_rdf" in experiments and "rdf_nodes" in df.columns:
            print(f"\n--- Graph Stats (RDF) ---")
            print(f"  Avg nodes: {df['rdf_nodes'].dropna().mean():.1f}")
            print(f"  Avg edges: {df['rdf_edges'].dropna().mean():.1f}")

        # Exact match / substring match
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
    parser = argparse.ArgumentParser(description="Run GNN+LLM comparison experiments")
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=["llm", "text_rag", "graph_lpg", "graph_rdf"],
        choices=sorted(VALID_EXPERIMENTS),
        help="Which experiments to run",
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

    # Setup logging
    setup_logging(level=args.log_level)

    # Config
    config = ExperimentConfig()
    config.soft_prompt_format = args.format_style
    config.seed = args.seed
    config.sample_size = args.sample_size
    if args.parquet_path:
        config.parquet_path = args.parquet_path
    config.validate()

    # Reproducibility
    set_seed(args.seed)

    # Load data
    df = load_dataset(config)

    # Run
    experiments = set(args.experiments)
    exp = LocalExperiment(config)
    try:
        exp.setup(experiments)
        exp.run_all(df, experiments, sample_size=args.sample_size)
    finally:
        exp.cleanup()


if __name__ == "__main__":
    main()
