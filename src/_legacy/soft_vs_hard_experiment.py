"""
Soft Prompt vs Hard Prompt (GNN+LLM) Comparison Experiment

This experiment compares two approaches for injecting graph information into LLMs:

1. SOFT PROMPT (Text-based Graph RAG):
   - Graph → Text conversion (nodes + edges as structured text)
   - Injected into LLM's context window
   - LLM processes graph info through standard attention

2. HARD PROMPT (GNN+LLM):
   - Graph → GNN encoding → Graph embedding
   - Embedding projected into LLM's hidden space
   - Structure directly encoded, not verbalized

Key differences:
- Soft: Relies on LLM's text understanding of graph structure
- Hard: GNN explicitly captures structural patterns (neighbors, paths, etc.)
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from neo4j import GraphDatabase
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool


class PromptType(Enum):
    """Types of prompts for experiment"""
    LLM_ONLY = "llm_only"           # No graph context (baseline)
    SOFT_PROMPT = "soft_prompt"     # Graph as text in context
    HARD_PROMPT = "hard_prompt"     # Graph encoded via GNN


@dataclass
class ExperimentConfig:
    """Configuration for soft vs hard prompt experiment"""
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "finderlpg"

    # Models
    llm_model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # GNN settings (for hard prompt)
    gnn_hidden_dim: int = 256
    gnn_output_dim: int = 384  # Match embedding dim
    gnn_num_layers: int = 2
    gnn_heads: int = 4
    gnn_dropout: float = 0.1

    # Retrieval
    top_k_nodes: int = 20
    max_hops: int = 2
    max_context_nodes: int = 30
    max_context_edges: int = 50

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.0  # Deterministic for fair comparison

    # Soft prompt formatting options
    soft_prompt_format: str = "structured"  # "structured", "natural", "triple"
    include_node_properties: bool = True
    include_edge_types: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit: bool = False  # Quantization for memory efficiency


# =============================================================================
# Graph Encoder (for Hard Prompt)
# =============================================================================

class GNNEncoder(nn.Module):
    """
    GNN encoder that produces graph-level embeddings
    for injection into LLM hidden states.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else hidden_dim * heads
            self.convs.append(GATConv(in_ch, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        self.output_proj = nn.Linear(hidden_dim * heads, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for multiple graphs

        Returns:
            Graph-level embedding [batch_size, output_dim] or [output_dim] if single graph
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = torch.relu(x)

        # Message passing layers
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
            # Skip connection if dimensions match
            if x_res.shape == x.shape:
                x = x + x_res

        # Global pooling
        if batch is not None:
            graph_emb = global_mean_pool(x, batch)
        else:
            graph_emb = x.mean(dim=0, keepdim=True)

        return self.output_proj(graph_emb)


# =============================================================================
# Projection Layer (GNN → LLM hidden space)
# =============================================================================

class GraphProjector(nn.Module):
    """
    Projects GNN embeddings into LLM's hidden dimension.
    Used for hard prompt injection.
    """

    def __init__(self, graph_dim: int, llm_dim: int, num_tokens: int = 8):
        """
        Args:
            graph_dim: Dimension of GNN output
            llm_dim: Dimension of LLM hidden states
            num_tokens: Number of virtual tokens to create from graph
        """
        super().__init__()
        self.num_tokens = num_tokens

        self.projector = nn.Sequential(
            nn.Linear(graph_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim * num_tokens)
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_emb: [batch_size, graph_dim]

        Returns:
            Virtual tokens: [batch_size, num_tokens, llm_dim]
        """
        projected = self.projector(graph_emb)
        batch_size = graph_emb.shape[0]
        return projected.view(batch_size, self.num_tokens, -1)


# =============================================================================
# Soft Prompt Formatter
# =============================================================================

class SoftPromptFormatter:
    """
    Converts graph data to text for soft prompt injection.
    Multiple formatting strategies available.
    """

    @staticmethod
    def format_structured(nodes: List[Dict], edges: List[Dict],
                         include_props: bool = True) -> str:
        """
        Structured format with clear sections.

        Example:
        === ENTITIES ===
        [Company] Apple Inc: technology company founded in 1976
        [Person] Tim Cook: CEO of Apple

        === RELATIONSHIPS ===
        Tim Cook --[CEO_OF]--> Apple Inc
        """
        parts = []

        # Entities section
        parts.append("=== ENTITIES ===")
        for n in nodes[:30]:  # Limit for context window
            label = n.get('label', 'Entity')
            name = n.get('name', n.get('id', 'Unknown'))
            desc = ""
            if include_props:
                props = n.get('properties', {})
                prop_items = [f"{k}={v}" for k, v in props.items()
                             if k not in ['id', 'label', 'question_ids', 'embedding'] and v]
                if prop_items:
                    desc = f": {', '.join(prop_items[:3])}"
            parts.append(f"[{label}] {name}{desc}")

        # Relationships section
        parts.append("\n=== RELATIONSHIPS ===")
        for e in edges[:50]:  # Limit
            src = e.get('source', e.get('src', '?'))
            tgt = e.get('target', e.get('tgt', '?'))
            rel = e.get('type', e.get('relType', 'RELATED'))
            parts.append(f"{src} --[{rel}]--> {tgt}")

        return "\n".join(parts)

    @staticmethod
    def format_natural(nodes: List[Dict], edges: List[Dict]) -> str:
        """
        Natural language format.

        Example:
        The knowledge graph contains the following information:
        Apple Inc is a technology company. Tim Cook is the CEO of Apple Inc.
        """
        sentences = ["The knowledge graph contains the following information:"]

        # Create node descriptions
        node_map = {n.get('id', n.get('name')): n for n in nodes}

        for e in edges[:30]:
            src_id = e.get('source', e.get('src'))
            tgt_id = e.get('target', e.get('tgt'))
            rel = e.get('type', e.get('relType', 'is related to'))

            src_name = node_map.get(src_id, {}).get('name', src_id)
            tgt_name = node_map.get(tgt_id, {}).get('name', tgt_id)

            # Convert relationship type to natural language
            rel_natural = rel.lower().replace('_', ' ')
            sentences.append(f"{src_name} {rel_natural} {tgt_name}.")

        return " ".join(sentences)

    @staticmethod
    def format_triples(nodes: List[Dict], edges: List[Dict]) -> str:
        """
        Simple triple format (subject, predicate, object).

        Example:
        (Apple Inc, founded_in, 1976)
        (Tim Cook, CEO_OF, Apple Inc)
        """
        lines = ["Knowledge triples:"]

        for e in edges[:50]:
            src = e.get('source', e.get('src', '?'))
            tgt = e.get('target', e.get('tgt', '?'))
            rel = e.get('type', e.get('relType', 'RELATED'))
            lines.append(f"({src}, {rel}, {tgt})")

        return "\n".join(lines)

    @classmethod
    def format(cls, nodes: List[Dict], edges: List[Dict],
               style: str = "structured", **kwargs) -> str:
        """Main entry point for formatting."""
        formatters = {
            "structured": cls.format_structured,
            "natural": cls.format_natural,
            "triple": cls.format_triples
        }
        formatter = formatters.get(style, cls.format_structured)
        return formatter(nodes, edges, **kwargs) if style == "structured" else formatter(nodes, edges)


# =============================================================================
# Data Loader
# =============================================================================

class GraphDataLoader:
    """Load graph data from Neo4j"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )

    def verify_connection(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def load_questions(self, limit: int = None) -> pd.DataFrame:
        """Load questions from database"""
        query = """
        MATCH (q:Question)
        RETURN q.id as id, q.text as text, q.answer as answer,
               q.category as category, q.type as type
        """
        if limit:
            query += f" LIMIT {limit}"

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query)
            data = [dict(r) for r in result]

        return pd.DataFrame(data)

    def get_subgraph(self, question_id: str, max_hops: int = 2) -> Dict:
        """Get subgraph related to a question"""
        query = """
        MATCH (e:Entity)
        WHERE $qid IN e.question_ids
        WITH collect(e) as seeds
        UNWIND seeds as seed
        OPTIONAL MATCH path = (seed)-[*1..%d]-(connected:Entity)
        WITH seeds, collect(DISTINCT connected) as connected_nodes
        WITH [n IN seeds + connected_nodes WHERE n IS NOT NULL] as all_nodes
        UNWIND all_nodes as n
        WITH DISTINCT n
        OPTIONAL MATCH (n)-[r]->(m:Entity)
        WHERE m IN all_nodes
        RETURN collect(DISTINCT {
            id: n.id,
            label: n.label,
            name: coalesce(n.name, n.id),
            properties: properties(n)
        }) as nodes,
        collect(DISTINCT {
            source: startNode(r).id,
            target: endNode(r).id,
            type: type(r)
        }) as edges
        """ % max_hops

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, qid=question_id)
            record = result.single()

        if record:
            return {
                "nodes": [n for n in record["nodes"] if n],
                "edges": [e for e in record["edges"] if e.get("source") and e.get("target")]
            }
        return {"nodes": [], "edges": []}

    def close(self):
        self.driver.close()


# =============================================================================
# Main Experiment Class
# =============================================================================

class SoftVsHardExperiment:
    """
    Main experiment class comparing soft prompt and hard prompt approaches.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Components (initialized lazily)
        self.data_loader: Optional[GraphDataLoader] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.llm = None
        self.tokenizer = None
        self.gnn: Optional[GNNEncoder] = None
        self.projector: Optional[GraphProjector] = None

        # Results storage
        self.results: List[Dict] = []

    def setup(self, components: List[str] = ["neo4j", "embeddings", "llm", "gnn"]):
        """Initialize specified components"""
        if "neo4j" in components:
            self._setup_neo4j()
        if "embeddings" in components:
            self._setup_embeddings()
        if "llm" in components:
            self._setup_llm()
        if "gnn" in components:
            self._setup_gnn()

    def _setup_neo4j(self):
        print("Setting up Neo4j connection...")
        self.data_loader = GraphDataLoader(self.config)
        if not self.data_loader.verify_connection():
            raise ConnectionError("Failed to connect to Neo4j")
        print("Neo4j connected.")

    def _setup_embeddings(self):
        print(f"Loading embedding model: {self.config.embedding_model_id}")
        self.embedding_model = SentenceTransformer(
            self.config.embedding_model_id,
            device=self.config.device
        )

        # Try to load cached embeddings
        cache_path = "entity_embeddings_cache.npz"
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self.entity_embeddings = dict(data["embeddings"].item())
        print(f"Embedding model ready. Cached entities: {len(self.entity_embeddings)}")

    def _setup_llm(self):
        print(f"Loading LLM: {self.config.llm_model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config if needed
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        self.llm.eval()
        print(f"LLM loaded on {next(self.llm.parameters()).device}")

    def _setup_gnn(self):
        print("Setting up GNN encoder...")
        self.gnn = GNNEncoder(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.gnn_hidden_dim,
            output_dim=self.config.gnn_output_dim,
            num_layers=self.config.gnn_num_layers,
            heads=self.config.gnn_heads,
            dropout=self.config.gnn_dropout
        ).to(self.device)

        # Note: projector requires LLM hidden dim
        # Will be initialized when LLM is loaded
        if self.llm is not None:
            llm_dim = self.llm.config.hidden_size
            self.projector = GraphProjector(
                graph_dim=self.config.gnn_output_dim,
                llm_dim=llm_dim,
                num_tokens=8
            ).to(self.device)

        print(f"GNN parameters: {sum(p.numel() for p in self.gnn.parameters()):,}")

    def _embed_nodes(self, nodes: List[Dict]) -> torch.Tensor:
        """Get embeddings for nodes, generating if needed"""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        for i, n in enumerate(nodes):
            node_id = n.get('id', n.get('name', str(i)))

            if node_id in self.entity_embeddings:
                embeddings.append(self.entity_embeddings[node_id])
            else:
                # Create text representation for embedding
                label = n.get('label', 'Entity')
                name = n.get('name', node_id)
                props = n.get('properties', {})
                prop_text = ", ".join(
                    f"{k}: {v}" for k, v in props.items()
                    if k not in ['id', 'label', 'question_ids', 'embedding'] and v
                )
                text = f"{label}: {name}. {prop_text}"
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                embeddings.append(None)  # Placeholder

        # Batch embed missing nodes
        if texts_to_embed:
            new_embeddings = self.embedding_model.encode(
                texts_to_embed,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            for idx, emb in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = emb
                # Cache it
                node_id = nodes[idx].get('id', nodes[idx].get('name', str(idx)))
                self.entity_embeddings[node_id] = emb

        return torch.tensor(np.stack(embeddings), dtype=torch.float32)

    def _build_pyg_data(self, subgraph: Dict) -> Optional[Data]:
        """Convert subgraph to PyTorch Geometric Data"""
        nodes = subgraph["nodes"]
        edges = subgraph["edges"]

        if not nodes:
            return None

        # Node ID to index mapping
        node_ids = [n.get('id', n.get('name', str(i))) for i, n in enumerate(nodes)]
        id_to_idx = {id_: idx for idx, id_ in enumerate(node_ids)}

        # Node features
        x = self._embed_nodes(nodes)

        # Edge index
        edge_list = []
        for e in edges:
            src = e.get('source', e.get('src'))
            tgt = e.get('target', e.get('tgt'))
            if src in id_to_idx and tgt in id_to_idx:
                edge_list.append([id_to_idx[src], id_to_idx[tgt]])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def _generate_response(self, question: str, context: Optional[str] = None,
                           graph_embedding: Optional[torch.Tensor] = None) -> Tuple[str, Dict]:
        """
        Generate LLM response.

        Args:
            question: The question to answer
            context: Optional text context (for soft prompt)
            graph_embedding: Optional graph embedding (for hard prompt - currently unused)

        Returns:
            Tuple of (response_text, metadata_dict)
        """
        # Build prompt
        if context:
            system_msg = (
                "You are a knowledgeable assistant. Answer based on the provided context. "
                "Be concise and accurate. If the answer is not in the context, say so."
            )
            user_msg = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            system_msg = "You are a knowledgeable assistant. Be concise and accurate."
            user_msg = question

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.llm.device)

        # Record input length for analysis
        input_len = input_ids.shape[1]

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.llm.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                output_attentions=False
            )
        gen_time = time.time() - start_time

        # Decode
        response = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        ).strip()

        metadata = {
            "input_tokens": input_len,
            "output_tokens": outputs.shape[1] - input_len,
            "generation_time": gen_time,
            "context_length": len(context) if context else 0
        }

        return response, metadata

    # =========================================================================
    # Experiment Methods
    # =========================================================================

    def run_llm_only(self, question: str) -> Tuple[str, Dict]:
        """Baseline: LLM with no graph context"""
        return self._generate_response(question)

    def run_soft_prompt(self, question: str, subgraph: Dict) -> Tuple[str, Dict]:
        """
        Soft Prompt: Graph information as text in context.

        This is the standard Graph RAG approach where graph data
        is serialized to text and added to the prompt.
        """
        # Format graph as text
        context = SoftPromptFormatter.format(
            nodes=subgraph["nodes"],
            edges=subgraph["edges"],
            style=self.config.soft_prompt_format,
            include_props=self.config.include_node_properties
        )

        response, metadata = self._generate_response(question, context=context)
        metadata["prompt_type"] = "soft"
        metadata["num_nodes"] = len(subgraph["nodes"])
        metadata["num_edges"] = len(subgraph["edges"])

        return response, metadata

    def run_hard_prompt(self, question: str, subgraph: Dict) -> Tuple[str, Dict]:
        """
        Hard Prompt: Graph encoded via GNN.

        In a full implementation, the GNN embedding would be:
        1. Projected to LLM's hidden dimension
        2. Prepended as virtual tokens OR
        3. Used as cross-attention input

        Current implementation: GNN embedding + minimal text context
        (Full integration requires model modification)
        """
        # Build PyG graph
        pyg_data = self._build_pyg_data(subgraph)

        if pyg_data is None or pyg_data.x.shape[0] == 0:
            # Fallback to LLM only
            response, metadata = self.run_llm_only(question)
            metadata["prompt_type"] = "hard_fallback"
            return response, metadata

        # GNN encoding
        pyg_data = pyg_data.to(self.device)
        self.gnn.eval()

        start_time = time.time()
        with torch.no_grad():
            graph_emb = self.gnn(pyg_data.x, pyg_data.edge_index)
        gnn_time = time.time() - start_time

        # For fair comparison, we still need to provide some context
        # In a full implementation, graph_emb would be injected directly
        # Here we provide minimal textual context + note the GNN embedding

        # Minimal context: just entity names (structure captured by GNN)
        entity_names = [n.get('name', n.get('id', '?')) for n in subgraph["nodes"][:20]]
        minimal_context = f"Relevant entities: {', '.join(entity_names)}"

        response, metadata = self._generate_response(
            question,
            context=minimal_context,
            graph_embedding=graph_emb
        )

        metadata["prompt_type"] = "hard"
        metadata["num_nodes"] = len(subgraph["nodes"])
        metadata["num_edges"] = len(subgraph["edges"])
        metadata["gnn_time"] = gnn_time
        metadata["graph_emb_norm"] = graph_emb.norm().item()

        return response, metadata

    def run_comparison(self, question_id: str, question: str,
                       ground_truth: str = None) -> Dict:
        """
        Run all three approaches on a single question and compare.
        """
        result = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth
        }

        # Get subgraph
        subgraph = self.data_loader.get_subgraph(
            question_id,
            max_hops=self.config.max_hops
        )
        result["subgraph_nodes"] = len(subgraph["nodes"])
        result["subgraph_edges"] = len(subgraph["edges"])

        # 1. LLM Only (baseline)
        print("  [1/3] LLM only...")
        resp, meta = self.run_llm_only(question)
        result["llm_only_response"] = resp
        result["llm_only_meta"] = meta

        # 2. Soft Prompt
        print("  [2/3] Soft prompt...")
        resp, meta = self.run_soft_prompt(question, subgraph)
        result["soft_prompt_response"] = resp
        result["soft_prompt_meta"] = meta

        # 3. Hard Prompt (GNN)
        print("  [3/3] Hard prompt (GNN)...")
        resp, meta = self.run_hard_prompt(question, subgraph)
        result["hard_prompt_response"] = resp
        result["hard_prompt_meta"] = meta

        # Memory cleanup
        torch.cuda.empty_cache()

        return result

    def run_experiment(self, questions_df: pd.DataFrame,
                       sample_size: int = None) -> pd.DataFrame:
        """
        Run full experiment on a dataset.
        """
        if sample_size:
            questions_df = questions_df.head(sample_size)

        results = []

        for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df),
                            desc="Running experiment"):
            print(f"\nQuestion {idx}: {row['text'][:60]}...")

            result = self.run_comparison(
                question_id=row['id'],
                question=row['text'],
                ground_truth=row.get('answer')
            )
            results.append(result)

        self.results = results
        return pd.DataFrame(results)

    def save_results(self, path: str = "soft_vs_hard_results.json"):
        """Save experiment results"""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {path}")

    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY: Soft Prompt vs Hard Prompt (GNN)")
        print("="*80)

        # Aggregate metrics
        soft_tokens = [r["soft_prompt_meta"]["input_tokens"] for r in self.results]
        hard_tokens = [r["hard_prompt_meta"]["input_tokens"] for r in self.results]
        soft_times = [r["soft_prompt_meta"]["generation_time"] for r in self.results]
        hard_times = [r["hard_prompt_meta"]["generation_time"] for r in self.results]

        print(f"\nSamples: {len(self.results)}")
        print(f"\nToken Usage (avg):")
        print(f"  Soft Prompt: {np.mean(soft_tokens):.0f} tokens")
        print(f"  Hard Prompt: {np.mean(hard_tokens):.0f} tokens")
        print(f"  Reduction:   {(1 - np.mean(hard_tokens)/np.mean(soft_tokens))*100:.1f}%")

        print(f"\nGeneration Time (avg):")
        print(f"  Soft Prompt: {np.mean(soft_times):.2f}s")
        print(f"  Hard Prompt: {np.mean(hard_times):.2f}s")

        if self.results[0].get("hard_prompt_meta", {}).get("gnn_time"):
            gnn_times = [r["hard_prompt_meta"]["gnn_time"] for r in self.results]
            print(f"  GNN Encoding: {np.mean(gnn_times):.4f}s (avg)")

        # Sample responses
        print(f"\n{'='*80}")
        print("SAMPLE RESPONSES")
        print("="*80)

        for i, r in enumerate(self.results[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Q: {r['question'][:80]}...")
            print(f"GT: {r['ground_truth'][:80] if r['ground_truth'] else 'N/A'}...")
            print(f"\n[LLM Only]: {r['llm_only_response'][:150]}...")
            print(f"\n[Soft Prompt]: {r['soft_prompt_response'][:150]}...")
            print(f"\n[Hard Prompt]: {r['hard_prompt_response'][:150]}...")

    def cleanup(self):
        """Release resources"""
        if self.data_loader:
            self.data_loader.close()

        # Save embedding cache
        if self.entity_embeddings:
            np.savez("entity_embeddings_cache.npz", embeddings=self.entity_embeddings)

        torch.cuda.empty_cache()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution"""

    config = ExperimentConfig(
        # Update for your environment
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database="finderlpg",

        # Model settings
        llm_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",

        # Experiment settings
        soft_prompt_format="structured",  # Try "natural", "triple"
        top_k_nodes=20,
        max_hops=2,

        # For memory-constrained environments
        use_4bit=False,
    )

    print("="*80)
    print("SOFT PROMPT vs HARD PROMPT (GNN+LLM) EXPERIMENT")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"LLM: {config.llm_model_id}")
    print(f"Soft prompt format: {config.soft_prompt_format}")

    exp = SoftVsHardExperiment(config)

    try:
        # Setup
        exp.setup(["neo4j", "embeddings", "llm", "gnn"])

        # Load questions
        questions_df = exp.data_loader.load_questions(limit=100)
        print(f"\nLoaded {len(questions_df)} questions")

        # Run experiment
        results_df = exp.run_experiment(questions_df, sample_size=5)

        # Save and summarize
        exp.save_results()
        exp.print_summary()

        # Also save as CSV for easy viewing
        results_df.to_csv("soft_vs_hard_results.csv", index=False)

    finally:
        exp.cleanup()

    return exp


if __name__ == "__main__":
    exp = main()
