"""
FinDER KG GNN+LLM Experiment for Google Colab (A100)

This script performs:
1. Neo4j connection and data loading
2. Embedding generation (OpenAI or local)
3. LLM Baseline experiment
4. Text RAG experiment
5. Graph RAG (GNN+LLM) experiment

Environment: Google Colab with A100 GPU, High RAM
"""

# =============================================================================
# CELL 1: Environment Setup (Run this first in Colab)
# =============================================================================
"""
# Uncomment and run in Colab:

import torch
TORCH = torch.__version__.split('+')[0]
CUDA = 'cu' + torch.version.cuda.replace('.', '')
print(f"PyTorch: {TORCH}, CUDA: {CUDA}")

# Install PyTorch Geometric
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
!pip install torch_geometric transformers accelerate huggingface_hub
!pip install neo4j pandas pyarrow openai sentence-transformers
!pip install pcst_fast  # For subgraph pruning
"""

# =============================================================================
# CELL 2: Imports and Configuration
# =============================================================================

import os
import json
import ast
from typing import Optional
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Neo4j
from neo4j import GraphDatabase

# Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# PyTorch Geometric
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# Embeddings
from sentence_transformers import SentenceTransformer


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Neo4j connection (use ngrok tunnel from Colab)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "finderlpg"

    # Model settings
    llm_model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast
    embedding_dim: int = 384

    # GNN settings
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 2
    gnn_heads: int = 4

    # Retrieval settings
    top_k_nodes: int = 10
    max_hops: int = 2

    # Generation settings
    max_new_tokens: int = 256

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# CELL 3: Neo4j Data Loader
# =============================================================================

class Neo4jDataLoader:
    """Load data from Neo4j database"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )

    def verify_connection(self) -> bool:
        try:
            self.driver.verify_connectivity()
            print(f"Connected to Neo4j: {self.config.neo4j_uri}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def load_questions(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load question data from Neo4j"""
        query = """
        MATCH (q:Question)
        RETURN q.id as id, q.text as text, q.answer as answer,
               q.category as category, q.type as type, q.reasoning as reasoning
        """
        if limit:
            query += f" LIMIT {limit}"

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query)
            data = [dict(r) for r in result]

        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} questions")
        return df

    def get_subgraph_for_question(self, question_id: str, max_hops: int = 2) -> dict:
        """Get subgraph related to a question"""
        # Get entities related to this question
        query = """
        MATCH (e:Entity)
        WHERE $qid IN e.question_ids
        WITH collect(e) as seed_nodes
        UNWIND seed_nodes as seed
        OPTIONAL MATCH path = (seed)-[*1..%d]-(connected:Entity)
        WITH seed_nodes, collect(DISTINCT connected) as connected_nodes
        WITH seed_nodes + connected_nodes as all_nodes
        UNWIND all_nodes as n
        WITH DISTINCT n
        OPTIONAL MATCH (n)-[r]->(m:Entity)
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
                "nodes": record["nodes"],
                "edges": [e for e in record["edges"] if e["source"] and e["target"]]
            }
        return {"nodes": [], "edges": []}

    def get_all_entities(self, limit: Optional[int] = None) -> list[dict]:
        """Get all entity nodes for embedding"""
        query = """
        MATCH (e:Entity)
        RETURN e.id as id, e.label as label,
               coalesce(e.name, e.id) as name,
               properties(e) as properties
        """
        if limit:
            query += f" LIMIT {limit}"

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query)
            return [dict(r) for r in result]

    def close(self):
        self.driver.close()


# =============================================================================
# CELL 4: Embedding Generator
# =============================================================================

class EmbeddingGenerator:
    """Generate embeddings for nodes and queries"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device

        print(f"Loading embedding model: {config.embedding_model_id}")
        self.model = SentenceTransformer(config.embedding_model_id, device=self.device)
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def embed_entities(self, entities: list[dict]) -> dict[str, np.ndarray]:
        """Generate embeddings for entity nodes"""
        # Create text representation for each entity
        texts = []
        ids = []
        for e in entities:
            # Combine name and properties into text
            props = e.get("properties", {})
            prop_text = ", ".join(f"{k}: {v}" for k, v in props.items()
                                  if k not in ["id", "label", "question_ids"] and v)
            text = f"{e.get('label', 'Entity')}: {e.get('name', e['id'])}. {prop_text}"
            texts.append(text)
            ids.append(e["id"])

        embeddings = self.embed_text(texts)
        return {id_: emb for id_, emb in zip(ids, embeddings)}


# =============================================================================
# CELL 5: GNN Model
# =============================================================================

class GraphEncoder(torch.nn.Module):
    """GNN encoder for subgraph representation"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(torch.nn.LayerNorm(hidden_dim * heads))

        self.output_proj = torch.nn.Linear(hidden_dim * heads, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = torch.relu(x)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        # Global mean pooling
        graph_embedding = x.mean(dim=0)
        return self.output_proj(graph_embedding)


# =============================================================================
# CELL 6: Experiment Runner
# =============================================================================

class FinDERExperiment:
    """Main experiment class combining all components"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        print(f"Using device: {self.device}")

        # Initialize components
        self.data_loader = None
        self.embedding_gen = None
        self.entity_embeddings = None
        self.llm = None
        self.tokenizer = None
        self.gnn = None

    def setup_neo4j(self):
        """Initialize Neo4j connection"""
        print("\n" + "="*60)
        print("Setting up Neo4j connection...")
        self.data_loader = Neo4jDataLoader(self.config)
        if not self.data_loader.verify_connection():
            raise ConnectionError("Failed to connect to Neo4j")

    def setup_embeddings(self, force_regenerate: bool = False):
        """Initialize embedding model and generate entity embeddings"""
        print("\n" + "="*60)
        print("Setting up embeddings...")

        self.embedding_gen = EmbeddingGenerator(self.config)

        cache_path = "entity_embeddings.npz"
        if os.path.exists(cache_path) and not force_regenerate:
            print(f"Loading cached embeddings from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self.entity_embeddings = dict(data["embeddings"].item())
        else:
            print("Generating entity embeddings...")
            entities = self.data_loader.get_all_entities()
            self.entity_embeddings = self.embedding_gen.embed_entities(entities)
            np.savez(cache_path, embeddings=self.entity_embeddings)
            print(f"Cached embeddings to {cache_path}")

        print(f"Total entity embeddings: {len(self.entity_embeddings)}")

    def setup_llm(self):
        """Initialize LLM"""
        print("\n" + "="*60)
        print(f"Loading LLM: {self.config.llm_model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"  # A100 optimized
        )
        self.llm.eval()
        print(f"LLM loaded on {next(self.llm.parameters()).device}")

    def setup_gnn(self):
        """Initialize GNN encoder"""
        print("\n" + "="*60)
        print("Setting up GNN encoder...")

        self.gnn = GraphEncoder(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.gnn_hidden_dim,
            output_dim=self.config.embedding_dim,
            num_layers=self.config.gnn_num_layers,
            heads=self.config.gnn_heads
        ).to(self.device)
        print(f"GNN parameters: {sum(p.numel() for p in self.gnn.parameters()):,}")

    def build_pyg_graph(self, subgraph: dict) -> Optional[Data]:
        """Convert subgraph dict to PyTorch Geometric Data object"""
        nodes = subgraph["nodes"]
        edges = subgraph["edges"]

        if not nodes:
            return None

        # Create node ID to index mapping
        node_ids = [n["id"] for n in nodes]
        id_to_idx = {id_: idx for idx, id_ in enumerate(node_ids)}

        # Node features
        x = []
        for n in nodes:
            if n["id"] in self.entity_embeddings:
                x.append(self.entity_embeddings[n["id"]])
            else:
                x.append(np.zeros(self.config.embedding_dim))
        x = torch.tensor(np.stack(x), dtype=torch.float32)

        # Edge index
        edge_index = []
        for e in edges:
            if e["source"] in id_to_idx and e["target"] in id_to_idx:
                edge_index.append([id_to_idx[e["source"]], id_to_idx[e["target"]]])

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Node descriptions for context
        node_descs = []
        for n in nodes:
            props = n.get("properties", {})
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items()
                                if k not in ["id", "label", "question_ids", "embedding"] and v)
            desc = f"[{n.get('label', 'Entity')}] {n.get('name', n['id'])}"
            if prop_str:
                desc += f" ({prop_str})"
            node_descs.append(desc)

        edge_descs = [f"{e['source']} --{e['type']}--> {e['target']}" for e in edges]

        return Data(
            x=x,
            edge_index=edge_index,
            node_descs=node_descs,
            edge_descs=edge_descs
        )

    def generate_response(self, question: str, context: Optional[str] = None,
                         graph_embedding: Optional[torch.Tensor] = None) -> str:
        """Generate LLM response"""
        if context:
            messages = [
                {"role": "system", "content": "You are a financial expert. Answer based ONLY on the provided context. Be concise and accurate."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a financial expert. Answer based on your knowledge. Be concise."},
                {"role": "user", "content": question}
            ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.llm.device)

        with torch.no_grad():
            outputs = self.llm.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def run_llm_baseline(self, question: str) -> str:
        """Experiment A: LLM only (no context)"""
        return self.generate_response(question)

    def run_text_rag(self, question: str, references: str) -> str:
        """Experiment B: Text RAG with references"""
        try:
            parsed = ast.literal_eval(references) if isinstance(references, str) else references
            context = "\n".join(parsed) if isinstance(parsed, list) else str(references)
        except:
            context = str(references)
        return self.generate_response(question, context=context)

    def run_graph_rag(self, question: str, question_id: str) -> str:
        """Experiment C: Graph RAG with GNN"""
        # Get subgraph
        subgraph = self.data_loader.get_subgraph_for_question(
            question_id,
            max_hops=self.config.max_hops
        )

        # Build PyG graph
        pyg_data = self.build_pyg_graph(subgraph)

        if pyg_data is None or pyg_data.x.shape[0] == 0:
            return self.run_llm_baseline(question)

        # Get GNN embedding
        pyg_data = pyg_data.to(self.device)
        self.gnn.eval()
        with torch.no_grad():
            graph_emb = self.gnn(pyg_data.x, pyg_data.edge_index)

        # Build context from graph
        context_parts = []
        context_parts.append("=== Relevant Entities ===")
        for desc in pyg_data.node_descs[:20]:  # Limit to top 20
            context_parts.append(f"- {desc}")

        context_parts.append("\n=== Relationships ===")
        for desc in pyg_data.edge_descs[:30]:  # Limit to top 30
            context_parts.append(f"- {desc}")

        context = "\n".join(context_parts)
        return self.generate_response(question, context=context, graph_embedding=graph_emb)

    def run_experiment(self, df: pd.DataFrame, sample_indices: list[int] = None,
                      experiments: list[str] = ["llm", "text_rag", "graph_rag"]) -> pd.DataFrame:
        """Run all experiments on selected samples"""
        if sample_indices is None:
            sample_indices = list(range(min(10, len(df))))

        results = []

        for idx in tqdm(sample_indices, desc="Running experiments"):
            row = df.iloc[idx]
            question = row["text"]
            ground_truth = row["answer"]
            question_id = row["id"]

            result = {
                "idx": idx,
                "question_id": question_id,
                "question": question,
                "ground_truth": ground_truth,
                "category": row.get("category", ""),
            }

            print(f"\n{'='*80}")
            print(f"Sample {idx}: {question[:80]}...")
            print(f"Ground Truth: {ground_truth[:200]}...")

            if "llm" in experiments:
                print("\n[A] LLM Baseline...")
                result["llm_response"] = self.run_llm_baseline(question)
                print(f"Response: {result['llm_response'][:200]}...")

            if "text_rag" in experiments and "references" in row:
                print("\n[B] Text RAG...")
                result["text_rag_response"] = self.run_text_rag(question, row.get("references", ""))
                print(f"Response: {result['text_rag_response'][:200]}...")

            if "graph_rag" in experiments:
                print("\n[C] Graph RAG...")
                result["graph_rag_response"] = self.run_graph_rag(question, question_id)
                print(f"Response: {result['graph_rag_response'][:200]}...")

            results.append(result)

            # Memory cleanup
            torch.cuda.empty_cache()

        return pd.DataFrame(results)

    def visualize_attention(self, question: str, response: str = None):
        """Visualize attention patterns"""
        if response is None:
            response = self.run_llm_baseline(question)

        messages = [
            {"role": "system", "content": "You are a financial expert."},
            {"role": "user", "content": question}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.llm.device)

        with torch.no_grad():
            outputs = self.llm(input_ids, output_attentions=True)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        view_len = min(len(tokens), 100)
        tokens_view = tokens[-view_len:]

        # Last layer attention
        attn_matrix = outputs.attentions[-1][0].mean(dim=0).float().cpu().numpy()
        attn_view = attn_matrix[-view_len:, -view_len:]

        plt.figure(figsize=(14, 10))
        sns.heatmap(attn_view, xticklabels=tokens_view, yticklabels=tokens_view,
                   cmap="viridis", square=True)
        plt.title("Last Layer Attention Map")
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        plt.show()

        return response

    def cleanup(self):
        """Cleanup resources"""
        if self.data_loader:
            self.data_loader.close()
        torch.cuda.empty_cache()


# =============================================================================
# CELL 7: Main Execution
# =============================================================================

def main():
    """Main execution function for Colab"""

    # Configuration
    config = ExperimentConfig(
        # Update these for your environment:
        neo4j_uri="bolt://localhost:7687",  # Use ngrok tunnel URL if remote
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database="finderlpg",

        # Model settings
        llm_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",

        # A100 optimized settings
        max_new_tokens=256,
        gnn_hidden_dim=256,
        gnn_num_layers=2,
    )

    # Initialize experiment
    exp = FinDERExperiment(config)

    try:
        # Setup all components
        exp.setup_neo4j()
        exp.setup_embeddings()
        exp.setup_llm()
        exp.setup_gnn()

        # Load questions
        df = exp.data_loader.load_questions(limit=100)
        print(f"\nLoaded {len(df)} questions")
        print(df.head())

        # Run experiments on first 5 samples
        results = exp.run_experiment(
            df,
            sample_indices=[0, 1, 2, 3, 4],
            experiments=["llm", "graph_rag"]  # Skip text_rag if no references
        )

        # Save results
        results.to_csv("experiment_results.csv", index=False)
        print("\nResults saved to experiment_results.csv")

        # Display summary
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        for _, row in results.iterrows():
            print(f"\nQ: {row['question'][:60]}...")
            print(f"GT: {row['ground_truth'][:60]}...")
            if 'llm_response' in row:
                print(f"LLM: {row['llm_response'][:60]}...")
            if 'graph_rag_response' in row:
                print(f"GraphRAG: {row['graph_rag_response'][:60]}...")

    finally:
        exp.cleanup()

    return exp, results


# Run if executed directly
if __name__ == "__main__":
    exp, results = main()


# =============================================================================
# CELL 8: Interactive Usage (for Colab notebooks)
# =============================================================================
"""
# After running main(), you can interactively test:

# Single question test
question = "What is Cboe's share repurchase program?"
print("LLM:", exp.run_llm_baseline(question))
print("GraphRAG:", exp.run_graph_rag(question, "30eb0cd9"))

# Visualize attention
exp.visualize_attention(question)

# Run on more samples
more_results = exp.run_experiment(df, sample_indices=range(10, 20))
"""
