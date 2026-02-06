"""
Configuration management with validation for GNN+LLM experiments.

Provides dataclass-based configuration with:
- Environment variable support
- Validation for all fields
- Type checking
- Sensible defaults
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Literal
import json

import torch
from dotenv import load_dotenv

from .exceptions import ConfigurationError
from .logging_config import get_logger

logger = get_logger("config")

# Load environment variables from .env file
load_dotenv()


def _get_env(key: str, default: str = None, required: bool = False) -> Optional[str]:
    """Get environment variable with optional requirement check."""
    value = os.getenv(key, default)
    if required and value is None:
        raise ConfigurationError(f"Required environment variable not set", field=key)
    return value


def _validate_path(path: str, must_exist: bool = False, create_parents: bool = False) -> Path:
    """Validate a file/directory path."""
    p = Path(path)
    if must_exist and not p.exists():
        raise ConfigurationError(f"Path does not exist: {path}", field="path", value=path)
    if create_parents and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _validate_device(device: str) -> str:
    """Validate and normalize device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if ":" in device:
            # Specific GPU index
            gpu_idx = int(device.split(":")[1])
            if gpu_idx >= torch.cuda.device_count():
                raise ConfigurationError(
                    f"GPU index {gpu_idx} not available",
                    field="device",
                    value=device
                )
    return device


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str = field(default_factory=lambda: _get_env("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: _get_env("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: _get_env("NEO4J_PASSWORD", "password"))
    database: str = field(default_factory=lambda: _get_env("NEO4J_DATABASE", "finderlpg"))

    # Connection settings
    max_connection_lifetime: int = 3600  # seconds
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0  # seconds

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds (base for exponential backoff)
    retry_max_delay: float = 30.0  # seconds

    def validate(self) -> "Neo4jConfig":
        """Validate configuration values."""
        if not self.uri:
            raise ConfigurationError("Neo4j URI is required", field="uri")
        if not self.uri.startswith(("bolt://", "bolt+s://", "neo4j://", "neo4j+s://")):
            raise ConfigurationError(
                "Invalid Neo4j URI scheme",
                field="uri",
                value=self.uri
            )
        if self.max_retries < 0:
            raise ConfigurationError("max_retries must be non-negative", field="max_retries")
        if self.connection_timeout <= 0:
            raise ConfigurationError("connection_timeout must be positive", field="connection_timeout")
        return self


@dataclass
class ModelConfig:
    """LLM and embedding model configuration."""
    # LLM settings (local HuggingFace model)
    llm_model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm_torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    llm_device_map: str = "auto"
    llm_attn_implementation: str = "eager"  # "eager", "flash_attention_2", "sdpa"
    llm_use_4bit: bool = False
    llm_use_8bit: bool = False

    # LLM API settings (for remote inference via OpenAI-compatible API)
    llm_api_base_url: str = field(default_factory=lambda: _get_env("LLM_API_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: _get_env("LLM_API_KEY", ""))
    llm_api_model: str = field(default_factory=lambda: _get_env("LLM_API_MODEL", ""))

    # Embedding model settings
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # GNN settings
    gnn_hidden_dim: int = 256
    gnn_output_dim: int = 384
    gnn_num_layers: int = 2
    gnn_heads: int = 4
    gnn_dropout: float = 0.1

    # Device
    device: str = "auto"

    @property
    def use_api(self) -> bool:
        """Whether to use remote LLM API instead of local model."""
        return bool(self.llm_api_base_url and self.llm_api_key and self.llm_api_model)

    def validate(self) -> "ModelConfig":
        """Validate configuration values."""
        # Validate device
        self.device = _validate_device(self.device)

        # Validate API settings if provided
        if self.llm_api_base_url:
            if not self.llm_api_base_url.startswith(("http://", "https://")):
                raise ConfigurationError(
                    "LLM API base URL must start with http:// or https://",
                    field="llm_api_base_url",
                    value=self.llm_api_base_url
                )
            if not self.llm_api_key:
                raise ConfigurationError(
                    "LLM API key is required when base URL is set",
                    field="llm_api_key"
                )
            if not self.llm_api_model:
                raise ConfigurationError(
                    "LLM API model is required when base URL is set",
                    field="llm_api_model"
                )

        # Validate dtype
        valid_dtypes = ["float16", "bfloat16", "float32"]
        if self.llm_torch_dtype not in valid_dtypes:
            raise ConfigurationError(
                f"Invalid torch dtype, must be one of {valid_dtypes}",
                field="llm_torch_dtype",
                value=self.llm_torch_dtype
            )

        # Validate attention implementation
        valid_attn = ["eager", "flash_attention_2", "sdpa"]
        if self.llm_attn_implementation not in valid_attn:
            raise ConfigurationError(
                f"Invalid attention implementation, must be one of {valid_attn}",
                field="llm_attn_implementation",
                value=self.llm_attn_implementation
            )

        # Validate GNN settings
        if self.gnn_hidden_dim <= 0:
            raise ConfigurationError("GNN hidden dim must be positive", field="gnn_hidden_dim")
        if self.gnn_num_layers <= 0:
            raise ConfigurationError("GNN num layers must be positive", field="gnn_num_layers")
        if self.gnn_dropout < 0 or self.gnn_dropout > 1:
            raise ConfigurationError("GNN dropout must be between 0 and 1", field="gnn_dropout")

        return self




@dataclass
class AttentionConfig:
    """Configuration for attention score extraction."""
    enabled: bool = True
    layers_to_extract: List[int] = field(default_factory=lambda: [-1, -2, -3])
    aggregate_heads: str = "mean"  # "mean" | "max" | "none" (per-head)
    save_context_attention_only: bool = True
    top_k_tokens: int = 50
    output_dir: str = "results/attention"

    def validate(self) -> "AttentionConfig":
        if self.aggregate_heads not in ("mean", "max", "none"):
            raise ConfigurationError(
                "aggregate_heads must be 'mean', 'max', or 'none'",
                field="aggregate_heads",
                value=self.aggregate_heads,
            )
        if self.top_k_tokens <= 0:
            raise ConfigurationError("top_k_tokens must be positive", field="top_k_tokens")
        self.output_dir = str(_validate_path(self.output_dir, create_parents=True))
        return self


@dataclass
class FewShotConfig:
    """Configuration for few-shot example selection."""
    enabled: bool = False
    num_examples_per_category: int = 1
    include_graph_context: bool = False
    selection_strategy: str = "representative"  # centroid-nearest
    cache_path: str = "data/processed/few_shot_examples.json"

    def validate(self) -> "FewShotConfig":
        if self.num_examples_per_category <= 0:
            raise ConfigurationError(
                "num_examples_per_category must be positive",
                field="num_examples_per_category",
            )
        if self.selection_strategy not in ("representative", "random"):
            raise ConfigurationError(
                "selection_strategy must be 'representative' or 'random'",
                field="selection_strategy",
                value=self.selection_strategy,
            )
        return self


@dataclass
class TrainingConfig:
    """Configuration for GNN/KGE link prediction training."""

    # Training hyperparameters
    epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 512
    patience: int = 15
    min_delta: float = 0.001

    # Model selection
    gnn_model_type: str = "gat"    # "gat" | "gcn" | "graph_transformer"
    kge_model_type: str = "transe"  # "transe" | "distmult" | "complex" | "rotate"

    # Dimensions
    hidden_dim: int = 256
    output_dim: int = 384

    # GNN-specific
    gnn_num_layers: int = 2
    gnn_heads: int = 4
    gnn_dropout: float = 0.1
    decoder_type: str = "dot"  # "dot" | "mlp"

    # KGE-specific
    kge_margin: float = 1.0
    kge_p_norm: float = 1.0

    # Data split
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Checkpoint
    checkpoint_dir: str = "results/checkpoints"
    device: str = "auto"
    seed: int = 42

    def validate(self) -> "TrainingConfig":
        """Validate configuration values."""
        self.device = _validate_device(self.device)

        if self.epochs <= 0:
            raise ConfigurationError("epochs must be positive", field="epochs")
        if self.learning_rate <= 0:
            raise ConfigurationError("learning_rate must be positive", field="learning_rate")
        if self.patience <= 0:
            raise ConfigurationError("patience must be positive", field="patience")

        valid_gnn = {"gat", "gcn", "graph_transformer"}
        if self.gnn_model_type not in valid_gnn:
            raise ConfigurationError(
                f"gnn_model_type must be one of {valid_gnn}",
                field="gnn_model_type",
                value=self.gnn_model_type,
            )

        valid_kge = {"transe", "distmult", "complex", "rotate"}
        if self.kge_model_type not in valid_kge:
            raise ConfigurationError(
                f"kge_model_type must be one of {valid_kge}",
                field="kge_model_type",
                value=self.kge_model_type,
            )

        valid_decoder = {"dot", "mlp"}
        if self.decoder_type not in valid_decoder:
            raise ConfigurationError(
                f"decoder_type must be one of {valid_decoder}",
                field="decoder_type",
                value=self.decoder_type,
            )

        if self.gnn_dropout < 0 or self.gnn_dropout > 1:
            raise ConfigurationError("gnn_dropout must be between 0 and 1", field="gnn_dropout")
        if self.val_ratio < 0 or self.val_ratio > 0.5:
            raise ConfigurationError("val_ratio must be between 0 and 0.5", field="val_ratio")
        if self.test_ratio < 0 or self.test_ratio > 0.5:
            raise ConfigurationError("test_ratio must be between 0 and 0.5", field="test_ratio")

        self.checkpoint_dir = str(_validate_path(self.checkpoint_dir, create_parents=True))
        return self

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Nested configs
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)

    # Data settings
    parquet_path: str = "data/raw/FinDER_KG_Merged.parquet"
    cache_dir: str = "data/processed"
    results_dir: str = "results"

    # Retrieval settings
    top_k_nodes: int = 20
    max_hops: int = 2
    max_context_nodes: int = 30
    max_context_edges: int = 50

    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.0  # 0 = deterministic
    do_sample: bool = False

    # Experiment settings
    batch_size: int = 500
    sample_size: Optional[int] = None  # None = all samples
    checkpoint_interval: int = 5  # Save every N samples

    # Soft prompt settings
    soft_prompt_format: Literal["structured", "natural", "triple"] = "structured"
    include_node_properties: bool = True
    include_edge_types: bool = True

    # 3-axis experiment matrix
    model_aliases: List[str] = field(default_factory=lambda: ["llama8b"])
    context_conditions: List[str] = field(
        default_factory=lambda: ["none", "lpg", "rdf", "lpg_rdf"]
    )

    # Evaluation settings
    eval_bertscore: bool = True
    eval_rouge: bool = True

    # Reproducibility
    seed: Optional[int] = 42

    # Logging
    log_level: str = "INFO"
    log_dir: str = "results/logs"

    def validate(self) -> "ExperimentConfig":
        """Validate all configuration values."""
        # Validate nested configs
        self.neo4j.validate()
        self.model.validate()
        self.attention.validate()
        self.few_shot.validate()

        # Validate paths
        self.cache_dir = str(_validate_path(self.cache_dir, create_parents=True))
        self.results_dir = str(_validate_path(self.results_dir, create_parents=True))
        self.log_dir = str(_validate_path(self.log_dir, create_parents=True))

        # Validate experiment settings
        if self.top_k_nodes <= 0:
            raise ConfigurationError("top_k_nodes must be positive", field="top_k_nodes")
        if self.max_hops <= 0:
            raise ConfigurationError("max_hops must be positive", field="max_hops")
        if self.max_new_tokens <= 0:
            raise ConfigurationError("max_new_tokens must be positive", field="max_new_tokens")
        if self.temperature < 0:
            raise ConfigurationError("temperature must be non-negative", field="temperature")
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive", field="batch_size")
        if self.checkpoint_interval <= 0:
            raise ConfigurationError("checkpoint_interval must be positive", field="checkpoint_interval")

        # Validate context conditions
        valid_contexts = {"none", "lpg", "rdf", "lpg_rdf", "text"}
        for ctx in self.context_conditions:
            if ctx not in valid_contexts:
                raise ConfigurationError(
                    f"Invalid context condition '{ctx}', must be one of {valid_contexts}",
                    field="context_conditions",
                    value=ctx,
                )

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid log level, must be one of {valid_levels}",
                field="log_level",
                value=self.log_level
            )

        return self

    def to_dict(self) -> dict:
        """Convert config to dictionary (for serialization)."""
        return {
            "neo4j": {
                "uri": self.neo4j.uri,
                "user": self.neo4j.user,
                "database": self.neo4j.database,
                "max_retries": self.neo4j.max_retries,
            },
            "model": {
                "llm_model_id": self.model.llm_api_model or self.model.llm_model_id,
                "llm_api_base_url": self.model.llm_api_base_url,
                "llm_api_model": self.model.llm_api_model,
                "use_api": self.model.use_api,
                "embedding_model_id": self.model.embedding_model_id,
                "embedding_dim": self.model.embedding_dim,
                "gnn_hidden_dim": self.model.gnn_hidden_dim,
                "gnn_num_layers": self.model.gnn_num_layers,
                "device": self.model.device,
            },
            "attention": {
                "enabled": self.attention.enabled,
                "layers_to_extract": self.attention.layers_to_extract,
                "aggregate_heads": self.attention.aggregate_heads,
                "top_k_tokens": self.attention.top_k_tokens,
            },
            "few_shot": {
                "enabled": self.few_shot.enabled,
                "num_examples_per_category": self.few_shot.num_examples_per_category,
                "selection_strategy": self.few_shot.selection_strategy,
            },
            "experiment": {
                "top_k_nodes": self.top_k_nodes,
                "max_hops": self.max_hops,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "soft_prompt_format": self.soft_prompt_format,
                "seed": self.seed,
                "model_aliases": self.model_aliases,
                "context_conditions": self.context_conditions,
                "eval_bertscore": self.eval_bertscore,
                "eval_rouge": self.eval_rouge,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        neo4j = Neo4jConfig(**data.get("neo4j", {}))
        model = ModelConfig(**data.get("model", {}))
        attention_data = data.get("attention", {})
        attention = AttentionConfig(**attention_data) if attention_data else AttentionConfig()
        few_shot_data = data.get("few_shot", {})
        few_shot = FewShotConfig(**few_shot_data) if few_shot_data else FewShotConfig()
        experiment_data = data.get("experiment", {})

        return cls(
            neo4j=neo4j,
            model=model,
            attention=attention,
            few_shot=few_shot,
            **experiment_data
        )


# Convenience aliases for backward compatibility
BaseConfig = ExperimentConfig


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    """
    Load configuration from file or environment.

    Args:
        path: Optional path to JSON config file

    Returns:
        Validated ExperimentConfig instance
    """
    if path and Path(path).exists():
        logger.info(f"Loading config from {path}")
        with open(path) as f:
            data = json.load(f)
        config = ExperimentConfig.from_dict(data)
    else:
        logger.info("Using default config with environment variables")
        config = ExperimentConfig()

    return config.validate()


def save_config(config: ExperimentConfig, path: str) -> None:
    """Save configuration to JSON file."""
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Config saved to {path}")
