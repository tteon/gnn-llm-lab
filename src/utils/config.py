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
class ExperimentConfig:
    """Complete experiment configuration."""
    # Nested configs
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

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
            "experiment": {
                "top_k_nodes": self.top_k_nodes,
                "max_hops": self.max_hops,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "soft_prompt_format": self.soft_prompt_format,
                "seed": self.seed,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        neo4j = Neo4jConfig(**data.get("neo4j", {}))
        model = ModelConfig(**data.get("model", {}))
        experiment_data = data.get("experiment", {})

        return cls(
            neo4j=neo4j,
            model=model,
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
