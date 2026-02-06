"""
Utility modules for GNN+LLM experiments.

Provides:
- Structured logging
- Configuration management with validation
- Robust Neo4j client with retry logic
- Graph formatting utilities
- Custom exceptions
- Reproducibility utilities
- OpenAI-compatible LLM client
"""

from .logging_config import setup_logging, get_logger
from .config import BaseConfig, Neo4jConfig, ModelConfig, ExperimentConfig, load_config
from .exceptions import (
    GNNLLMError,
    ConfigurationError,
    Neo4jConnectionError,
    DataLoadError,
    ModelLoadError,
    GraphProcessingError,
)
from .neo4j_client import Neo4jClient
from .formatting import GraphFormatter, flatten_properties, parse_json_field
from .reproducibility import (
    set_seed,
    get_experiment_metadata,
    save_experiment_metadata,
    ExperimentTracker,
)
from .llm_client import LLMClient, LLMResponse

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    # Config
    "BaseConfig",
    "Neo4jConfig",
    "ModelConfig",
    "ExperimentConfig",
    "load_config",
    # Exceptions
    "GNNLLMError",
    "ConfigurationError",
    "Neo4jConnectionError",
    "DataLoadError",
    "ModelLoadError",
    "GraphProcessingError",
    # Neo4j
    "Neo4jClient",
    # Formatting
    "GraphFormatter",
    "flatten_properties",
    "parse_json_field",
    # Reproducibility
    "set_seed",
    "get_experiment_metadata",
    "save_experiment_metadata",
    "ExperimentTracker",
    # LLM Client
    "LLMClient",
    "LLMResponse",
]
