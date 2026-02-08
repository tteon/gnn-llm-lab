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
from .config import (
    BaseConfig, Neo4jConfig, ModelConfig, ExperimentConfig,
    AttentionConfig, FewShotConfig, TrainingConfig, load_config,
)
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
from .attention import AttentionExtractor, AttentionResult
from .attention_analysis import AttentionAnalyzer
from .local_llm import LocalLLMManager, LocalLLMResponse, MODEL_REGISTRY
from .few_shot import FewShotSelector
from .evaluation import Evaluator, EvaluationResult

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
    # Config (extended)
    "AttentionConfig",
    "FewShotConfig",
    "TrainingConfig",
    # Attention
    "AttentionExtractor",
    "AttentionResult",
    "AttentionAnalyzer",
    # Local LLM
    "LocalLLMManager",
    "LocalLLMResponse",
    "MODEL_REGISTRY",
    # Few-shot
    "FewShotSelector",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
]
