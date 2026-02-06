"""
Custom exceptions for GNN+LLM experiments.

Provides specific exception types for better error handling and debugging.
"""

from typing import Optional, Any


class GNNLLMError(Exception):
    """Base exception for all GNN+LLM errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message


class ConfigurationError(GNNLLMError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = repr(value)
        super().__init__(message, details)
        self.field = field
        self.value = value


class Neo4jConnectionError(GNNLLMError):
    """Raised when Neo4j connection fails."""

    def __init__(
        self,
        message: str,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        attempts: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {}
        if uri:
            details["uri"] = uri
        if database:
            details["database"] = database
        if attempts:
            details["attempts"] = attempts
        super().__init__(message, details)
        self.uri = uri
        self.database = database
        self.attempts = attempts
        self.original_error = original_error


class DataLoadError(GNNLLMError):
    """Raised when data loading fails."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        data_type: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {}
        if file_path:
            details["file_path"] = file_path
        if data_type:
            details["data_type"] = data_type
        super().__init__(message, details)
        self.file_path = file_path
        self.data_type = data_type
        self.original_error = original_error


class ModelLoadError(GNNLLMError):
    """Raised when model loading fails."""

    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        model_type: Optional[str] = None,
        device: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {}
        if model_id:
            details["model_id"] = model_id
        if model_type:
            details["model_type"] = model_type
        if device:
            details["device"] = device
        super().__init__(message, details)
        self.model_id = model_id
        self.model_type = model_type
        self.device = device
        self.original_error = original_error


class GraphProcessingError(GNNLLMError):
    """Raised when graph processing fails."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if num_nodes is not None:
            details["num_nodes"] = num_nodes
        if num_edges is not None:
            details["num_edges"] = num_edges
        super().__init__(message, details)
        self.operation = operation
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.original_error = original_error


class EmbeddingError(GNNLLMError):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        message: str,
        entity_id: Optional[str] = None,
        batch_size: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {}
        if entity_id:
            details["entity_id"] = entity_id
        if batch_size:
            details["batch_size"] = batch_size
        super().__init__(message, details)
        self.entity_id = entity_id
        self.batch_size = batch_size
        self.original_error = original_error


class GenerationError(GNNLLMError):
    """Raised when LLM generation fails."""

    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        input_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {}
        if model_id:
            details["model_id"] = model_id
        if input_tokens:
            details["input_tokens"] = input_tokens
        if max_tokens:
            details["max_tokens"] = max_tokens
        super().__init__(message, details)
        self.model_id = model_id
        self.input_tokens = input_tokens
        self.max_tokens = max_tokens
        self.original_error = original_error
