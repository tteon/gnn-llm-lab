"""Tests for utility modules."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging,
    get_logger,
    ExperimentConfig,
    Neo4jConfig,
    ModelConfig,
    ConfigurationError,
    GraphFormatter,
    set_seed,
)
from src.utils.formatting import flatten_properties, parse_json_field


class TestLogging:
    """Test logging configuration."""

    def test_setup_logging(self, tmp_path):
        """Test logging setup creates logger."""
        logger = setup_logging(
            level="DEBUG",
            log_dir=str(tmp_path),
            module_name="test_logger"
        )
        assert logger is not None
        assert logger.name == "test_logger"

    def test_get_logger(self):
        """Test getting child logger."""
        logger = get_logger("test_module")
        assert logger.name == "gnnllm.test_module"


class TestConfiguration:
    """Test configuration management."""

    def test_neo4j_config_defaults(self):
        """Test Neo4j config has sensible defaults."""
        config = Neo4jConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.user == "neo4j"
        assert config.max_retries == 3

    def test_neo4j_config_validation(self):
        """Test Neo4j config validation."""
        config = Neo4jConfig()
        validated = config.validate()
        assert validated is config

    def test_neo4j_config_invalid_uri(self):
        """Test invalid URI raises error."""
        config = Neo4jConfig(uri="http://invalid:7687")
        with pytest.raises(ConfigurationError) as excinfo:
            config.validate()
        assert "uri" in str(excinfo.value).lower()

    def test_model_config_defaults(self):
        """Test model config has sensible defaults."""
        config = ModelConfig()
        assert "llama" in config.llm_model_id.lower()
        assert config.embedding_dim == 384

    def test_experiment_config_full(self):
        """Test full experiment config."""
        config = ExperimentConfig()
        validated = config.validate()
        assert validated.neo4j is not None
        assert validated.model is not None
        assert validated.seed == 42


class TestGraphFormatter:
    """Test graph formatting utilities."""

    def setup_method(self):
        """Setup test data."""
        self.nodes = [
            {"id": "1", "label": "Company", "name": "Apple Inc", "properties": {"founded": "1976"}},
            {"id": "2", "label": "Person", "name": "Tim Cook", "properties": {"role": "CEO"}},
        ]
        self.edges = [
            {"source": "2", "target": "1", "type": "CEO_OF"},
        ]

    def test_format_structured(self):
        """Test structured formatting."""
        result = GraphFormatter.format(self.nodes, self.edges, style="structured")
        assert "ENTITIES" in result
        assert "RELATIONSHIPS" in result
        assert "Apple Inc" in result
        assert "CEO_OF" in result

    def test_format_natural(self):
        """Test natural language formatting."""
        result = GraphFormatter.format(self.nodes, self.edges, style="natural")
        assert "knowledge graph" in result.lower()
        assert "ceo of" in result.lower()

    def test_format_triple(self):
        """Test triple formatting."""
        result = GraphFormatter.format(self.nodes, self.edges, style="triple")
        assert "triples" in result.lower()
        assert "(2, CEO_OF, 1)" in result

    def test_format_with_limits(self):
        """Test formatting respects limits."""
        many_nodes = [{"id": str(i), "name": f"Node{i}"} for i in range(100)]
        result = GraphFormatter.format(many_nodes, [], max_nodes=10)
        # Should only include 10 nodes
        assert result.count("Node") <= 15  # Some buffer for formatting

    def test_empty_graph(self):
        """Test formatting empty graph."""
        result = GraphFormatter.format([], [])
        assert result == ""


class TestFormatting:
    """Test low-level formatting functions."""

    def test_flatten_properties(self):
        """Test property flattening."""
        props = {
            "name": "Test",
            "nested": {"a": 1, "b": 2},
            "list_of_dicts": [{"x": 1}, {"x": 2}],
            "simple_list": [1, 2, 3],
        }
        flat = flatten_properties(props)
        assert flat["name"] == "Test"
        assert flat["simple_list"] == [1, 2, 3]
        # Nested should be JSON string
        assert '"a":' in flat["nested"] or '"a": ' in flat["nested"]

    def test_parse_json_field_string(self):
        """Test parsing JSON string."""
        result = parse_json_field('{"a": 1}')
        assert result == {"a": 1}

    def test_parse_json_field_already_parsed(self):
        """Test parsing already parsed value."""
        data = {"a": 1}
        result = parse_json_field(data)
        assert result == data

    def test_parse_json_field_invalid(self):
        """Test parsing invalid JSON returns original."""
        result = parse_json_field("not json")
        assert result == "not json"


class TestReproducibility:
    """Test reproducibility utilities."""

    def test_set_seed(self):
        """Test seed setting works without error."""
        set_seed(42)
        # Just check it runs without error

    def test_set_seed_deterministic(self):
        """Test deterministic results after seed."""
        import numpy as np

        set_seed(12345)
        a = np.random.rand(10)

        set_seed(12345)
        b = np.random.rand(10)

        assert np.allclose(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
