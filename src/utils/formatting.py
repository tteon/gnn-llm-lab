"""
Graph formatting utilities for converting graph data to text.

Provides multiple formatting strategies for injecting graph context into LLMs.
"""

import json
from typing import Any, Dict, List, Literal, Optional

from .logging_config import get_logger

logger = get_logger("formatting")


class GraphFormatter:
    """
    Converts graph data to text format for LLM context injection.

    Supports multiple formatting strategies:
    - structured: Clear sections with entities and relationships
    - natural: Natural language descriptions
    - triple: Simple (subject, predicate, object) format
    - csv: CSV-style tabular format
    """

    # Property keys to exclude from output
    EXCLUDED_PROPS = {"id", "label", "question_ids", "embedding", "vector"}

    @classmethod
    def format(
        cls,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        style: Literal["structured", "natural", "triple", "csv"] = "structured",
        max_nodes: int = 30,
        max_edges: int = 50,
        include_props: bool = True,
        include_edge_types: bool = True,
    ) -> str:
        """
        Format graph data as text.

        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            style: Formatting style
            max_nodes: Maximum nodes to include
            max_edges: Maximum edges to include
            include_props: Include node properties
            include_edge_types: Include relationship types

        Returns:
            Formatted text representation
        """
        formatters = {
            "structured": cls._format_structured,
            "natural": cls._format_natural,
            "triple": cls._format_triple,
            "csv": cls._format_csv,
        }

        formatter = formatters.get(style)
        if formatter is None:
            logger.warning(f"Unknown format style '{style}', using 'structured'")
            formatter = cls._format_structured

        # Limit nodes and edges
        limited_nodes = nodes[:max_nodes] if nodes else []
        limited_edges = edges[:max_edges] if edges else []

        return formatter(
            limited_nodes,
            limited_edges,
            include_props=include_props,
            include_edge_types=include_edge_types,
        )

    @classmethod
    def _format_structured(
        cls,
        nodes: List[Dict],
        edges: List[Dict],
        include_props: bool = True,
        include_edge_types: bool = True,
    ) -> str:
        """
        Structured format with clear sections.

        Example:
        === ENTITIES ===
        [Company] Apple Inc: technology company
        [Person] Tim Cook: CEO

        === RELATIONSHIPS ===
        Tim Cook --[CEO_OF]--> Apple Inc
        """
        parts = []

        # Entities section
        if nodes:
            parts.append("=== ENTITIES ===")
            for n in nodes:
                label = n.get("label", "Entity")
                name = n.get("name", n.get("id", "Unknown"))

                desc = ""
                if include_props:
                    props = n.get("properties", {})
                    prop_items = cls._filter_properties(props)
                    if prop_items:
                        desc = f": {', '.join(prop_items[:3])}"

                parts.append(f"[{label}] {name}{desc}")

        # Relationships section
        if edges:
            parts.append("\n=== RELATIONSHIPS ===")
            for e in edges:
                src = e.get("source", e.get("src", "?"))
                tgt = e.get("target", e.get("tgt", "?"))
                rel = e.get("type", e.get("relType", "RELATED"))

                if include_edge_types:
                    parts.append(f"{src} --[{rel}]--> {tgt}")
                else:
                    parts.append(f"{src} --> {tgt}")

        return "\n".join(parts)

    @classmethod
    def _format_natural(
        cls,
        nodes: List[Dict],
        edges: List[Dict],
        include_props: bool = True,
        include_edge_types: bool = True,
    ) -> str:
        """
        Natural language format.

        Example:
        The knowledge graph contains the following information:
        Apple Inc is a technology company. Tim Cook is the CEO of Apple Inc.
        """
        sentences = ["The knowledge graph contains the following information:"]

        # Build node name lookup
        node_map = {}
        for n in nodes:
            node_id = n.get("id", n.get("name"))
            node_map[node_id] = n.get("name", node_id)

        # Convert edges to sentences
        for e in edges:
            src_id = e.get("source", e.get("src"))
            tgt_id = e.get("target", e.get("tgt"))
            rel = e.get("type", e.get("relType", "is related to"))

            src_name = node_map.get(src_id, src_id)
            tgt_name = node_map.get(tgt_id, tgt_id)

            # Convert relationship type to natural language
            rel_natural = cls._relation_to_natural(rel) if include_edge_types else "is related to"
            sentences.append(f"{src_name} {rel_natural} {tgt_name}.")

        return " ".join(sentences)

    @classmethod
    def _format_triple(
        cls,
        nodes: List[Dict],
        edges: List[Dict],
        include_props: bool = True,
        include_edge_types: bool = True,
    ) -> str:
        """
        Simple triple format (subject, predicate, object).

        Example:
        Knowledge triples:
        (Apple Inc, founded_in, 1976)
        (Tim Cook, CEO_OF, Apple Inc)
        """
        lines = ["Knowledge triples:"]

        for e in edges:
            src = e.get("source", e.get("src", "?"))
            tgt = e.get("target", e.get("tgt", "?"))
            rel = e.get("type", e.get("relType", "RELATED"))

            if include_edge_types:
                lines.append(f"({src}, {rel}, {tgt})")
            else:
                lines.append(f"({src}, related_to, {tgt})")

        return "\n".join(lines)

    @classmethod
    def _format_csv(
        cls,
        nodes: List[Dict],
        edges: List[Dict],
        include_props: bool = True,
        include_edge_types: bool = True,
    ) -> str:
        """
        CSV-style tabular format.

        Example:
        NODES:
        id,label,name,properties
        1,Company,Apple Inc,"tech company"

        EDGES:
        source,type,target
        Tim Cook,CEO_OF,Apple Inc
        """
        lines = []

        # Nodes table
        if nodes:
            lines.append("NODES:")
            lines.append("id,label,name" + (",properties" if include_props else ""))
            for n in nodes:
                node_id = n.get("id", "")
                label = n.get("label", "Entity")
                name = n.get("name", node_id)

                row = f"{node_id},{label},{name}"
                if include_props:
                    props = n.get("properties", {})
                    prop_str = ";".join(cls._filter_properties(props)[:3])
                    row += f",\"{prop_str}\""
                lines.append(row)

        # Edges table
        if edges:
            lines.append("\nEDGES:")
            lines.append("source,type,target" if include_edge_types else "source,target")
            for e in edges:
                src = e.get("source", e.get("src", ""))
                tgt = e.get("target", e.get("tgt", ""))
                rel = e.get("type", e.get("relType", "RELATED"))

                if include_edge_types:
                    lines.append(f"{src},{rel},{tgt}")
                else:
                    lines.append(f"{src},{tgt}")

        return "\n".join(lines)

    @classmethod
    def _filter_properties(cls, props: Dict[str, Any]) -> List[str]:
        """Filter and format properties for display."""
        items = []
        for k, v in props.items():
            if k.lower() in cls.EXCLUDED_PROPS:
                continue
            if v is None or v == "":
                continue
            # Truncate long values
            v_str = str(v)
            if len(v_str) > 50:
                v_str = v_str[:47] + "..."
            items.append(f"{k}={v_str}")
        return items

    @classmethod
    def _relation_to_natural(cls, rel: str) -> str:
        """Convert relationship type to natural language."""
        # Handle common patterns
        rel_lower = rel.lower()

        # Direct mappings
        mappings = {
            "ceo_of": "is the CEO of",
            "subsidiary_of": "is a subsidiary of",
            "parent_of": "is the parent company of",
            "located_in": "is located in",
            "founded_in": "was founded in",
            "works_for": "works for",
            "member_of": "is a member of",
            "part_of": "is part of",
            "type_of": "is a type of",
            "has_a": "has a",
            "owns": "owns",
            "acquired": "acquired",
        }

        if rel_lower in mappings:
            return mappings[rel_lower]

        # Generic conversion: SOME_RELATION -> some relation
        return rel.lower().replace("_", " ")

    @classmethod
    def format_node_description(
        cls,
        node: Dict[str, Any],
        include_props: bool = True,
    ) -> str:
        """
        Format a single node for description.

        Args:
            node: Node dictionary
            include_props: Include properties

        Returns:
            Formatted node description
        """
        label = node.get("label", "Entity")
        name = node.get("name", node.get("id", "Unknown"))

        desc = f"[{label}] {name}"

        if include_props:
            props = node.get("properties", {})
            prop_items = cls._filter_properties(props)
            if prop_items:
                desc += f" ({', '.join(prop_items[:5])})"

        return desc

    @classmethod
    def format_edge_description(
        cls,
        edge: Dict[str, Any],
        node_map: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Format a single edge for description.

        Args:
            edge: Edge dictionary
            node_map: Optional mapping of node IDs to names

        Returns:
            Formatted edge description
        """
        src = edge.get("source", edge.get("src", "?"))
        tgt = edge.get("target", edge.get("tgt", "?"))
        rel = edge.get("type", edge.get("relType", "RELATED"))

        if node_map:
            src = node_map.get(src, src)
            tgt = node_map.get(tgt, tgt)

        return f"{src} --[{rel}]--> {tgt}"


def flatten_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested properties for Neo4j compatibility.

    Neo4j only supports primitive types and arrays of primitives.
    Nested dicts/objects are serialized to JSON strings.

    Args:
        props: Property dictionary (possibly nested)

    Returns:
        Flattened property dictionary
    """
    flattened = {}
    for k, v in props.items():
        if isinstance(v, dict):
            flattened[k] = json.dumps(v)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            flattened[k] = json.dumps(v)
        else:
            flattened[k] = v
    return flattened


def parse_json_field(value: Any) -> Any:
    """
    Safely parse a JSON field that might be a string or already parsed.

    Args:
        value: Value to parse

    Returns:
        Parsed value or original if parsing fails
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value
