"""
Load FinDER KG data from parquet into Neo4j as both LPG and RDF graphs.

Creates two separate database instances:
- finder_lpg: Labeled Property Graph representation
- finder_rdf: RDF triple representation using n10s

Features:
- Robust error handling with retry logic
- Progress logging with timestamps
- Validation of input data
- Batch processing for large datasets
"""

import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging,
    get_logger,
    Neo4jClient,
    Neo4jConfig,
    ExperimentConfig,
    DataLoadError,
    Neo4jConnectionError,
)
from src.utils.formatting import flatten_properties, parse_json_field

# Initialize logging
logger = get_logger("load_finder_kg")


class FinDERDataLoader:
    """
    Load FinDER KG data into Neo4j databases.

    Handles both LPG and RDF representations with proper error handling
    and progress tracking.
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        batch_size: int = 500,
    ):
        """
        Initialize the data loader.

        Args:
            config: Experiment configuration (uses defaults if None)
            batch_size: Number of items per batch for Neo4j operations
        """
        self.config = config or ExperimentConfig()
        self.batch_size = batch_size
        self.client: Optional[Neo4jClient] = None

    def connect(self) -> "FinDERDataLoader":
        """Establish Neo4j connection."""
        self.client = Neo4jClient(self.config.neo4j)
        self.client.connect()
        return self

    def close(self) -> None:
        """Close Neo4j connection."""
        if self.client:
            self.client.close()
            self.client = None

    def __enter__(self) -> "FinDERDataLoader":
        """Context manager entry."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def load_parquet(self, path: str) -> pd.DataFrame:
        """
        Load parquet file with validation.

        Args:
            path: Path to parquet file

        Returns:
            Loaded DataFrame

        Raises:
            DataLoadError: If file not found or invalid
        """
        parquet_path = Path(path)
        if not parquet_path.exists():
            raise DataLoadError(
                f"Parquet file not found: {path}",
                file_path=str(parquet_path),
                data_type="parquet"
            )

        logger.info(f"Loading parquet from {parquet_path}")
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise DataLoadError(
                f"Failed to read parquet file: {e}",
                file_path=str(parquet_path),
                data_type="parquet",
                original_error=e
            )

        # Validate required columns
        required_cols = ["_id", "text", "answer", "lpg_nodes", "lpg_edges", "rdf_triples"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise DataLoadError(
                f"Missing required columns: {missing}",
                file_path=str(parquet_path),
                data_type="parquet"
            )

        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df

    def create_database(self, db_name: str) -> None:
        """
        Create a database if it doesn't exist.

        Args:
            db_name: Name of the database to create
        """
        if not self.client:
            raise Neo4jConnectionError("Not connected to Neo4j")

        try:
            # Check existing databases
            result = self.client.query("SHOW DATABASES", database="system")
            existing = [r["name"] for r in result]

            if db_name not in existing:
                self.client.execute(f"CREATE DATABASE {db_name}", database="system")
                logger.info(f"Created database: {db_name}")
                # Wait for database to be online
                time.sleep(2)
            else:
                logger.info(f"Database already exists: {db_name}")

        except Exception as e:
            logger.warning(f"Could not create database {db_name}: {e}")
            # Continue anyway - database might exist or we might not have permissions

    def clear_database(self, db_name: str) -> None:
        """
        Clear all nodes and relationships in a database.

        Args:
            db_name: Name of the database to clear
        """
        if not self.client:
            raise Neo4jConnectionError("Not connected to Neo4j")

        logger.info(f"Clearing database: {db_name}")
        try:
            self.client.execute("MATCH (n) DETACH DELETE n", database=db_name)
            logger.info(f"Cleared database: {db_name}")
        except Exception as e:
            logger.error(f"Failed to clear database {db_name}: {e}")
            raise

    def load_lpg_data(self, df: pd.DataFrame, db_name: str = "finderlpg") -> dict:
        """
        Load LPG nodes and edges into Neo4j.

        Args:
            df: DataFrame with lpg_nodes and lpg_edges columns
            db_name: Target database name

        Returns:
            Statistics dictionary with node/edge counts
        """
        if not self.client:
            raise Neo4jConnectionError("Not connected to Neo4j")

        logger.info(f"Loading LPG data into {db_name}")

        # Collect all unique nodes and edges
        all_nodes: dict[str, dict[str, Any]] = {}
        all_edges: list[dict[str, Any]] = []
        errors = []

        for idx, row in df.iterrows():
            question_id = row["_id"]

            try:
                nodes = parse_json_field(row["lpg_nodes"])
                edges = parse_json_field(row["lpg_edges"])

                if not isinstance(nodes, list):
                    logger.warning(f"Row {idx}: lpg_nodes is not a list, skipping")
                    continue

                # Deduplicate nodes by id, merge properties
                for node in nodes:
                    if not isinstance(node, dict) or "id" not in node:
                        continue

                    node_id = node["id"]
                    if node_id not in all_nodes:
                        all_nodes[node_id] = {
                            "id": node_id,
                            "label": node.get("label", "Entity"),
                            "properties": flatten_properties(node.get("properties", {})),
                            "question_ids": [question_id]
                        }
                    else:
                        if question_id not in all_nodes[node_id]["question_ids"]:
                            all_nodes[node_id]["question_ids"].append(question_id)

                # Collect edges
                if isinstance(edges, list):
                    for edge in edges:
                        if not isinstance(edge, dict):
                            continue
                        if not edge.get("source") or not edge.get("target"):
                            continue

                        edge_data = {
                            "source": edge["source"],
                            "target": edge["target"],
                            "type": edge.get("type", "RELATED"),
                            "properties": flatten_properties(edge.get("properties", {})),
                            "question_id": question_id
                        }
                        all_edges.append(edge_data)

            except Exception as e:
                errors.append(f"Row {idx}: {e}")
                if len(errors) <= 5:
                    logger.warning(f"Error processing row {idx}: {e}")

        if errors:
            logger.warning(f"Encountered {len(errors)} errors during data collection")

        logger.info(f"Unique nodes: {len(all_nodes)}, Total edges: {len(all_edges)}")

        # Create constraints
        try:
            self.client.execute(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
                database=db_name
            )
        except Exception as e:
            logger.warning(f"Could not create constraint: {e}")

        # Load nodes in batches
        nodes_list = list(all_nodes.values())
        for i in range(0, len(nodes_list), self.batch_size):
            batch = nodes_list[i:i + self.batch_size]
            self.client.execute(
                """
                UNWIND $nodes AS node
                MERGE (n:Entity {id: node.id})
                SET n.label = node.label,
                    n.question_ids = node.question_ids,
                    n += node.properties
                """,
                {"nodes": batch},
                database=db_name
            )
            logger.debug(f"Loaded nodes: {min(i + self.batch_size, len(nodes_list))}/{len(nodes_list)}")

        # Add secondary labels using APOC
        try:
            self.client.execute(
                """
                MATCH (n:Entity)
                WHERE n.label IS NOT NULL
                WITH n, n.label AS lbl
                CALL apoc.create.addLabels(n, [lbl]) YIELD node
                RETURN count(node)
                """,
                database=db_name
            )
        except Exception as e:
            logger.warning(f"Could not add secondary labels (APOC may not be installed): {e}")

        # Load edges in batches
        for i in range(0, len(all_edges), self.batch_size):
            batch = all_edges[i:i + self.batch_size]
            try:
                self.client.execute(
                    """
                    UNWIND $edges AS edge
                    MATCH (source:Entity {id: edge.source})
                    MATCH (target:Entity {id: edge.target})
                    CALL apoc.create.relationship(source, edge.type, edge.properties, target) YIELD rel
                    RETURN count(rel)
                    """,
                    {"edges": batch},
                    database=db_name
                )
            except Exception as e:
                # Fallback to simple MERGE if APOC not available
                logger.warning(f"APOC not available, using fallback: {e}")
                self.client.execute(
                    """
                    UNWIND $edges AS edge
                    MATCH (source:Entity {id: edge.source})
                    MATCH (target:Entity {id: edge.target})
                    MERGE (source)-[r:RELATED]->(target)
                    SET r.type = edge.type, r += edge.properties
                    """,
                    {"edges": batch},
                    database=db_name
                )
            logger.debug(f"Loaded edges: {min(i + self.batch_size, len(all_edges))}/{len(all_edges)}")

        # Verify
        stats = self.client.get_database_info(db_name)
        logger.info(f"LPG loaded: {stats['node_count']} nodes, {stats['edge_count']} relationships")
        return stats

    def load_rdf_data(self, df: pd.DataFrame, db_name: str = "finderrdf") -> dict:
        """
        Load RDF triples into Neo4j using n10s-style representation.

        Args:
            df: DataFrame with rdf_triples column
            db_name: Target database name

        Returns:
            Statistics dictionary
        """
        if not self.client:
            raise Neo4jConnectionError("Not connected to Neo4j")

        logger.info(f"Loading RDF data into {db_name}")

        # Collect all triples
        all_triples: list[dict[str, Any]] = []
        errors = []

        for idx, row in df.iterrows():
            question_id = row["_id"]

            try:
                triples = parse_json_field(row["rdf_triples"])

                if not isinstance(triples, list):
                    continue

                for triple in triples:
                    if not isinstance(triple, dict):
                        continue
                    if not triple.get("subject") or not triple.get("object"):
                        continue

                    triple_data = {
                        "subject": triple["subject"],
                        "predicate": triple.get("predicate", "related_to"),
                        "object": triple["object"],
                        "is_literal": triple.get("is_literal", False),
                        "question_id": question_id
                    }
                    all_triples.append(triple_data)

            except Exception as e:
                errors.append(f"Row {idx}: {e}")
                if len(errors) <= 5:
                    logger.warning(f"Error processing row {idx}: {e}")

        logger.info(f"Total triples: {len(all_triples)}")

        # Create constraints
        try:
            self.client.execute(
                "CREATE CONSTRAINT resource_uri IF NOT EXISTS FOR (n:Resource) REQUIRE n.uri IS UNIQUE",
                database=db_name
            )
        except Exception as e:
            logger.warning(f"Could not create constraint: {e}")

        # Process triples in batches
        for i in range(0, len(all_triples), self.batch_size):
            batch = all_triples[i:i + self.batch_size]

            # Separate literal and non-literal triples
            literal_triples = [t for t in batch if t["is_literal"]]
            resource_triples = [t for t in batch if not t["is_literal"]]

            # For resource-to-resource relationships
            if resource_triples:
                try:
                    self.client.execute(
                        """
                        UNWIND $triples AS t
                        MERGE (s:Resource {uri: t.subject})
                        MERGE (o:Resource {uri: t.object})
                        WITH s, o, t
                        CALL apoc.create.relationship(s, t.predicate, {question_id: t.question_id}, o) YIELD rel
                        RETURN count(rel)
                        """,
                        {"triples": resource_triples},
                        database=db_name
                    )
                except Exception as e:
                    # Fallback without APOC
                    logger.warning(f"APOC not available for RDF, using fallback: {e}")
                    self.client.execute(
                        """
                        UNWIND $triples AS t
                        MERGE (s:Resource {uri: t.subject})
                        MERGE (o:Resource {uri: t.object})
                        MERGE (s)-[r:TRIPLE]->(o)
                        SET r.predicate = t.predicate, r.question_id = t.question_id
                        """,
                        {"triples": resource_triples},
                        database=db_name
                    )

            # For literal values
            if literal_triples:
                try:
                    self.client.execute(
                        """
                        UNWIND $triples AS t
                        MERGE (s:Resource {uri: t.subject})
                        WITH s, t
                        CALL apoc.create.setProperty(s, t.predicate, t.object) YIELD node
                        RETURN count(node)
                        """,
                        {"triples": literal_triples},
                        database=db_name
                    )
                except Exception as e:
                    logger.warning(f"Could not set literal properties: {e}")

            logger.debug(f"Loaded triples: {min(i + self.batch_size, len(all_triples))}/{len(all_triples)}")

        # Verify
        stats = self.client.get_database_info(db_name)
        logger.info(f"RDF loaded: {stats['node_count']} nodes, {stats['edge_count']} relationships")
        return stats

    def load_questions_metadata(self, df: pd.DataFrame, db_name: str) -> int:
        """
        Load question metadata as separate nodes.

        Args:
            df: DataFrame with question data
            db_name: Target database name

        Returns:
            Number of questions loaded
        """
        if not self.client:
            raise Neo4jConnectionError("Not connected to Neo4j")

        logger.info(f"Loading question metadata into {db_name}")

        questions = []
        for _, row in df.iterrows():
            questions.append({
                "id": row["_id"],
                "text": row["text"],
                "category": row.get("category", ""),
                "answer": row["answer"],
                "type": row.get("type", ""),
                "reasoning": row.get("reasoning", "")
            })

        # Create constraint
        try:
            self.client.execute(
                "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (n:Question) REQUIRE n.id IS UNIQUE",
                database=db_name
            )
        except Exception as e:
            logger.warning(f"Could not create constraint: {e}")

        # Load in batches
        for i in range(0, len(questions), self.batch_size):
            batch = questions[i:i + self.batch_size]
            self.client.execute(
                """
                UNWIND $questions AS q
                MERGE (n:Question {id: q.id})
                SET n.text = q.text,
                    n.category = q.category,
                    n.answer = q.answer,
                    n.type = q.type,
                    n.reasoning = q.reasoning
                """,
                {"questions": batch},
                database=db_name
            )

        logger.info(f"Loaded {len(questions)} questions")
        return len(questions)


def main():
    """Main execution function."""
    # Setup logging
    setup_logging(level="INFO")
    logger.info("Starting FinDER KG data loading")

    # Configuration
    config = ExperimentConfig()
    config.validate()

    logger.info(f"Neo4j URI: {config.neo4j.uri}")
    logger.info(f"Parquet path: {config.parquet_path}")

    # Load data
    with FinDERDataLoader(config) as loader:
        # Load parquet
        df = loader.load_parquet(config.parquet_path)
        logger.info(f"Loaded {len(df)} rows from parquet")

        # Create and populate LPG database
        loader.create_database("finderlpg")
        loader.clear_database("finderlpg")
        loader.load_lpg_data(df, "finderlpg")
        loader.load_questions_metadata(df, "finderlpg")

        # Create and populate RDF database
        loader.create_database("finderrdf")
        loader.clear_database("finderrdf")
        loader.load_rdf_data(df, "finderrdf")
        loader.load_questions_metadata(df, "finderrdf")

    logger.info("Data loading completed successfully")
    logger.info("Databases created:")
    logger.info("  - finderlpg: LPG representation")
    logger.info("  - finderrdf: RDF triple representation")


if __name__ == "__main__":
    main()
