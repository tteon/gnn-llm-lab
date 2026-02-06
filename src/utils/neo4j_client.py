"""
Robust Neo4j client with retry logic and connection management.

Features:
- Automatic retry with exponential backoff
- Connection pooling
- Context manager support
- Comprehensive error handling
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import (
    ServiceUnavailable,
    SessionExpired,
    TransientError,
    AuthError,
    ClientError,
)

from .config import Neo4jConfig
from .exceptions import Neo4jConnectionError
from .logging_config import get_logger

logger = get_logger("neo4j")


class Neo4jClient:
    """
    Robust Neo4j client with automatic retry and connection management.

    Usage:
        # As context manager (recommended)
        with Neo4jClient(config) as client:
            data = client.query("MATCH (n) RETURN n LIMIT 10")

        # Manual management
        client = Neo4jClient(config)
        client.connect()
        try:
            data = client.query(...)
        finally:
            client.close()
    """

    # Exceptions that should trigger retry
    TRANSIENT_ERRORS = (ServiceUnavailable, SessionExpired, TransientError)

    def __init__(self, config: Neo4jConfig):
        """
        Initialize client with configuration.

        Args:
            config: Neo4j configuration object
        """
        self.config = config
        self._driver: Optional[Driver] = None
        self._connected = False

    @property
    def driver(self) -> Driver:
        """Get the driver, raising error if not connected."""
        if self._driver is None:
            raise Neo4jConnectionError(
                "Not connected to Neo4j. Call connect() first.",
                uri=self.config.uri
            )
        return self._driver

    @property
    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        return self._connected and self._driver is not None

    def connect(self) -> "Neo4jClient":
        """
        Establish connection to Neo4j with retry.

        Returns:
            Self for method chaining

        Raises:
            Neo4jConnectionError: If connection fails after all retries
        """
        if self._connected:
            return self

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(f"Connecting to Neo4j (attempt {attempt}/{self.config.max_retries})")

                self._driver = GraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.user, self.config.password),
                    max_connection_lifetime=self.config.max_connection_lifetime,
                    max_connection_pool_size=self.config.max_connection_pool_size,
                    connection_timeout=self.config.connection_timeout,
                )

                # Verify connectivity
                self._driver.verify_connectivity()
                self._connected = True

                logger.info(f"Connected to Neo4j: {self.config.uri}")
                return self

            except AuthError as e:
                # Authentication errors should not be retried
                raise Neo4jConnectionError(
                    "Authentication failed",
                    uri=self.config.uri,
                    attempts=attempt,
                    original_error=e
                )

            except self.TRANSIENT_ERRORS as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = min(
                        self.config.retry_delay * (2 ** (attempt - 1)),
                        self.config.retry_max_delay
                    )
                    logger.warning(
                        f"Connection attempt {attempt} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error connecting to Neo4j: {e}")
                break

        raise Neo4jConnectionError(
            f"Failed to connect after {self.config.max_retries} attempts",
            uri=self.config.uri,
            attempts=self.config.max_retries,
            original_error=last_error
        )

    def close(self) -> None:
        """Close the connection."""
        if self._driver:
            try:
                self._driver.close()
                logger.debug("Neo4j connection closed")
            except Exception as e:
                logger.warning(f"Error closing Neo4j connection: {e}")
            finally:
                self._driver = None
                self._connected = False

    def __enter__(self) -> "Neo4jClient":
        """Context manager entry."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    @contextmanager
    def session(self, database: Optional[str] = None) -> Generator[Session, None, None]:
        """
        Get a session with automatic cleanup.

        Args:
            database: Database name (uses config default if None)

        Yields:
            Neo4j session
        """
        db = database or self.config.database
        session = self.driver.session(database=db)
        try:
            yield session
        finally:
            session.close()

    def query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query with retry logic.

        Args:
            cypher: Cypher query string
            parameters: Query parameters
            database: Database name (uses config default if None)

        Returns:
            List of result records as dictionaries

        Raises:
            Neo4jConnectionError: If query fails after all retries
        """
        parameters = parameters or {}
        last_error = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                with self.session(database) as session:
                    result = session.run(cypher, parameters)
                    return [dict(record) for record in result]

            except self.TRANSIENT_ERRORS as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = min(
                        self.config.retry_delay * (2 ** (attempt - 1)),
                        self.config.retry_max_delay
                    )
                    logger.warning(
                        f"Query attempt {attempt} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            except ClientError as e:
                # Client errors (syntax, constraint violations) should not be retried
                logger.error(f"Cypher error: {e}")
                raise Neo4jConnectionError(
                    f"Cypher query failed: {e}",
                    database=database or self.config.database,
                    original_error=e
                )

        raise Neo4jConnectionError(
            f"Query failed after {self.config.max_retries} attempts",
            database=database or self.config.database,
            attempts=self.config.max_retries,
            original_error=last_error
        )

    def query_single(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a query expecting a single result.

        Args:
            cypher: Cypher query string
            parameters: Query parameters
            database: Database name

        Returns:
            Single result record or None
        """
        results = self.query(cypher, parameters, database)
        return results[0] if results else None

    def execute(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a write query and return summary.

        Args:
            cypher: Cypher query string
            parameters: Query parameters
            database: Database name

        Returns:
            Query summary with counters
        """
        parameters = parameters or {}

        with self.session(database) as session:
            result = session.run(cypher, parameters)
            summary = result.consume()
            counters = summary.counters

            return {
                "nodes_created": counters.nodes_created,
                "nodes_deleted": counters.nodes_deleted,
                "relationships_created": counters.relationships_created,
                "relationships_deleted": counters.relationships_deleted,
                "properties_set": counters.properties_set,
            }

    def execute_batch(
        self,
        cypher: str,
        batch_data: List[Dict[str, Any]],
        batch_key: str = "batch",
        database: Optional[str] = None,
        batch_size: int = 500,
    ) -> Dict[str, int]:
        """
        Execute a query in batches.

        Args:
            cypher: Cypher query with UNWIND $batch_key AS item
            batch_data: List of items to process
            batch_key: Parameter name for the batch
            database: Database name
            batch_size: Number of items per batch

        Returns:
            Aggregated summary counters
        """
        total_counters = {
            "nodes_created": 0,
            "nodes_deleted": 0,
            "relationships_created": 0,
            "relationships_deleted": 0,
            "properties_set": 0,
        }

        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}, items {i}-{i + len(batch)}")

            counters = self.execute(cypher, {batch_key: batch}, database)
            for key in total_counters:
                total_counters[key] += counters[key]

        return total_counters

    def get_database_info(self, database: Optional[str] = None) -> Dict[str, Any]:
        """Get database statistics."""
        db = database or self.config.database

        node_count = self.query_single(f"MATCH (n) RETURN count(n) as count", database=db)
        edge_count = self.query_single(f"MATCH ()-[r]->() RETURN count(r) as count", database=db)
        labels = self.query(f"CALL db.labels() YIELD label RETURN collect(label) as labels", database=db)
        rel_types = self.query(f"CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types", database=db)

        return {
            "database": db,
            "node_count": node_count["count"] if node_count else 0,
            "edge_count": edge_count["count"] if edge_count else 0,
            "labels": labels[0]["labels"] if labels else [],
            "relationship_types": rel_types[0]["types"] if rel_types else [],
        }

    def load_questions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load question data from database.

        Args:
            limit: Maximum number of questions to load

        Returns:
            List of question records
        """
        query = """
        MATCH (q:Question)
        RETURN q.id as id, q.text as text, q.answer as answer,
               q.category as category, q.type as type, q.reasoning as reasoning
        """
        if limit:
            query += f" LIMIT {limit}"

        return self.query(query)

    def get_subgraph(
        self,
        question_id: str,
        max_hops: int = 2,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get subgraph related to a question.

        Args:
            question_id: Question identifier
            max_hops: Maximum hops for expansion

        Returns:
            Dictionary with 'nodes' and 'edges' lists
        """
        query = """
        MATCH (e:Entity)
        WHERE $qid IN e.question_ids
        WITH collect(e) as seeds
        UNWIND seeds as seed
        OPTIONAL MATCH path = (seed)-[*1..%d]-(connected:Entity)
        WITH seeds + collect(DISTINCT connected) as all_raw
        WITH [n IN all_raw WHERE n IS NOT NULL] as all_nodes
        UNWIND all_nodes as n
        WITH collect(DISTINCT n) as all_nodes
        UNWIND all_nodes as n
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

        result = self.query_single(query, {"qid": question_id})

        if result:
            return {
                "nodes": [n for n in result["nodes"] if n],
                "edges": [e for e in result["edges"] if e.get("source") and e.get("target")]
            }
        return {"nodes": [], "edges": []}

    def get_all_entities(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all entity nodes.

        Args:
            limit: Maximum number of entities

        Returns:
            List of entity records
        """
        query = """
        MATCH (e:Entity)
        RETURN e.id as id, e.label as label,
               coalesce(e.name, e.id) as name,
               properties(e) as properties
        """
        if limit:
            query += f" LIMIT {limit}"

        return self.query(query)
