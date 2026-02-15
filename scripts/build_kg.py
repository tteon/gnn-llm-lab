#!/usr/bin/env python3
"""
Stage 2: Build Knowledge Graph from FinDER dataset using LLM entity/relationship extraction.

Reads data/raw/FinDER.parquet, calls LLM API for entity extraction and relationship
linking, converts to LPG + RDF formats, and outputs data/raw/FinDER_KG_Merged.parquet.

Supports checkpointing and resume for long-running extraction.

Usage:
    uv run python scripts/build_kg.py
    uv run python scripts/build_kg.py --sample-size 10     # test subset
    uv run python scripts/build_kg.py --resume              # resume from checkpoint
    uv run python scripts/build_kg.py --model gpt-4o --rpm 100
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

try:
    import opik
    from opik import opik_context

    _HAS_OPIK = True
except ImportError:
    _HAS_OPIK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FIBO predicate mapping: LPG edge type → RDF predicate
FIBO_PREDICATE_MAP = {
    "OWNS": "fibo-fnd-oac-own:owns",
    "INCURRED": "fibo-fnd-acc-acc:incurred",
    "REPORTED": "fibo-fnd-acc-acc:reportedIn",
    "HAS_VALUE": "fibo-fnd-acc-acc:hasValue",
    "COMPLY_WITH": "fibo-fnd-acc-std:conformsTo",
    "INCREASED_BY": "fibo-fnd-acc-acc:increasedBy",
    "DECREASED_BY": "fibo-fnd-acc-acc:decreasedBy",
    "LOCATED_IN": "fibo-fnd-law-jur:hasJurisdiction",
    "WORKS_FOR": "fibo-fnd-pty-pty:isPartyTo",
    "PART_OF": "fibo-fnd-rel-rel:isPartOf",
    "RELATED_TO": "fibo-fnd-rel-rel:isRelatedTo",
    "HAPPENED_ON": "fibo-fnd-dt-fd:hasDate",
}

FINANCIAL_CATEGORIES = {"Financials", "Accounting", "Shareholder return"}


# ---------------------------------------------------------------------------
# Prompt Manager
# ---------------------------------------------------------------------------


class PromptManager:
    """Load YAML prompt templates and render with Jinja2."""

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._templates: Dict[str, Dict[str, Template]] = {}
        self._load_all()

    def _load_all(self) -> None:
        for yaml_path in self.prompts_dir.glob("*.yaml"):
            name = yaml_path.stem
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            # Find the top-level key (extraction or linking)
            top_key = next(iter(data))
            section = data[top_key]
            self._templates[name] = {
                "system": Template(section["system"]),
                "user": Template(section["user"]),
            }

    def render(self, template_name: str, **kwargs) -> Tuple[str, str]:
        """Render a prompt template. Returns (system_msg, user_msg)."""
        t = self._templates[template_name]
        return t["system"].render(**kwargs), t["user"].render(**kwargs)

    def get_extraction_prompt(self, category: str, text: str) -> Tuple[str, str]:
        """Select extraction prompt based on category."""
        name = "fibo_extraction" if category in FINANCIAL_CATEGORIES else "base_extraction"
        return self.render(name, text=text)

    def get_linking_prompt(
        self, category: str, text: str, entities: str
    ) -> Tuple[str, str]:
        """Select linking prompt based on category."""
        name = "fibo_linking" if category in FINANCIAL_CATEGORIES else "base_linking"
        return self.render(name, text=text, entities=entities)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class ExtractionCheckpoint:
    """JSONL-based resumable checkpoint."""

    def __init__(self, checkpoint_dir: str = "data/intermediate"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / "kg_extraction.jsonl"
        self.progress_path = self.dir / "progress.json"

    def get_completed_ids(self) -> Set[str]:
        """Read all completed sample IDs."""
        ids = set()
        if self.jsonl_path.exists():
            with open(self.jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            ids.add(data["_id"])
                        except (json.JSONDecodeError, KeyError):
                            continue
        return ids

    def append(self, result: dict) -> None:
        """Append one result to JSONL."""
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def save_progress(self, processed: int, total: int, errors: int) -> None:
        with open(self.progress_path, "w") as f:
            json.dump(
                {"processed": processed, "total": total, "errors": errors},
                f, indent=2,
            )

    def load_all_results(self) -> Dict[str, dict]:
        """Load all results keyed by _id."""
        results = {}
        if self.jsonl_path.exists():
            with open(self.jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            results[data["_id"]] = data
                        except (json.JSONDecodeError, KeyError):
                            continue
        return results


# ---------------------------------------------------------------------------
# KG Builder
# ---------------------------------------------------------------------------


class KGBuilder:
    """Build Knowledge Graph from text using LLM extraction."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        api_base_url: Optional[str] = None,
        prompts_dir: str = "prompts",
        requests_per_minute: int = 500,
        max_retries: int = 3,
        use_opik: bool = False,
        opik_project: str = "FinDER_KG_Build",
    ):
        client_kwargs = {"api_key": api_key}
        if api_base_url:
            client_kwargs["base_url"] = api_base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.prompts = PromptManager(prompts_dir)
        self.rpm = requests_per_minute
        self.max_retries = max_retries
        self._request_interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0
        self._use_opik = use_opik and _HAS_OPIK

        # Wrap methods with Opik tracing
        if self._use_opik:
            self.process_sample = opik.track(
                name="kg_build_pipeline", project_name=opik_project
            )(self.process_sample)
            self.extract_entities = opik.track(
                name="entity_extraction"
            )(self.extract_entities)
            self.extract_relationships = opik.track(
                name="entity_linking"
            )(self.extract_relationships)
            self._to_rdf_triples = opik.track(
                name="rdf_conversion"
            )(self._to_rdf_triples)

    def _rate_limit(self) -> None:
        """Simple rate limiting via sleep."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)
        self._last_request_time = time.time()

    def _update_span(self, **metadata):
        """Update current Opik span metadata (no-op when tracing disabled)."""
        if self._use_opik:
            opik_context.update_current_span(metadata=metadata)

    def _call_llm(self, system: str, user: str) -> str:
        """Call LLM API with retry. Returns raw text response."""
        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries:
                    delay = min(2.0 * (2 ** (attempt - 1)), 30.0)
                    print(f"  API error (attempt {attempt}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise

    @staticmethod
    def _parse_llm_json(raw: str) -> dict:
        """Parse JSON from LLM output with common issue handling."""
        # Strip markdown code fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            text = "\n".join(lines)

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Remove trailing commas before ] or }
        cleaned = re.sub(r",\s*([\]}])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}

    def extract_entities(self, text: str, category: str) -> List[dict]:
        """Extract entities from text. Returns list of node dicts."""
        system, user = self.prompts.get_extraction_prompt(category, text)
        raw = self._call_llm(system, user)
        parsed = self._parse_llm_json(raw)
        nodes = parsed.get("nodes", [])

        # Ensure each node has required fields
        valid_nodes = []
        for n in nodes:
            if isinstance(n, dict) and n.get("id") and n.get("label"):
                if "properties" not in n:
                    n["properties"] = {}
                if "name" not in n["properties"]:
                    n["properties"]["name"] = n["id"]
                valid_nodes.append(n)

        self._update_span(
            category=category,
            prompt_type="fibo" if category in FINANCIAL_CATEGORIES else "base",
            entity_count=len(valid_nodes),
        )
        return valid_nodes

    def extract_relationships(
        self, text: str, category: str, entities: List[dict]
    ) -> List[dict]:
        """Extract relationships given text and entities. Returns list of edge dicts."""
        entities_str = json.dumps(entities, ensure_ascii=False)
        system, user = self.prompts.get_linking_prompt(category, text, entities_str)
        raw = self._call_llm(system, user)
        parsed = self._parse_llm_json(raw)
        rels = parsed.get("relationships", [])

        # Validate and normalize
        valid_edges = []
        for r in rels:
            if isinstance(r, dict) and r.get("source") and r.get("target"):
                edge = {
                    "source": r["source"],
                    "target": r["target"],
                    "type": r.get("type", "RELATED_TO"),
                    "properties": r.get("properties", {}),
                }
                valid_edges.append(edge)

        self._update_span(
            category=category,
            prompt_type="fibo" if category in FINANCIAL_CATEGORIES else "base",
            relationship_count=len(valid_edges),
        )
        return valid_edges

    @staticmethod
    def _sanitize_uri(name: str) -> str:
        """Sanitize a name for use in a URI."""
        s = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "unknown"

    @staticmethod
    def _to_rdf_triples(nodes: List[dict], edges: List[dict]) -> List[dict]:
        """Convert LPG nodes/edges to RDF triples with FIBO predicates."""
        triples = []

        # Build name → URI mapping
        name_to_uri = {}
        for n in nodes:
            name = n.get("properties", {}).get("name", n["id"])
            uri = f"ex:{KGBuilder._sanitize_uri(name)}"
            name_to_uri[name] = uri
            name_to_uri[n["id"]] = uri

            # rdf:type triple
            triples.append({
                "subject": uri,
                "predicate": "rdf:type",
                "object": f"ex:{n['label']}",
                "is_literal": False,
            })

            # Property triples
            for key, val in n.get("properties", {}).items():
                if key in ("name", "linked_id", "context"):
                    continue
                if val is not None and val != "":
                    triples.append({
                        "subject": uri,
                        "predicate": f"ex:{key}",
                        "object": str(val),
                        "is_literal": True,
                    })

        # Edge triples
        for e in edges:
            source_uri = name_to_uri.get(e["source"], f"ex:{KGBuilder._sanitize_uri(e['source'])}")
            target_uri = name_to_uri.get(e["target"], f"ex:{KGBuilder._sanitize_uri(e['target'])}")
            edge_type = e.get("type", "RELATED_TO").upper()
            predicate = FIBO_PREDICATE_MAP.get(edge_type, f"ex:{edge_type}")
            triples.append({
                "subject": source_uri,
                "predicate": predicate,
                "object": target_uri,
                "is_literal": False,
            })

        return triples

    def process_sample(self, row: dict) -> dict:
        """
        Full extraction pipeline for one sample.

        Args:
            row: Dict with at least '_id', 'text', 'category'

        Returns:
            Dict with _id, lpg_nodes, lpg_edges, rdf_triples (all JSON strings)
        """
        text = row["text"]
        category = row.get("category", "")

        # 1. Extract entities (1 API call)
        nodes = self.extract_entities(text, category)

        # 2. Extract relationships (1 API call)
        edges = self.extract_relationships(text, category, nodes) if nodes else []

        # 3. Convert to RDF (deterministic)
        rdf_triples = self._to_rdf_triples(nodes, edges)

        self._update_span(
            question_id=row["_id"],
            category=category,
            node_count=len(nodes),
            edge_count=len(edges),
            rdf_triple_count=len(rdf_triples),
        )

        return {
            "_id": row["_id"],
            "lpg_nodes": json.dumps(nodes, ensure_ascii=False),
            "lpg_edges": json.dumps(edges, ensure_ascii=False),
            "rdf_triples": json.dumps(rdf_triples, ensure_ascii=False),
        }

    def run(
        self,
        df: pd.DataFrame,
        checkpoint: ExtractionCheckpoint,
        resume: bool = False,
    ) -> Dict[str, dict]:
        """
        Process all samples with progress tracking and checkpointing.

        Returns:
            Dict mapping _id → extraction result
        """
        # Get already-processed IDs for resume
        completed_ids = checkpoint.get_completed_ids() if resume else set()
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} samples already processed")

        # Filter to unprocessed
        pending = df[~df["_id"].isin(completed_ids)]
        total = len(df)
        errors = 0

        print(f"Processing {len(pending)}/{total} samples (model: {self.model}, RPM: {self.rpm})")

        for i, (_, row) in enumerate(tqdm(
            pending.iterrows(), total=len(pending), desc="Extracting"
        )):
            try:
                result = self.process_sample(row.to_dict())
                checkpoint.append(result)
            except Exception as e:
                errors += 1
                print(f"\n  Error on {row['_id']}: {e}")
                # Save empty result so we don't retry on resume
                checkpoint.append({
                    "_id": row["_id"],
                    "lpg_nodes": "[]",
                    "lpg_edges": "[]",
                    "rdf_triples": "[]",
                    "error": str(e),
                })

            # Save progress periodically
            processed = len(completed_ids) + i + 1
            if (i + 1) % 50 == 0:
                checkpoint.save_progress(processed, total, errors)

        checkpoint.save_progress(len(completed_ids) + len(pending), total, errors)
        print(f"\nDone: {total} total, {errors} errors")

        return checkpoint.load_all_results()


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_and_save(
    raw_df: pd.DataFrame,
    results: Dict[str, dict],
    output_path: str,
) -> pd.DataFrame:
    """Merge extraction results into the raw DataFrame and save as parquet."""
    # Add extraction columns
    lpg_nodes_col = []
    lpg_edges_col = []
    rdf_triples_col = []

    for _, row in raw_df.iterrows():
        qid = row["_id"]
        r = results.get(qid, {})
        lpg_nodes_col.append(r.get("lpg_nodes", "[]"))
        lpg_edges_col.append(r.get("lpg_edges", "[]"))
        rdf_triples_col.append(r.get("rdf_triples", "[]"))

    raw_df = raw_df.copy()
    raw_df["lpg_nodes"] = lpg_nodes_col
    raw_df["lpg_edges"] = lpg_edges_col
    raw_df["rdf_triples"] = rdf_triples_col

    # Validate required columns for load_finder_kg.py
    required = ["_id", "text", "answer", "lpg_nodes", "lpg_edges", "rdf_triples"]
    missing = [c for c in required if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(output, index=False)

    # Stats
    has_nodes = raw_df["lpg_nodes"].apply(lambda x: x != "[]").sum()
    has_edges = raw_df["lpg_edges"].apply(lambda x: x != "[]").sum()
    has_triples = raw_df["rdf_triples"].apply(lambda x: x != "[]").sum()
    print(f"\nMerged parquet saved: {output}")
    print(f"  Total rows: {len(raw_df)}")
    print(f"  Rows with LPG nodes: {has_nodes}")
    print(f"  Rows with LPG edges: {has_edges}")
    print(f"  Rows with RDF triples: {has_triples}")

    return raw_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build Knowledge Graph from FinDER dataset using LLM extraction"
    )
    parser.add_argument(
        "--input", type=str, default="data/raw/FinDER.parquet",
        help="Input parquet (from download_dataset.py)",
    )
    parser.add_argument(
        "--output", type=str, default="data/raw/FinDER_KG_Merged.parquet",
        help="Output merged parquet",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="data/intermediate",
        help="Checkpoint directory for resume",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Process only N samples (for testing)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model for extraction (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--rpm", type=int, default=500,
        help="Requests per minute rate limit (default: 500)",
    )
    parser.add_argument(
        "--prompts-dir", type=str, default="prompts",
        help="Directory containing prompt YAML files",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip extraction, only merge existing checkpoint into parquet",
    )
    parser.add_argument(
        "--no-opik", action="store_true",
        help="Disable Opik tracing",
    )
    parser.add_argument(
        "--opik-project", type=str, default="FinDER_KG_Build",
        help="Opik project name (default: FinDER_KG_Build)",
    )

    args = parser.parse_args()
    load_dotenv()

    # Configure Opik tracing
    use_opik = not args.no_opik and _HAS_OPIK
    if use_opik:
        opik.configure(use_local=True)
        print(f"Opik tracing enabled (project: {args.opik_project})")

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        print("Run 'uv run python scripts/download_dataset.py' first.")
        return

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    if args.sample_size:
        df = df.head(args.sample_size)
        print(f"Using first {len(df)} samples")

    checkpoint = ExtractionCheckpoint(args.checkpoint_dir)

    if not args.merge_only:
        # Validate API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not set in .env")
            return

        api_base = os.environ.get("OPENAI_API_BASE_URL")

        builder = KGBuilder(
            api_key=api_key,
            model=args.model,
            api_base_url=api_base,
            prompts_dir=args.prompts_dir,
            requests_per_minute=args.rpm,
            use_opik=use_opik,
            opik_project=args.opik_project,
        )

        results = builder.run(df, checkpoint, resume=args.resume)
    else:
        results = checkpoint.load_all_results()
        print(f"Loaded {len(results)} results from checkpoint")

    # Merge and save
    merge_and_save(df, results, args.output)


if __name__ == "__main__":
    main()
