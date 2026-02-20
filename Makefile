.PHONY: all pipeline download build-kg load-neo4j build-pyg experiment setup clean-intermediate clean lint test

# Full pipeline: download → extract → merge → load Neo4j
pipeline: download build-kg load-neo4j
	uv run python scripts/generate_common_question_ids.py
	@echo "Pipeline complete. Run experiments: make experiment"

# Stage 1: Download FinDER dataset from HuggingFace
download:
	uv run python scripts/download_dataset.py

# Stage 2: Entity/relationship extraction → merged parquet
build-kg:
	uv run python scripts/build_kg.py

# Stage 2 (resume): Resume from checkpoint
build-kg-resume:
	uv run python scripts/build_kg.py --resume

# Stage 2 (merge only): Skip extraction, merge existing checkpoint
build-kg-merge:
	uv run python scripts/build_kg.py --merge-only

# Stage 3: Load parquet → Neo4j (finderlpg + finderrdf)
load-neo4j:
	docker-compose up -d
	@echo "Waiting for Neo4j to start..."
	sleep 10
	uv run python src/load_finder_kg.py

# Stage 4: Build PyG dataset (LPG + RDF dual-graph per question)
build-pyg:
	uv run python scripts/build_pyg_dataset.py

# Run Opik experiment
experiment:
	uv run python src/opik_experiment.py \
		--models llama8b --contexts none lpg rdf --sample-size 50

# First-time setup
setup:
	uv sync
	uv pip install datasets opik
	@echo "Edit .env with your API keys, then run: make pipeline"

# Clean intermediate checkpoint files
clean-intermediate:
	rm -rf data/intermediate/

# Clean all generated data
clean:
	rm -rf data/intermediate/
	rm -f data/raw/FinDER.parquet
	rm -f data/raw/FinDER_KG_Merged.parquet

# Lint
lint:
	uv run ruff check src/ scripts/
	uv run black --check src/ scripts/

# Test
test:
	uv run pytest tests/
