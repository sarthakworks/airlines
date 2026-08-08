# Convenience targets. Activate your venv first (source .venv/bin/activate).
.PHONY: help install data db-up db-down load index api ui eval health

help:
	@echo "install   - pip install -r requirements.txt"
	@echo "data      - download CSV + FAQ PDF into data/"
	@echo "db-up     - start local Postgres (docker compose)"
	@echo "db-down   - stop local Postgres"
	@echo "load      - load the flights CSV into Postgres"
	@echo "index     - build the RAG vector index from the FAQ PDF"
	@echo "api       - run the FastAPI backend on :8000"
	@echo "ui        - run the Streamlit frontend on :8501"
	@echo "eval      - run the evaluation harness"
	@echo "health    - curl the /health endpoint"

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

db-up:
	docker compose up -d

db-down:
	docker compose down

load:
	python scripts/load_db.py

index:
	python scripts/build_index.py

api:
	uvicorn app.api:app --reload --port 8000

ui:
	API_BASE_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py --server.port 8501

eval:
	python evaluation/evaluate.py

health:
	@curl -s http://localhost:8000/health | python -m json.tool
