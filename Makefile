PY := .venv/Scripts/python.exe
ifeq ($(OS),)
PY := .venv/bin/python
endif

.PHONY: help venv install corpus test lint run eval sweep docker up down clean

help:
	@echo "make install   - create .venv and install dependencies"
	@echo "make corpus    - generate the demo corpus"
	@echo "make test      - run the test suite"
	@echo "make run       - start the API on :8000"
	@echo "make eval      - evaluate the default configuration"
	@echo "make sweep     - run the ablation sweep into docs/RESULTS.md"
	@echo "make up        - docker compose up --build"

venv:
	python -m venv .venv

install: venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

corpus:
	$(PY) scripts/make_sample_corpus.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check app eval scripts tests

run:
	$(PY) -m uvicorn app.main:app --reload --port 8000

eval:
	$(PY) -m eval.run_eval

sweep:
	$(PY) -m eval.sweep --out docs/RESULTS.md --json-out eval-results/sweep.json

docker:
	docker build -t rag-assistant:latest .

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf data/index data/uploads .pytest_cache .ruff_cache **/__pycache__
