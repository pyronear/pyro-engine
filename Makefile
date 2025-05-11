# This target runs checks on all files
quality:
	ruff format --check --diff .
	ruff check --diff .
	mypy

# This target auto-fixes lint issues where possible
style:
	ruff format --check --diff .
	ruff check --diff .

# Run tests for the library
test:
	coverage run -m pytest tests/

# Build documentation for current version
single-docs:
	sphinx-build docs/source docs/_build -a


# update requirements.txt
lock:
	poetry lock
	poetry export -f requirements.txt --without-hashes --output requirements.txt

# Build the docker
build-app:
	docker build . -t pyronear/pyro-engine:latest

build-lib:
	pip install -e .

build-optional-lib:
	pip install -e .[test]
	pip install -e .[quality]
	pip install -e .[docs]
	pip install -e .[dev]

# Run the engine wrapper
run:
	docker build . -t pyronear/pyro-engine:latest
	docker compose up -d

# Get log from engine wrapper
log: 
	docker logs -f --tail 50 engine

# Get log from live_stream wrapper
log-st: 
	docker logs -f --tail 50 live_stream

# Stop the engine wrapper
stop:
	docker compose down
