# this target runs checks on all files
quality:
	isort . -c
	flake8
	mypy
	pydocstyle
	black --check .

# this target runs checks on all files and potentially modifies some of them
style:
	isort .
	black .

# Run tests for the library
test:
	coverage run -m pytest tests/

# Build documentation for current version
single-docs:
	sphinx-build docs/source docs/_build -a


# update requirements.txt
lock:
	cd src; poetry lock
	cd src; poetry export -f requirements.txt --without-hashes --output requirements.txt

# Build the docker
build-app:
	docker build . -t pyronear/pyro-engine:python3.8.1-slim
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
	cd src; poetry export -f requirements.txt --without-hashes --output requirements.txt
	docker build . -t pyronear/pyro-engine:latest
	docker compose up -d

# Get log from engine wrapper
log: 
	docker logs -f --tail 50 pyro-engine-run

# Stop the engine wrapper
stop:
	docker compose down
