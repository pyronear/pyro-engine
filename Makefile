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

# Build the docker
docker-pkg:
	docker build . -t pyroengine:python3.8.1-slim

# Run the engine wrapper
run:
	docker-compose -f src/docker-compose.yml up -d --build

# Stop the engine wrapper
stop:
	docker-compose -f src/docker-compose.yml down
