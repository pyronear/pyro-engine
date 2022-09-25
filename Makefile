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
docker:
	docker build . -t pyronear/pyro-engine:python3.8.1-slim

# Run the engine wrapper
run:
	docker build . -t pyronear/pyro-engine:latest
	docker-compose up -d

# Stop the engine wrapper
stop:
	docker-compose down