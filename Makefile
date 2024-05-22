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
build:
	docker build . -t pyronear/pyro-engine:python3.8.1-slim
	docker build . -t pyronear/pyro-engine:latest

# Run the engine wrapper
run:
	bash scripts/setup-docker-compose.sh
	docker build . -t pyronear/pyro-engine:latest
	docker compose up -d
	rm docker-compose.yml.bak

# Get log from engine wrapper
log:
	docker logs -f --tail 50 pyro-engine-run

# Stop the engine wrapper
stop:
	docker compose down
