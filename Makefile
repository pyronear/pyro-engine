# This target runs checks on all files
quality:
	ruff format --check --diff .
	ruff check --diff .
	mypy

# This target auto-fixes lint issues where possible
style:
	ruff format .
	ruff check --fix .

# Run tests for the library
test:
	coverage run -m pytest tests/

# Build documentation for current version
single-docs:
	sphinx-build docs/source docs/_build -a

# Update requirements.txt for the main project
lock:
	poetry lock
	poetry export -f requirements.txt --without-hashes --output requirements-all.txt
	grep 'git+' requirements-all.txt > requirements-git.txt || true
	grep -v 'git+' requirements-all.txt | grep -v 'file://' | grep -v '^opencv-python==' > requirements.txt
	rm requirements-all.txt

# Generate requirements and build camera API Docker image
build-api:
	poetry export -C pyro_camera_api -f requirements.txt --without-hashes --output pyro_camera_api/requirements.txt
	docker build -f pyro_camera_api/Dockerfile pyro_camera_api -t pyronear/pyro-camera-api:latest

# Build the engine Docker image
build-app:
	docker build . -t pyronear/pyro-engine:latest

build-lib:
	pip install -e .

build-optional-lib:
	pip install -e .[test]
	pip install -e .[quality]
	pip install -e .[docs]
	pip install -e .[dev]

# Pull latest images and run the stack
run:
	docker pull pyronear/pyro-engine:latest
	docker pull pyronear/pyro-camera-api:latest
	docker compose up -d

# Build images locally (no Docker Hub) and run the stack
run-local: build-api build-app
	docker compose up -d

# Get log from engine wrapper
log:
	docker logs -f --tail 50 engine

# Get log from camera API wrapper
log-api:
	docker logs -f --tail 50 pyro-camera-api

# Stop the stack
stop:
	docker compose down
