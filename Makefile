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
	$(info If you need to update the client change the hash in the .toml and use make lock before)
	docker build . -t pyronear/pyro-engine:latest

# Build the light docker
build-cpu-app: 
	$(info The pyro-client version is hardcoded in the Dockerfile)
	docker build . -t pyronear/pyro-engine:latest -f Dockerfile-cpu

build-lib:
	pip install -e .
	python setup.py sdist bdist_wheel

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
	docker logs -f --tail 50 pyro-engine-run

# Stop the engine wrapper
stop:
	docker compose down
