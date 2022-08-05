![PyroNear Logo](docs/source/_static/img/pyronear-logo-dark.png)

<p align="center">
  <a href="https://github.com/pyronear/pyro-engine/actions?query=workflow%3Abuilds">
    <img alt="CI Status" src="https://img.shields.io/github/workflow/status/pyronear/pyro-engine/builds?label=CI&logo=github&style=flat-square">
  </a>
  <a href="https://pyronear.org/pyro-engine">
    <img src="https://img.shields.io/github/workflow/status/pyronear/pyro-engine/docs?label=docs&logo=read-the-docs&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/pyronear/pyro-engine">
    <img src="https://img.shields.io/codecov/c/github/pyronear/pyro-engine.svg?logo=codecov&style=flat-square" alt="Test coverage percentage">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://www.codacy.com/gh/pyronear/pyro-engine/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pyronear/pyro-engine&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/108f5fe8a7ac4f40a7bbd1985e26d5f9"/></a>
</p>
<p align="center">
  <a href="https://pypi.org/project/pyroengine/">
    <img src="https://img.shields.io/pypi/v/pyroengine.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPi Status">
  </a>
  <a href="https://hub.docker.com/repository/docker/pyronear/pyro-engine">
    <img alt="DockerHub version" src="https://img.shields.io/docker/v/pyronear/pyro-engine?arch=arm64&label=Docker&logo=Docker&logoColor=white">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/pyroengine.svg?style=flat-square" alt="pyversions">
  <img src="https://img.shields.io/pypi/l/pyroengine.svg?style=flat-square" alt="license">
</p>


# PyroEngine: Wildfire detection on edge devices

PyroEngine provides a high-level interface to use Deep learning models in production while being connected to the alert API.

## Quick Tour

### Running your engine locally

You can use the library like any other python package to detect wildfires as follows:

```python
from pyroengine.core import Engine
from PIL import Image

engine = Engine("pyronear/rexnet1_3x")

im = Image.open("path/to/your/image.jpg").convert('RGB')

prediction = engine.predict(image) 
```

## Setup

Python 3.6 (or higher) and [pip](https://pip.pypa.io/en/stable/)/[conda](https://docs.conda.io/en/latest/miniconda.html) are required to install PyroVision.

### Stable release

You can install the last stable release of the package using [pypi](https://pypi.org/project/pyroengine/) as follows:

```shell
pip install pyroengine
```

### Developer installation

Alternatively, if you wish to use the latest features of the project that haven't made their way to a release yet, you can install the package from source:

```shell
git clone https://github.com/pyronear/pyro-engine.git
pip install -e pyro-engine/.
```

### Full docker orchestration

Finally, you will need a `.env` file to enable camera & Alert API interactions. Your file should include a few mandatory entries:
```
API_URL=http://my-api.myhost.com
LAT=48.88
LON=2.38
CAM_USER=my_dummy_login
CAM_PWD=my_dummy_pwd
```

Additionally, you'll need a `./data` folder which contains:
- `credentials.json`: a dictionary with the IP address of your cameras as key, and dictionary with entries `login` & `password` for their Alert API credentials
- `model.onnx`: optional, will overrides the model weights download from HuggingFace Hub
- `config.json`: optional, will overrides the model config download from HuggingFace Hub

## Documentation

The full package documentation is available [here](https://pyronear.org/pyro-engine/) for detailed specifications.

## Contributing

Please refer to [`CONTRIBUTING`](CONTRIBUTING.md) if you wish to contribute to this project.



## Credits

This project is developed and maintained by the repo owner and volunteers from [Data for Good](https://dataforgood.fr/).



## License

Distributed under the Apache 2 License. See [`LICENSE`](LICENSE) for more information.
