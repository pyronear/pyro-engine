![PyroNear Logo](docs/source/_static/img/pyronear-logo-dark.png)

<p align="center">
    <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
    <a href="https://app.codacy.com/gh/pyronear/pyro-engine?utm_source=github.com&utm_medium=referral&utm_content=pyronear/pyro-engine&utm_campaign=Badge_Grade_Settings">
        <img src="https://api.codacy.com/project/badge/Grade/d7f62736901d4e5c97c744411d8e02e3"/></a>
    <a href="https://github.com/pyronear/pyro-engine/actions?query=workflow%3Abuilds">
        <img src="https://github.com/pyronear/pyro-engine/workflows/builds/badge.svg" /></a>
    <a href="https://codecov.io/gh/pyronear/pyro-engine">
      <img src="https://codecov.io/gh/pyronear/pyro-engine/branch/master/graph/badge.svg" />
    </a>
    <a href="https://pyronear.github.io/pyro-engine">
  		<img src="https://img.shields.io/badge/docs-available-blue.svg" /></a>
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

The full package documentation is available [here](https://pyronear.github.io/pyro-engine/) for detailed specifications. The documentation was built with [Sphinx](https://www.sphinx-doc.org) using a [theme](https://github.com/readthedocs/sphinx_rtd_theme) provided by [Read the Docs](https://readthedocs.org).



## Contributing

Please refer to [`CONTRIBUTING`](CONTRIBUTING.md) if you wish to contribute to this project.



## Credits

This project is developed and maintained by the repo owner and volunteers from [Data for Good](https://dataforgood.fr/).



## License

Distributed under the Apache 2 License. See [`LICENSE`](LICENSE) for more information.
