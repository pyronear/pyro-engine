![PyroNear Logo](docs/source/_static/img/pyronear-logo-dark.png)

<p align="center">
    <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
    <a href="https://app.codacy.com/gh/pyronear/pyro-engine?utm_source=github.com&utm_medium=referral&utm_content=pyronear/pyro-engine&utm_campaign=Badge_Grade_Settings">
        <img src="https://api.codacy.com/project/badge/Grade/d7f62736901d4e5c97c744411d8e02e3"/></a>
    <a href="https://github.com/pyronear/pyro-engine/actions?query=workflow%3Apython-package">
        <img src="https://github.com/pyronear/pyro-engine/workflows/python-package/badge.svg" /></a>
    <a href="https://codecov.io/gh/pyronear/pyro-engine">
      <img src="https://codecov.io/gh/pyronear/pyro-engine/branch/master/graph/badge.svg" />
    </a>
    <a href="https://pyronear.github.io/pyro-engine">
  		<img src="https://img.shields.io/badge/docs-available-blue.svg" /></a>
</p>



# pyroengine: Deploy Pyronear wildfire detection

The increasing adoption of mobile phones have significantly shortened the time required for firefighting agents to be alerted of a starting wildfire. In less dense areas, limiting and minimizing this duration remains critical to preserve forest areas.

![pyrovision](https://github.com/pyronear/pyro-vision) aims at providing the means to create a wildfire early detection system with state-of-the-art performances at minimal deployment costs.

pyroengine aims to deploy pyrovision wildfire detection system



## Table of Contents

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Documentation](#documentation)
* [Contributing](#contributing)
* [Credits](#credits)
* [License](#license)



## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)

### Installation

You can install the package using [pypi](https://pypi.org/project/pyronear/) as follows:

```shell
pip install pyroengine
```
### Environment files 

The `pyroengine/pi_utils/python.env` file must contain:
- `WEBSERVER_IP`: the IP address of the main rpi once it is installed on site
- `WEBSERVER_PORT`: the port exposed on the main rpi for the local webserver

### Test Engine

You can test to run a prediction using our Pyronear Engine using the following:

```shell
from pyroengine.engine import PyronearEngine
from PIL import Image

engine = PyronearEngine()

im = Image.open("path/to/your/image.jpg").convert('RGB')

prediction = engine.predict(image) 
```

This is a quick demo without api setup, so without sending the alert

## Documentation

The full package documentation is available [here](https://pyronear.github.io/pyro-engine/) for detailed specifications. The documentation was built with [Sphinx](https://www.sphinx-doc.org) using a [theme](https://github.com/readthedocs/sphinx_rtd_theme) provided by [Read the Docs](https://readthedocs.org).



## Contributing

Please refer to `CONTRIBUTING` if you wish to contribute to this project.



## Credits

This project is developed and maintained by the repo owner and volunteers from [Data for Good](https://dataforgood.fr/).



## License

Distributed under the Apache 2 License. See `LICENSE` for more information.
