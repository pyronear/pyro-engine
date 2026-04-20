
************
Installation
************

This library requires `Python <https://www.python.org/downloads/>`_ 3.11 or higher.

``pyroengine`` is not published to PyPI because it depends on two in-tree
packages (``pyro_camera_api_client`` and ``pyro_predictor``) that live inside
this repository. Install by cloning the repository and using
`uv <https://docs.astral.sh/uv/>`_:

.. code:: bash

    git clone https://github.com/pyronear/pyro-engine.git
    cd pyro-engine
    uv sync
