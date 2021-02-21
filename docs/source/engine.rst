pyroengine.engine
=================

The models subpackage contains everuthing to manage the whole Fire Detection process by capturing and saving the image and
by predicting if there is a fire or not based on this image


.. currentmodule:: pyroengine.engine

Pyronear Predictor
------------------

The pyronear_predictor class that loads the last pytonear model and makes a prediction on a given image.

.. autoclass:: PyronearPredictor


Pyronear Engine
---------------

The Pyronear Engine class that manages the whole process of wildfire detection, from the image loading to the api call.

.. autoclass:: PyronearEngine

