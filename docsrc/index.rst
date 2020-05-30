Titanic: Machine Learning from Disaster
=======================================

This is a use case for `ccdspy <https://github.com/ensley/ccdspy>`_, a data science
project template for Python.

It implements a solution to the `Titanic Kaggle competition <https://www.kaggle.com/c/titanic>`_
that ranks in the top 15-20% on the leaderboard.

Installation
------------

Run

.. code-block:: bash

    pip install -r requirements.txt

Usage
-----

The easiest way to reproduce the project is by using the ``make`` directives.

.. code-block:: none

    Available rules:

    clean               Delete all models and data
    data                Transform raw data
    docs                Build documentation and copy it to the docs folder
    githooks            Set up githooks
    models              Train models
    mostlyclean         Delete all models and data, except for the raw downloads
    predictions         Create predictions
    raw_data            Pull raw data from Kaggle

The project pipeline is:

1. Download raw data from Kaggle.
2. Clean the raw data, create new features and transform existing ones.
3. Train the models using the training data.
4. Generate predictions using the test data.

Those four steps correspond to the following directives:

1. ``make raw_data``
2. ``make data``
3. ``make models``
4. ``make predictions``

For each of these steps, ``make`` will run as many of the preceding steps as
necessary to ensure the required input files exist. This means that, starting
from a completely empty (or nonexistent) ``data/`` directory, running
``make predictions`` will perform the entire pipeline from raw data download
to generating predictions.

See :doc:`cli` for details on each of these directives.


Additional Documentation
------------------------

.. toctree::
    cli
    api
    source/01-initial-data-exploration.ipynb
    source/02-feature-engineering.ipynb

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
