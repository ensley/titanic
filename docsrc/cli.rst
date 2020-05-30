Makefile rules
==============

The project comes with these ``make`` directives:

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

This page describes what each of these directives does in a bit more detail.

``make clean``
--------------

Deletes all models and data that are created over the course of the project
pipeline. This deletes the raw data as well, so the data will have to be
re-downloaded from Kaggle once ``make clean`` is performed. To delete
everything but keep the raw data, use :ref:`make_mostlyclean`.

``make data``
-------------

Takes the data from raw to processed and ready for modeling.
Runs :func:`~titanic.data.clean.main` and saves the output to ``data/interim``.
Then runs :func:`~titanic.features.transform.main` and saves the final data
to ``data/processed``.

``make docs``
-------------

Steps into ``docsrc/`` and runs ``make html``, copying the built documentation
to ``docs/``. This allows the documentation to easily be hosted on Github Pages.

``make githooks``
-----------------

Sets up the pre-commit git hook that's stored in the ``.githooks`` folder.
You can use this in your git repository to prevent Jupyter notebooks from
being committed without being converted to output-free Python scripts first.
It isn't necessary to use this command to replicate any part of the analysis.

``make models``
---------------

Trains the models using the processed training data. Serializes the fitted
models and stores them in ``models/fitted``.

.. _make_mostlyclean:

``make mostlyclean``
--------------------

Deletes all models and data that are created over the course of the project
pipeline, but keeps the raw data intact so that the Kaggle download does not
need to be repeated.

``make predictions``
--------------------

Generates predictions from the fitted models in ``models/fitted`` and using
the processed test data in ``data/processed``. Saves the predictions in
``models/predictions``.

``make raw_data``
-----------------

Downloads the raw data from Kaggle. This requires the `Kaggle API <https://www.kaggle.com/docs/api>`_
to be installed and properly configured.