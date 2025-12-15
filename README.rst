DL1 Data Handler
================


.. image:: https://zenodo.org/badge/72042185.svg
   :target: https://zenodo.org/badge/latestdoi/72042185
   :alt: DOI


.. image:: https://anaconda.org/ctlearn-project/dl1_data_handler/badges/version.svg
   :target: https://anaconda.org/ctlearn-project/dl1_data_handler/
   :alt: Anaconda-Server Badge


.. image:: https://img.shields.io/pypi/v/dl1-data-handler
    :target: https://pypi.org/project/dl1-data-handler/
    :alt: Latest Release


.. image:: https://github.com/cta-observatory/dl1-data-handler/actions/workflows/python-package-conda.yml/badge.svg
    :target: https://github.com/cta-observatory/dl1-data-handler/actions/workflows/python-package-conda.yml
    :alt: Continuos Integration

A package of utilities for reading and applying image processing to `Cherenkov Telescope Array Observatory (CTAO) <https://www.ctao.org/>`_ R1/DL0/DL1 data in a standardized format. Created primarily for testing machine learning image analysis techniques on IACT data.

Currently supports ctapipe  >= v6.0.0 data format. 

Previously named image-extractor (v0.1.0 - v0.6.0). Currently under development, intended for internal use only.


Installation
------------

First, create and activate a fresh conda environment:

.. code-block:: bash

   mamba create -n dl1dh -c conda-forge python==3.12 ctapipe
   mamba activate dl1dh

The lastest version fo this package can be installed as a conda or pip package:

.. code-block:: bash

   mamba install ctlearn-project::dl1_data_handler

or

.. code-block:: bash

   pip install dl1-data-handler

Installing as a conda package

This should automatically install all dependencies (NOTE: this may take some time, as by default MKL is included as a dependency of NumPy and it is very large).

Developers
------------

We welcome contributions to this package! If you're interested in contributing, please make sure to follow the `ctapipe developer guidelines <https://ctapipe.readthedocs.io/en/latest/developer-guide/getting-started.html>`_.

Links
-----


* `Cherenkov Telescope Array Observatory (CTAO) <https://www.ctao.org/>`_ - Homepage of the CTA Observatory
* `CTLearn <https://github.com/ctlearn-project/ctlearn/>`_ and `GammaLearn <https://gitlab.lapp.in2p3.fr/GammaLearn/GammaLearn>`_ - Repository of code for studies on applying deep learning to IACT analysis tasks. Maintained by groups at Universidad Complutense de Madrid, University of Geneva (CTLearn) and LAPP (GammaLearn).
* `ctapipe <https://cta-observatory.github.io/ctapipe/>`_ - Official documentation for the ctapipe analysis package (in development)

