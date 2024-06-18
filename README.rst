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

A package of utilities for reading, and applying image processing to `Cherenkov Telescope Array (CTA) <https://www.ctao.org/>`_ R0/R1/DL0/DL1 data in a standardized format. Created primarily for testing machine learning image analysis techniques on IACT data.

Currently supports ctapipe v6.0.0 data format. 

Previously named image-extractor (v0.1.0 - v0.6.0). Currently under development, intended for internal use only.


Installation
------------

The following installation method (for Linux) is recommended:

Installing as a conda package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install dl1-data-handler as a conda package, first install Anaconda by following the instructions here: https://www.anaconda.com/distribution/.

The following command will set up a conda virtual environment, add the
necessary package channels, and install dl1-data-handler specified version and its dependencies:

.. code-block:: bash

   DL1DH_VER=0.11.1
   wget https://raw.githubusercontent.com/cta-observatory/dl1-data-handler/v$DL1DH_VER/environment.yml
   conda env create -n [ENVIRONMENT_NAME] -f environment.yml
   conda activate [ENVIRONMENT_NAME]
   conda install -c ctlearn-project dl1_data_handler=$DL1DH_VER

This should automatically install all dependencies (NOTE: this may take some time, as by default MKL is included as a dependency of NumPy and it is very large).

If you want to import any functionality from dl1-data-handler into your own Python scripts, then you are all set. However, if you wish to make use of any of the scripts in dl1-data-handler/scripts (like write_data.py), you should also clone the repository locally and checkout the corresponding tag (i.e. for version v0.11.1):

.. code-block:: bash

   git clone https://github.com/cta-observatory/dl1-data-handler.git
   git checkout v0.11.1

dl1-data-handler should already have been installed in your environment by Conda, so no further installation steps (i.e. with setuptools or pip) are necessary and you should be able to run scripts/write_data.py directly.

Dependencies
------------

The main dependencies are:


* PyTables >= 3.8
* NumPy >= 1.20.0
* ctapipe == 0.21.1

Also see setup.py.

Usage
-----

ImageMapper
^^^^^^^^^^^

The ImageMapper class transforms the hexagonal input pixels into a 2D Cartesian output image. The basic usage is demonstrated in the `ImageMapper tutorial <https://github.com/cta-observatory/dl1-data-handler/blob/master/notebooks/test_image_mapper.ipynb>`_. It requires `ctapipe-extra <https://github.com/cta-observatory/ctapipe-extra>`_ outside of the dl1-data-handler. See this publication for a detailed description: `arXiv:1912.09898 <https://arxiv.org/abs/1912.09898>`_


Links
-----


* `Cherenkov Telescope Array (CTA) <https://www.ctao.org/>`_ - Homepage of the CTA Observatory 
* `CTLearn <https://github.com/ctlearn-project/ctlearn/>`_ and `GammaLearn <https://gitlab.lapp.in2p3.fr/GammaLearn/GammaLearn>`_ - Repository of code for studies on applying deep learning to IACT analysis tasks. Maintained by groups at Columbia University, Universidad Complutense de Madrid, Barnard College (CTLearn) and LAPP (GammaLearn).
* `ctapipe <https://cta-observatory.github.io/ctapipe/>`_ - Official documentation for the ctapipe analysis package (in development)

