DL1 Data Handler
================


.. image:: https://zenodo.org/badge/72042185.svg
   :target: https://zenodo.org/badge/latestdoi/72042185
   :alt: DOI


.. image:: https://travis-ci.org/cta-observatory/dl1-data-handler.svg?branch=master
   :target: https://travis-ci.org/cta-observatory/dl1-data-handler.svg?branch=master
   :alt: build status


.. image:: https://anaconda.org/ctlearn-project/dl1_data_handler/badges/installer/conda.svg
   :target: https://anaconda.org/ctlearn-project/dl1_data_handler/
   :alt: Anaconda-Server Badge


.. image:: https://img.shields.io/pypi/v/dl1-data-handler
    :target: https://pypi.org/project/dl1-data-handler/
    :alt: Latest Release


.. image:: https://coveralls.io/repos/github/cta-observatory/dl1-data-handler/badge.svg?branch=master
   :target: https://coveralls.io/github/cta-observatory/dl1-data-handler?branch=master
   :alt: Coverage Status


A package of utilities for writing (deprecated), reading, and applying image processing to `Cherenkov Telescope Array (CTA) <https://www.cta-observatory.org/>`_ DL1 data (calibrated images) in a standardized format. Created primarily for testing machine learning image analysis techniques on IACT data.

Currently supports data in the CTA pyhessio sim_telarray format, with the possibility of supporting other IACT data formats in the future. Built using ctapipe and PyTables.

Previously named image-extractor (v0.1.0 - v0.6.0). Currently under development, intended for internal use only.

Data Format
-----------

[Deprecated] DL1DataWriter implements a standardized format for storing simulated CTA DL1 event data into Pytables files. CTAMLDataDumper is the class which implements the conversion from ctapipe containers to the CTA ML data format. See the wiki page `here <https://github.com/cta-observatory/dl1-data-handler/wiki/CTA-ML-Data-Format>`_ for a full description of this data format and an FAQ.

ctapipe process tool should be used instead.

Installation
------------

The following installation method (for Linux) is recommended:

Installing as a conda package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install dl1-data-handler as a conda package, first install Anaconda by following the instructions here: https://www.anaconda.com/distribution/.

The following command will set up a conda virtual environment, add the
necessary package channels, and install dl1-data-handler specified version and its dependencies:

.. code-block:: bash

   DL1DH_VER=0.10.11
   wget https://raw.githubusercontent.com/cta-observatory/dl1-data-handler/v$DL1DH_VER/environment.yml
   conda env create -n [ENVIRONMENT_NAME] -f environment.yml
   conda activate [ENVIRONMENT_NAME]
   conda install -c ctlearn-project dl1_data_handler=$DL1DH_VER

This should automatically install all dependencies (NOTE: this may take some time, as by default MKL is included as a dependency of NumPy and it is very large).

If you want to import any functionality from dl1-data-handler into your own Python scripts, then you are all set. However, if you wish to make use of any of the scripts in dl1-data-handler/scripts (like write_data.py), you should also clone the repository locally and checkout the corresponding tag (i.e. for version v0.10.11):

.. code-block:: bash

   git clone https://github.com/cta-observatory/dl1-data-handler.git
   git checkout v0.10.11

dl1-data-handler should already have been installed in your environment by Conda, so no further installation steps (i.e. with setuptools or pip) are necessary and you should be able to run scripts/write_data.py directly.

Dependencies
------------

The main dependencies are:


* PyTables >= 3.7
* NumPy >= 1.16.0
* ctapipe == 0.19

Also see setup.py.

Usage
-----

[Deprecated] DL1DataWriter
^^^^^^^^^^^^^^^^^^^^^^^^^^
The DL1DataWriter is not supported by the default installation. Please follow the custom installation instructions:

.. code-block:: bash

   git clone https://github.com/cta-observatory/dl1-data-handler.git
   git checkout magic # for MAGIC data
   conda env create -n [ENVIRONMENT_NAME] -f environment-magic.yml
   conda activate [ENVIRONMENT_NAME]
   python setup_magic.py install

From the Command Line:
~~~~~~~~~~~~~~~~~~~~~~

To process data files into a desired format:

.. code-block:: bash

   dl1dh-write_data [runlist] [--config_file,-c CONFIG_FILE_PATH] [--output_dir,-o OUTPUT_DIR] [--debug]

on the command line.

ex:

.. code-block:: bash

   dl1dh-write_data runlist.yml -c example_config.yml --debug


* runlist - A YAML file containing groups of input files to load data from and output files to write to. See example runlist for format.
* config_file - The path to a YAML configuration file specifying all of the settings for data loading and writing. See example config file and documentation for details on each setting. If none is provided, default settings are used for everything.
* output_dir - Path to directory to write all output files to. If not provided, defaults to the current directory.
* debug - Optional flag to print additional debug information from the logger.

In a Python script:
~~~~~~~~~~~~~~~~~~~

If the package was installed with pip as described above, you can import and use it in Python like:

ex:

.. code-block:: python

   from dl1_data_handler import dl1_data_writer

   event_source_class = MyEventSourceClass
   event_source_settings = {'setting1': 'value1'}

   data_dumper_class = MyDataDumperClass
   data_dumper_settings = {'setting2': 'value2'}

   def my_cut_function(event):
       # custom cut logic here
       return True

   data_writer = dl1_data_writer.DL1DataWriter(event_source_class=event_source_class,
       event_source_settings=event_source_settings,
       data_dumper_class=data_dumper_class,
       data_dumper_settings=dumper_settings,
       preselection_cut_function=my_cut_function,
       output_file_size=10737418240,
       events_per_file=500)

   run_list = [
    {'inputs': ['file1.simtel.gz', 'file2.simtel.gz'],
     'target': 'output.h5'}
   ]

   data_writer.process_data(run_list)

Generating a run list
~~~~~~~~~~~~~~~~~~~~~

If processing data from simtel.gz files, as long as their filenames have the format ``[particle_type]_[ze]deg_[az]deg_run[run_number]___[production info].simtel.gz`` or ``[particle_type]_[ze]deg_[az]deg_run[run_number]___[production info]_cone[cone_num].simtel.gz`` the dl1dh-generate_runlist can be used to automatically generate a runlist in the correct format. The script can also generate a run list with the MAGIC-MARS superstar files.

It can be called as:

.. code-block:: bash

   dl1dh-generate_runlist [file_dir] [--num_inputs_per_run,-n NUM_INPUTS_PER_RUN] [--output_file_name,-f OUTPUT_FILE_NAME] [--output_dir,-o OUTPUT_DIR]


* file_dir - Path to a directory containing simtel.gz files with the filename format specified above.
* num_inputs_per_run - Number of input files with the same particle type, ze, az, and production info to group together into each run (defaults to 10).
* output_file - Path/filename of output runlist file without a postfix. Defaults to ./runlist
* output_dir - Path where to save generated files. By default, the input directory is used.

It will automatically sort the simtel files in the file_dir directory into groups with matching particle_type, zenith, azimuth, and production parameters. Within each of these groups, it will group together input files in sequential order into runs of size NUM_INPUTS_PER_RUN. The output filename for each run will be automatically generated as ``[particle_type]_[ze]deg_[az]deg_runs[run_number_range]___[production info].h5``. The output YAML file will be written to output_file.

ImageMapper
^^^^^^^^^^^

The ImageMapper class transforms the hexagonal input pixels into a 2D Cartesian output image. The basic usage is demonstrated in the `ImageMapper tutorial <https://github.com/cta-observatory/dl1-data-handler/blob/master/notebooks/test_image_mapper.ipynb>`_. It requires `ctapipe-extra <https://github.com/cta-observatory/ctapipe-extra>`_ outside of the dl1-data-handler. See this publication for a detailed description: `arXiv:1912.09898 <https://arxiv.org/abs/1912.09898>`_

Other scripts
^^^^^^^^^^^^^

All other scripts located in the scripts/deprecated directory are not currently updated to be compatible with dl1-data-handler >= 0.7.0 and should not be used.

Examples/Tips
-------------

* Vitables is very helpful for viewing and debugging PyTables-style HDF5 files. Installation/download instructions can be found in the link below.

Known Issues/Troubleshooting
----------------------------

Links
-----


* `Cherenkov Telescope Array (CTA) <https://www.cta-observatory.org/>`_ - Homepage of the CTA collaboration 
* `CTLearn <https://github.com/ctlearn-project/ctlearn/>`_ and `GammaLearn <https://gitlab.lapp.in2p3.fr/GammaLearn/GammaLearn>`_ - Repository of code for studies on applying deep learning to IACT analysis tasks. Maintained by groups at Columbia University, Universidad Complutense de Madrid, Barnard College (CTLearn) and LAPP (GammaLearn).
* `ctapipe <https://cta-observatory.github.io/ctapipe/>`_ - Official documentation for the ctapipe analysis package (in development)
* `ViTables <http://vitables.org/>`_ - Homepage for ViTables application for Pytables HDF5 file visualization

