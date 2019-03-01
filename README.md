# DL1 Data Handler

[![build status](https://travis-ci.org/cta-observatory/image-extractor.svg?branch=master)](https://travis-ci.org/cta-observatory/image-extractor.svg?branch=master)[![Coverage Status](https://coveralls.io/repos/github/cta-observatory/image-extractor/badge.svg?branch=master)](https://coveralls.io/github/cta-observatory/image-extractor?branch=master) [![Code Health](https://landscape.io/github/cta-observatory/image-extractor/master/landscape.svg?style=flat)](https://landscape.io/github/cta-observatory/image-extractor/master)

A package of utilities for writing, reading, and applying image processing to [Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org/ "CTA collaboration Homepage") DL1 data (calibrated images) in a standardized format. Created primarily for testing machine learning image analysis techniques on IACT data.

Currently supports data in the CTA pyhessio sim_telarray format, with the possibility of supporting other IACT data formats in the future. Built using ctapipe and PyTables.

Previously named image-extractor (v0.1.0 - v0.6.0). Currently under development, intended for internal use only.

## Data Format

DL1DataWriter implements a standardized format for storing simulated CTA DL1 event data into Pytables files. CTAMLDataDumper is the class which implements the conversion from ctapipe containers to the CTA ML data format. See the wiki page [here](https://github.com/cta-observatory/dl1-data-handler/wiki/CTA-ML-Data-Format) for a full description of this data format and an FAQ.

## Installation

The following installation method (for Linux) is recommended:

### Installing with pip/setuptools from source

You can install DL1 Data Handler using pip after cloning the repository:

```bash
git clone https://github.com/cta-observatory/dl1-data-handler.git
cd dl1-data-handler
```

To install into a virtualenv environment:

```bash
virtualenv /path/to/ENV
source /path/to/ENV/bin/activate
pip install .
```

### Installing as a conda package

To install it as a conda package, first install Anaconda by following the instructions here: https://www.anaconda.com/distribution/.

Then, create and enter a new Python 3.6 environment with:

```bash
conda create -n [ENVIRONMENT_NAME] python=3.6
source activate [ENVIRONMENT_NAME]
```

From the environment, add the necessary channels for all dependencies:

```bash
conda config --add channels anaconda
conda config --add channels conda-forge
conda config --add channels cta-observatory
```

Install the package:

```bash
conda install -c bryankim96 dl1_data_handler
```

This should automatically install all dependencies (NOTE: this may take some time, as by default MKL is included as a dependency of NumPy and it is very large).

If you want to import any functionality from DL1 Data Handler into your own Python scripts, then you are all set. However, if you wish to make use of any of the scripts in dl1-data-handler/scripts (like write_data.py), you should also clone the repository locally and checkout the corresponding tag (i.e. for version v0.7.2): 

```bash
git clone https://github.com/cta-observatory/dl1-data-handler.git
git checkout -b v0.7.2 v0.7.2
```

DL1 Data Handler should already have been installed in your environment by Conda, so no further installation steps (i.e. with setuptools or pip) are necessary and you should be able to run scripts/write_data.py directly.

## Dependencies

The main dependencies are:

for dl1-data-writer:

* PyTables >= 3.4.4
* NumPy >= 1.15.0
* ctapipe >= 0.6.2
* ctapipe-extra >= 0.2.16
* pyhessio >= 2.1.1

Also see setup.py.

## Usage

### DL1 Data Writer

#### From the Command Line:

To process data files into a desired format:

```bash
scripts/write_data.py [runlist] [--config_file CONFIG_FILE_PATH] [--output_dir OUTPUT_DIR] [--debug]
```
on the command line.

ex:

```bash
scripts/write_data.py runlist.yml --config_file example_config.yml --debug
```

* runlist - A YAML file containing groups of input files to load data from and output files to write to. See example runlist for format.
* config_file - The path to a YAML configuration file specifying all of the settings for data loading and writing. See example config file and documentation for details on each setting. If none is provided, default settings are used for everything.
* output_dir - Path to directory to write all output files to. If not provided, defaults to the current directory.
* debug - Optional flag to print additional debug information from the logger.

#### In a Python script:

If the package was installed with pip as described above, you can import and use it in Python like:

ex:

```python
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
    calibration_settings={
         'r1_product': 'HESSIOR1Calibrator',
         'extractor_product': 'NeighbourPeakIntegrator'
     },
     preselection_cut_function=my_cut_function,
     output_file_size=10737418240,
     events_per_file=500)

run_list = [
 {'inputs': ['file1.simtel.gz', 'file2.simtel.gz'],
  'target': 'output.h5'}
]

data_writer.process_data(run_list)

```
#### Generating a run list

If processing data from simtel.gz files, as long as their filenames have the format ``[particle_type]_[ze]deg_[az]deg_run[run_number]___[production info].simtel.gz`` or ``[particle_type]_[ze]deg_[az]deg_run[run_number]___[production info]_cone[cone_num].simtel.gz`` the scripts/generate_runlist.py can be used to automatically generate a runlist in the correct format.

It can be called as:

```bash
scripts/generate_runlist.py [file_dir] [--num_inputs_per_run NUM_INPUTS_PER_RUN] [--output_file OUTPUT_FILE]
```

* file_dir - Path to a directory containing simtel.gz files with the filename format specified above.
* num_inputs_per_run - Number of input files with the same particle type, ze, az, and production info to group together into each run (defaults to 10).
* output_file - Path/filename of output runlist file. Defaults to ./runlist.yml

It will automatically sort the simtel files in the file_dir directory into groups with matching particle_type, zenith, azimuth, and production parameters. Within each of these groups, it will group together input files in sequential order into runs of size NUM_INPUTS_PER_RUN. The output filename for each run will be automatically generated as ``[particle_type]_[ze]deg_[az]deg_runs[run_number_range]___[production info].h5``. The output YAML file will be written to output_file.

### Other scripts

All other scripts located in the scripts/deprecated directory are not currently updated to be compatible with dl1-data-handler >= 0.7.0 and should not be used.

## Examples/Tips

* Vitables is very helpful for viewing and debugging PyTables-style HDF5 files. Installation/download instructions can be found in the link below. NOTE: It is STRONGLY recommended that vitables be installed in a separate Anaconda environment with Python version 2.7 to avoid issues with conflicting PyQt5 versions. See this issue thread for details: [https://github.com/ContinuumIO/anaconda-issues/issues/1554](https://github.com/ContinuumIO/anaconda-issues/issues/1554)

## Known Issues/Troubleshooting

* As of v0.7.2 there appears to be an issue when processing files containing SCT data. A fix is planned for v0.7.3.
* ViTables PyQt5 dependency confict (pip vs. conda): [relevent issue thread](https://github.com/ContinuumIO/anaconda-issues/issues/1554)

## Links

* [Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org/ "CTA collaboration Homepage") - Homepage of the CTA collaboration
* [Deep Learning for CTA Analysis](https://github.com/bryankim96/deep-learning-CTA "Deep Learning for CTA Repository") - Repository of code for studies on applying deep learning to CTA analysis tasks. Maintained by groups at Columbia University and Barnard College.
* [ctapipe](https://cta-observatory.github.io/ctapipe/ "ctapipe Official Documentation Page") - Official documentation for the ctapipe analysis package (in development)
* [ViTables](http://vitables.org/ "ViTables Homepage") - Homepage for ViTables application for Pytables HDF5 file visualization
