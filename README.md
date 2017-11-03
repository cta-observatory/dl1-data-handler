# ImageExtractor

## Description
Software for loading simtel data files, processing them using ctapipe, and writing data to Pytables HDF5 format for use in machine learning and other analysis tasks. Designed to handle data storage for testing new analysis techniques for the 
[Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org/ "CTA collaboration Homepage") collaboration.

Currently in development, intended for internal use only.

## Installation

As of now, no packaged version or setuptools-style installation method is supported. The recommended way to use the code is to  fork/clone the repository locally, create an Anaconda environment, and install into it all of the relevant dependencies listed in requirements.txt. Alternatively, a particular release version can be downloaded from the releases page (dependency installation should be handled in the same way). 

NOTE: Depending on the release/version of ImageExtractor, it may be necessary to install the dev version of ctapipe (rather than the conda package version), as explained [here](https://cta-observatory.github.io/ctapipe/getting_started/index.html#get-the-ctapipe-software)

## Dependencies

See requirements.txt for up-to-date list.

As of release v0.2.1,

for image_extractor.py:

* Pytables 3.4.2
* Numpy 1.13.3
* Astropy 2.0.2
* configobj 5.0.6
* ctapipe 0.3.1 (dev)

for scripts in scripts directory:

* Pillow 4.3.0
* ROOT 0.0.1
* h5py 2.7.1
* matplotlib 2.1.0

## Usage

Once in an appropriate conda environment with all dependencies installed (or using another Python installation with all dependencies installed):

### For processing one simtel file:

image_extractor.py [simtel_file] [output_file] [config_file] [--bins_cuts_dict BINS_CUTS_DICT] [--debug] [--max_events MAX_EVENTS]

ex:

image_extractor.py "gamma_20deg_0deg_run30137___cta-prod3-lapalma-2147m-LaPalma-SCT_cone10.simtel.gz" "./dataset.h5" "configuration_settings.config" "bins_cuts_dict.pkl"

* simtel_file - The path to a simtel.gz file containing the events which you wish to process
* output_file - The path to an HDF5 file into which you wish to write your data. If it does not exist, it will be created. If it does, it will be appended to.
* config_file - The path to a configobj configuration file containing various data format settings and specifying cuts and bins (currently not applied directly in image_extractor, but through EventDisplay by saving in bins_cuts_dict. At the current time (v0.2.1) the necessary analysis stages are not yet implemented in ctapipe). Details in config.py
* bins_cuts_dict - Optional path to a pickled Python dictionary of a specified format (see prepare_cuts_dict.py) containing information from EventDisplay on which events (run_number,event_number) passed the cuts, what their reconstructed energies are, and which energy bin they are in. This information is prepared in advance using prepare_cuts_dict.py and the settings indicated in the config file.
* debug - Optional flag to print additional debug information.
* max_events - Optional argument to specify the maximum number of events to save from the simtel file

### For processing an entire dataset (format is designed for deep learning use):

NOTE: These scripts are specifically for internal use by the Columbia/Barnard deep learning group and are not likely to be part of the standard collaboration-wide format. 

Use generate_dataset_pytables.sh to process multiple simtel files (specified with a wilcard argument), then shuffle and split into train/validation/test tables using shuffle_split_events.py. 

NOTE: generate_datset_pytables.sh will leave behind a "raw" dataset file (before shuffling and splitting) which is named, by default, output_temp.h5. This can be kept or deleted as desired after it finishes. Be sure when running generate_dataset_pytables.sh that no such file already exists, as this will cause any data already in that file to be included in the final output HDF5 file. 

To generate dataset:

ex.

python generate_dataset_pytables.sh "simtel_directory/*.simtel.gz" "./dataset.h5" "bins_cuts_dict.pkl" "configuration_settings.config"

To shuffle/split a dataset:

ex.

python shuffle_split_events.py "-shuffle" "-no_test" "./dataset_raw.h5" "./dataset_final.h5"


### Config files and bins_cuts_dict files

"Default" configuration files and bins_cuts_dict files are provided in /aux/. The "1bin" files use standard cuts, but only 1 large energy bin. The non-"1bin" files use the standard cuts and 3 energy bins (low, medium, high) which are specified in the config files.

The config files can be modified directly using a text editor, although it should be ensured that they match the config spec located in config.py. Config.py can be used to generate an example default configuration file or validate an existing one.

Bins_cuts_dict.pkl files are the result of running the EventDisplay analysis on the simtel files through to the MSCW stage, then applying the cuts specified in the config file on the array/event-level parameters. The default cuts were selected to match those used by the Boosted Decision Tree analysis method for Gamma-Hadron separation currently being used in EventDisplay for VERITAS data. A new one can be created (for example, to match a new set of simtel files) by running the standard ED analysis on all simtel files up to the MSCW stage, then passing the MSCW.root files into prepare_bins_dict.py.

## Examples/Tips

* Vitables is very helpful for viewing and debugging PyTables-style HDF5 files. Installation/download instructions can be found in the link below. NOTE: It is STRONGLY recommended that vitables be installed in a separate Anaconda environment with Python version 2.7 to avoid issues with conflicting PyQt5 versions. See this issue thread for details: [https://github.com/ContinuumIO/anaconda-issues/issues/1554](https://github.com/ContinuumIO/anaconda-issues/issues/1554)

## Known Issues/Troubleshooting

* ViTables PyQt5 dependency confict (pip vs. conda): [relevent issue thread](https://github.com/ContinuumIO/anaconda-issues/issues/1554)
* Conda version of ctapipe does not support most recent features. Use dev version instead: [installation instructions](https://cta-observatory.github.io/ctapipe/getting_started/index.html#get-the-ctapipe-software)

## Links

* [Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org/ "CTA collaboration Homepage") - Homepage of the CTA collaboration
* [Deep Learning for CTA Analysis](https://github.com/bryankim96/deep-learning-CTA "Deep Learning for CTA Repository") - Repository of code for studies on applying deep learning to CTA analysis tasks. Maintained by groups at Columbia University and Barnard College.
* [ctapipe](https://cta-observatory.github.io/ctapipe/ "ctapipe Official Documentation Page") - Official documentation for the ctapipe analysis package (in development)
* [ViTables](http://vitables.org/ "ViTables Homepage") - Homepage for ViTables application for Pytables HDF5 file visualization





 
