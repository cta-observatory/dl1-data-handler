# ImageExtractor

Package for loading simtel data files, processing and calibrating the event data, and writing the processed data to a custom Pytables HDF5 format. Created for the testing of new machine learning and other analysis techniques for the
[Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org/ "CTA collaboration Homepage") collaboration. Built using Pytables and ctapipe.

Currently under development, intended for internal use only.

## Installation

The following installation method (for Linux) is recommended:

First, install the Anaconda Python distribution (Python 3.6), which can be found [here](https://www.anaconda.com/download/).

Clone the repository (or fork and clone if you wish to develop) into a local directory of your choice.

```bash
cd software
git clone https://github.com/bryankim96/image-extractor.git
cd image-extractor
```

Create a new Anaconda environment (defaults to Python 3.6), installing all of the dependencies using requirements.txt.

```bash
conda create -n [ENV_NAME] --file requirements.txt -c cta-observatory -c conda-forge -c openastronomy
```

### Install the bleeding edge version of ctapipe:

NOTE: As of v0.2.1, the conda package version of ctapipe appears to be lagging somewhat behind the development version and is missing key features which are used by image\_extractor. As a result, it is necessary (as of now) to install ctapipe from source. This is slightly messier and more time-consuming than the conda installation. 

Clone the ctapipe repository (or fork and clone if you wish to develop) into a local directory of your choice.

```bash
cd software
git clone https://github.com/cta-observatory/ctapipe.git
cd ctapipe
```

Activate the Anaconda environment you set up earlier:

```bash
source activate [ENV_NAME]
```

Then to install, simply run:

```bash
python setup.py install
```

You can verify that ctapipe was installed correctly by running:

```bash
conda list
```

and looking for ctapipe.

Do the same for ctapipe-extra (a repository containing additional resources for ctapipe):

```bash
cd software
git clone https://github.com/cta-observatory/ctapipe-extra.git
cd ctapipe-extra
source activate [ENV_NAME]
python setup.py install
```

If both packages are showing up correctly, you should now be able to run image\_extractor.py or any of the other scripts from the command line in your environment.

### Package Installation

Finally, you can install image-extractor as a package so that you can import and use it just like any other Python package.

To install image-extractor as a package in your main Python installation:

```bash
cd image-extractor
pip install . 
```

To install image-extractor as a package in your Anaconda environment:

```bash
source activate [ENV_NAME]
conda install pip

cd image-extractor
/path/to/anaconda/install/envs/[ENV_NAME]/bin/pip install .
```
where /path/to/anaconda/install is the path to your anaconda installation directory and ENV\_NAME is the name of your environment.

The path to the environment directory for the environment you wish to install into can be found quickly by running

```bash
conda env list
```

## Dependencies

See requirements.txt for the full list of dependencies.

The main dependencies are:

for image_extractor:

* Pytables
* Numpy
* Astropy
* configobj
* ctapipe

for additional scripts in scripts directory:

* Pillow
* ROOT
* matplotlib

## Usage

### From the Command Line:

To create a HDF5 dataset file out of a collection of .simtel.gz files, run:

```bash
image_extractor.py [path_to_simtel_files] [output_file] [config_file] [--bins_cuts_dict BINS_CUTS_DICT] [--max_events MAX_EVENTS] [--shuffle] [--split] [--debug]
```
on the command line.

ex:

```bash
image_extractor.py "/data/simtel/*.simtel.gz" "./dataset.h5" "./configuration_settings.config" "./bins_cuts_dict.pkl" --shuffle --split
```

* path_to_simtel_files - The path to .simtel.gz file(s) containing the events which you wish to process. Multiple files should be located in the same directory and indicated using a wildcard.
* output_file - The path to the HDF5 file into which you wish to write your data.
* config_file - The path to a configobj configuration file containing various data format settings and specifying cuts and bins (currently not applied directly in image_extractor, but through EventDisplay by saving in bins_cuts_dict. At the current time (v0.2.1) the necessary analysis stages are not yet implemented in ctapipe). Details in config.py.
* bins_cuts_dict - Optional path to a pickled Python dictionary of a specified format (see prepare_cuts_dict.py) containing information from EventDisplay on which events (run_number,event_number) passed the cuts, what their reconstructed energies are, and which energy bin they are in. This information is prepared in advance using prepare_cuts_dict.py and the settings indicated in the config file.
* max_events - Optional argument to specify the maximum number of events to save from the simtel file
* shuffle - Optional flag to randomly shuffle the data in the Events table after writing
* split - Optional flag to split the Events table into separate training/validation/test tables for convenience
* debug - Optional flag to print additional debug information.

NOTE: The shuffle and split options are primarily for convenience when using the data files for training convolutional neural networks. It is unlikely they will be included as part of the final standard data format.

NOTE: The split option is currently set to use a hardcoded default split setting (0.9 training, 0.1 validation). This can be modified in the script if desired.

NOTE: The splitting and shuffling handled by the split and shuffle options are NOT done in-place, so they require a temporary disk space overhead beyond the normal final size of the output .h5 file. Because only the Events tables are copied/duplicated and the majority of the final file size is in the telescope arrays, this overhead is likely insignificant for the 'mapped' storage format. However, it is much larger for the 'all' storage format, as the telescope images are stored directly in the Event tables and are therefore duplicated temporarily. Also, it is worth noting that this overhead becomes larger in absolute terms as the absolute size of the output files becomes larger.

### From a Python script:

If the package was installed locally as described above, you can import classes from it and use them directly in a Python script.

ex:

```python
from image_extractor import image_extractor

#max events to read per file
max_events = 50

#set parameters for your ImageExtractor
output_path = "/home/computer/user/data/dataset.h5"
mode = "gh_class"

#...
#more parameters

#load bins cuts dictionary from file
bins_cuts_dict = pkl.load(open(args.bins_cuts_dict_file, "rb" ))

#data file
data_file = "/home/computer/user/data/simtel/test.simtel.gz"

extractor = image_extractor.ImageExtractor(output_path,bins_cuts_dict,mode, ...)

extractor.process_data(data_file,max_events)

#...
```

### Config files and bins_cuts_dict files

"Default" configuration files and bins_cuts_dict files are provided in /aux/. The "1bin" files use standard cuts, but only 1 large energy bin containing all events. The non-"1bin" files use the standard cuts and 3 energy bins (low, medium, high) which are specified in the config files.

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





 
