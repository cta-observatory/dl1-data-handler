# -*- coding: utf-8 -*-
"""
Classes and methods for loading DL1 data from various formats into ctapipe via
event sources, calibrating it, and writing it to a standard file format.
"""

from abc import ABC, abstractmethod
import pkg_resources
import argparse
import logging
import os
import re
import multiprocessing
import warnings
import sys

import yaml
import numpy as np
import tables
from ctapipe import io, calib

import table_definitions

logger = logging.getLogger(__name__)

# Disable warnings by default
if not sys.warnoptions:
    warnings.simplefilter("ignore")

class DL1DataDumper(ABC):
    """Generic abstract class for dumping data from ctapipe DL1 containers to an output file.

    Attributes:
        output_filename (str): Full path + filename to target output file.
    """

    @abstractmethod
    def __init__(self, output_filename):
        self.output_filename = output_filename

    @abstractmethod
    def dump_instrument_info(self, instrument_container):
        """Abstract method to dump ctapipe instrument container to output file.

        Args:
            instrument_container (ctapipe.io.containers.InstrumentContainer):
                ctapipe container of instrument data
        """
        pass

    @abstractmethod
    def dump_mc_header_info(self, mc_header_container):
        """Abstract method to dump ctapipe instrument container to output file.

        Args:
            mc_header_container (ctapipe.io.containers.MCHeaderContainer):
                ctapipe container of monte carlo header data (for entire run).
        """
        pass

    # Write non-ctapipe container header info
    @abstractmethod
    def dump_header_info(self):
        """Abstract method to dump non-ctapipe header data to output file.
        """
        pass

    # Write a single event's information (dl1 data, monte carlo information)
    @abstractmethod
    def dump_event(self, event_container):
        """Abstract method to dump ctapipe event data (event parameters
        and images) to output file.

        Args:
            event_container (ctapipe.io.containers.DataContainer):
                ctapipe container of all event data for a given event.
        """
        pass

class CTAMLDataDumper(DL1DataDumper):
    """Class for dumping ctapipe DL1 data to the CTA ML data format.

    See the Github repository wiki page for a detailed description of the data format.

    Attributes:
        DEFAULT_IMGS_PER_EVENT (float): Default number of triggered telescopes
        (images) expected for any telescope. This value is used as a default if
        a given telescope type's expected_images_per_event is not specified.

        output_filename (str): Filename of the output file to write to.
        filter_settings (dict): Dictionary of filter settings (kwargs), passed
            to the constructor for tables.Filters. Determines compression settings.
        expected_tel_types (int): Number of expected telescope types in the
            '/Telescope_Type_Information' table. Used for setting the chunk size.
        expected_tels (int): Number of expected telescope types in the
            '/Telescope_Type_Information' table. Used for setting the chunk size.
        expected_events (int): Number of expected telescope types in the
            '/Telescope_Type_Information' table. Used for setting the chunk size.
        expected_images_per_event (dict): Dictionary containing telescope type
            names as keys, with the expected average number of triggered telescopes
            of that type per event (float) as the value. Used for setting the chunk size.
        index_columns (list): List of tuples of form (table_path, column_name),
            specifying the tables and columns in the output file on which to create
            indexes for faster search. Used for setting the chunk size.
    """
    DEFAULT_IMGS_PER_EVENT = 1.0

    def __init__(self,
            output_filename,
            filter_settings={
                'complib': 'lzo',
                'complevel': 1
            },
            expected_tel_types=10,
            expected_tels=300,
            expected_events=10000,
            expected_images_per_event={
                'LST:LSTCam': 0.5,
                'MST:NectarCam': 2.0,
                'MST:FlashCam': 2.0,
                'MST-SCT:SCTCam': 1.5,
                'SST:DigiCam': 1.25,
                'SST:ASTRICam': 1.25,
                'SST:CHEC': 1.25,
            },
            index_columns=[
                ('/Events', 'mc_energy'),
                ('/Events', 'alt'),
                ('/Events', 'az'),
                ('tel', 'event_index')
                ]):

        super().__init__(output_filename)
        self.file = tables.open_file(output_filename, mode="w")

        if filter_settings:
            self.filters = tables.Filters(**filter_settings)
        else:
            self.filters = None

        self.expected_tel_types = expected_tel_types
        self.expected_tels = expected_tels
        self.expected_events = expected_events
        self.expected_images_per_event = expected_images_per_event

        self.index_columns = index_columns

        self.image_tables = []

    def __del__(self):
        """ Destructor

        Calls finalize method to do any final processing, then closes the file.

        """
        self.finalize()
        self.file.close()

    def dump_instrument_info(self, instrument_container):
        """Dumps ctapipe instrument container to output file.

        If not present in the output file, creates two tables,
        '/Array_Information' and '/Telescope_Type_Information'. Then,
        populates them row by row with array data and telescope type data.

        Args:
            instrument_container (ctapipe.io.containers.InstrumentContainer):
                ctapipe container of instrument data
        """
        if "/Array_Information" not in self.file:
            logger.info("Writing array/subarray information to table...")
            array_table = self.file.create_table(self.file.root,
                            'Array_Information',
                            table_definitions.ArrayTableRow,
                            ("Table of array/subarray information"),
                            filters=self.filters,
                            expectedrows=self.expected_tels)

            row = array_table.row

            # Dump properties of all telescopes in array/subarray to table
            for tel_id in instrument_container.subarray.tels:
                telescope_description = instrument_container.subarray.tels[tel_id]
                row["id"] = tel_id
                row["x"] = instrument_container.subarray.positions[tel_id].value[0]
                row["y"] = instrument_container.subarray.positions[tel_id].value[1]
                row["z"] = instrument_container.subarray.positions[tel_id].value[2]
                row["type"] = str(telescope_description)
                row["camera"] = str(telescope_description.camera)
                row["optics"] = str(telescope_description.optics)
                row.append()

        if "/Telescope_Type_Information" not in self.file:
            logger.info("Writing telescope type information to table...")
            # Create a row description object for the telescope table
            tel_table_description = table_definitions.TelTableRow

            # Compute maximum number of camera pixels across all telescope types
            max_npix = max([len(x.camera.pix_id) for x in instrument_container.subarray.tel.values()])

            # Add a column field for the pixel position map
            tel_table_description.columns['pixel_positions'] = tables.Float32Col(shape=(max_npix, 2))

            # Create telescope information table
            tel_table = self.file.create_table(
                    self.file.root,
                    'Telescope_Type_Information',
                    tel_table_description,
                    "Table of telescope type information",
                    filters=self.filters,
                    expectedrows=self.expected_tel_types)

            row = tel_table.row

            # Dump properties of all telescope types to table
            for tel_type in instrument_container.subarray.telescope_types:
                tel_ids = instrument_container.subarray.get_tel_ids_for_type(tel_type)
                posx = np.hstack([instrument_container.subarray.tel[tel_ids[0]].camera.pix_x.value,
                                  np.zeros(max_npix - len(instrument_container.subarray.tel[tel_ids[0]].camera.pix_x))])
                posy = np.hstack([instrument_container.subarray.tel[tel_ids[0]].camera.pix_y.value,
                                  np.zeros(max_npix - len(instrument_container.subarray.tel[tel_ids[0]].camera.pix_y))])
                pos = np.stack([posx, posy], axis=1)

                row["type"] = tel_type
                row["num_pixels"] = len(instrument_container.subarray.tel[tel_ids[0]].camera.pix_id)
                row["pixel_positions"] = pos
                row.append()

    def dump_mc_header_info(self, mcheader_container):
        """Dumps ctapipe instrument container to output file.

        Dumps entire contents of MCHeaderContainer to file attributes.

        Args:
            mc_header_container (ctapipe.io.containers.MCHeaderContainer):
                ctapipe container of monte carlo header data (for entire run).
        """
        logger.info("Writing MC header information to file attributes...")

        attributes = self.file.root._v_attrs

        mcheader_dict = mcheader_container.as_dict()

        for field in mcheader_dict:
            attributes[field] = mcheader_dict[field]

    def dump_header_info(self):
        """Dumps all non-ctapipe header data to output file.

        Uses pkg_resources to get software versions in current Python installation.

        """
        logger.info("Writing general header information to file attributes...")

        attributes = self.file.root._v_attrs

        # Record software versions
        if not hasattr(attributes, 'dl1_data_writer_version'):
            attributes['dl1_data_writer_version'] = pkg_resources.get_distribution('dl1-data-handler').version
        if not hasattr(attributes, 'ctapipe_version'):
            attributes['ctapipe_version'] = pkg_resources.get_distribution('ctapipe').version

        # Record list of input files which went into this HDF5 file
        if not hasattr(attributes, 'runlist'):
            attributes.runlist = []
        attributes.runlist.append(self.output_filename)

    def dump_event(self, event_container):
        """Dumps ctapipe event data (event parameters
        and images) to output file.

        Creates '/Events' table in output file if not present, then does the same
        for all required image tables. Finally, writes all event parameters and
        images to tables.

        Args:
            event_container (ctapipe.io.containers.DataContainer):
                ctapipe container of all event data for a given event.
        """
        # Create event table
        if "/Events" not in self.file:
            logger.info("Creating event table...")

            event_table_description = table_definitions.EventTableRow

            for tel_type in event_container.inst.subarray.telescope_types:
                event_table_description.columns[tel_type + '_indices'] = tables.UInt32Col(shape=(len(event_container.inst.subarray.get_tel_ids_for_type(tel_type))))
                event_table_description.columns[tel_type + '_multiplicity'] = tables.UInt32Col()

            event_table = self.file.create_table(self.file.root,
                    'Events',
                    event_table_description,
                    "Table of Event Information",
                    filters=self.filters,
                    expectedrows=self.expected_events)
        else:
            event_table = self.file.root.Events

        # Create image tables (by telescope type)
        for telescope_description in set(event_container.inst.subarray.tels.values()):
            tel_type_name = str(telescope_description)
            if ("/" + tel_type_name) not in self.file:
                logger.info("Creating {} image table...".format(tel_type_name))
                self.image_tables.append(tel_type_name)

                image_shape = (len(telescope_description.camera.pix_id),)

                columns_dict = {
                    "event_index": tables.Int32Col(),
                    "charge": tables.Float32Col(shape=image_shape),
                    "peakpos": tables.Float32Col(shape=image_shape)
                    }

                description = type('description', (tables.IsDescription,), columns_dict)

                # Calculate expected number of rows for compression
                if tel_type in self.expected_images_per_event:
                    expected_rows = self.expected_images_per_event[tel_type_name] * self.expected_events
                else:
                    expected_rows = self.DEFAULT_IMGS_PER_EVENT * self.expected_events

                table = self.file.create_table(
                        self.file.root,
                        tel_type_name,
                        description,
                        "Table of {} images".format(tel_type_name),
                        filters=self.filters,
                        expectedrows=expected_rows)

                # Place blank image at index 0 of all image tables
                image_row = table.row

                image_row['charge'] = np.zeros(image_shape, dtype=np.float32)
                image_row['event_index'] = -1
                image_row['peakpos'] = np.zeros(image_shape, dtype=np.float32)

                image_row.append()
                table.flush()
            else:
                table = self.file.get_node("/" + tel_type_name, classname='Table')

        # Write event data
        event_table = self.file.root.Events
        event_row = event_table.row
        event_index = event_table.nrows

        # Write all general event properties
        event_row['event_id'] = event_container.dl0.event_id
        event_row['obs_id'] = event_container.dl0.obs_id

        # Write all monte carlo event properties if they are present
        if event_container.mc:
            event_row['shower_primary_id'] = event_container.mc.shower_primary_id
            event_row['core_x'] = event_container.mc.core_x.value
            event_row['core_y'] = event_container.mc.core_y.value
            event_row['h_first_int'] = event_container.mc.h_first_int.value
            event_row['x_max'] = event_container.mc.x_max.value
            event_row['mc_energy'] = event_container.mc.energy.value
            event_row['alt'] = event_container.mc.alt.value
            event_row['az'] = event_container.mc.az.value

        # Write images and image index vectors
        image_index_vectors = {tel_type: [] for tel_type in event_container.inst.subarray.telescope_types}

        for tel_type in event_container.inst.subarray.telescope_types:
            for tel_id in sorted(event_container.inst.subarray.get_tel_ids_for_type(tel_type)): # Note that index vectors are sorted by tel ID
                index_vector = image_index_vectors[tel_type]

                if tel_id in event_container.dl1.tel:
                    pixel_vector = event_container.dl1.tel[tel_id].image[0]
                    peaks_vector = event_container.dl1.tel[tel_id].peakpos[0]

                    image_table = self.file.get_node('/' + tel_type, classname='Table')

                    image_row = image_table.row
                    image_index = image_table.nrows

                    image_row['charge'] = pixel_vector
                    image_row['peakpos'] = peaks_vector
                    image_row["event_index"] = event_index

                    image_row.append()
                    image_table.flush()

                    index_vector.append(image_index)
                else:
                    index_vector.append(0)

        for tel_type in image_index_vectors:
            event_row[tel_type + '_indices'] = image_index_vectors[tel_type]
            event_row[tel_type + '_multiplicity'] = sum(index > 0 for index in image_index_vectors[tel_type])

        event_row.append()
        event_table.flush()

    def finalize(self):
        """Method to do any processing that must be done after writing the
        whole file.

        Currently only adds indexes to requested columns.

        """
        # Add all requested PyTables column indexes to tables
        if self.index_columns:
            logger.info("Adding indexed columns...")
            for location, col_name in self.index_columns:
                if location == 'tel':
                    table_names = ["/" + i for i in self.image_tables]
                else:
                    table_names = [location]

                for name in table_names:
                    try:
                        table = self.file.get_node(name, classname='Table')
                        table.cols._f_col(col_name).create_index()
                    except:
                        pass

class DL1DataWriter:
    """Class for managing data processing using event sources and DL1DataDumpers.

    Provides some options for controlling the output file sizes.

    Attributes:
        event_source_class (subclass of ctapipe.io.eventsource.EventSource):
            A subclass of EventSource which will be used to load and yield events
            as DataContainers.
        event_source_settings (dict): A dictionary of kwargs which will be passed
            into the constructor for the EventSource.
        data_dumper_class (subclass of dl1_data_writer.DL1DataDumper):
            A subclass of DL1DataDumper which will be used to write events from
            the EventSource to output files.
        data_dumper_settings (dict): A dictionary of kwargs which will be passed
            into the constructor for the DL1DataDumper.
        calibration_settings (dict): A dictionary of kwargs which will be passed
            into the constructor for ctapipe.calib.camera.CameraCalibrator.
        preselection_cut_function (function): A cut function used to determine
            which events in the input files to write to the output files. Takes a
            ctapipe.io.containers.DataContainer describing a single event and
            returns a boolean indicating if it passes the cut. If None, no cut
            will be applied.
        output_file_size (int): Maximum size of each output file. If the total
            amount of input data requested for a given output file exceeds this
            size, the output will be split across multiple files.
        events_per_file (int): Maximum number of events to write per output file.
            If the total number of input events requested for a given output file
            exceeds this number, the output will be split across multiple files.
    """
    def __init__(self,
                event_source_class=None,
                event_source_settings=None,
                data_dumper_class=CTAMLDataDumper,
                data_dumper_settings=None,
                calibration_settings={
                    'r1_product': 'HESSIOR1Calibrator',
                    'extractor_product': 'NeighbourPeakIntegrator'
                },
                preselection_cut_function=None,
                output_file_size=10737418240,
                events_per_file=None):

        self.event_source_class = event_source_class
        self.event_source_settings = event_source_settings if event_source_settings else {}

        self.data_dumper_class = data_dumper_class
        self.data_dumper_settings = data_dumper_settings if data_dumper_settings else {}

        self.preselection_cut_function = preselection_cut_function

        self.output_file_size = output_file_size
        self.events_per_file = events_per_file

        if self.output_file_size:
            logger.info("Max output file size set at {} bytes. Note that this may increase the number of output files.".format(self.output_file_size))
        if self.events_per_file:
            logger.info("Max number of output events per file set at {}. Note that this may increase the number of output files.".format(self.events_per_file))

        self.calibrator = calib.camera.calibrator.CameraCalibrator(None, None, **calibration_settings)

    def process_data(self, run_list):
        """ Process data from a list of runs in parallel.

        Creates one process for each requested run and executes them all in
        parallel.

        Args:
            run_list (list): A list of dictionaries, each containing two keys,
            'inputs' and 'target'. 'inputs' points to a list of input filenames
            (str) which are to be loaded. 'target' points to an output filename
            (str) to which the data from the input files should be written.

        """
        num_processes = len(run_list)
        logger.info("{} parallel processes requested.".format(num_processes))

        logger.info("Creating processes...")
        jobs = []
        for i in range(0, num_processes):
            process = multiprocessing.Process(target=self._process_data,
                                              args=(run_list[i]['inputs'],
                                                    run_list[i]['target']))
            jobs.append(process)

        logger.info("Starting processing...")

    	# Start all parallel processes
        for j in jobs:
            j.start()

    	# Wait for all processes to complete
        for j in jobs:
            j.join()

        logger.info("Done!")

    def _process_data(self, file_list, output_filename):
        """ Writes a single output file given a list of input files.

        Args:
            file_list (list): A list of input filenames (str) to read data from.
            output_filename (str): Filename of the output file to write data to.

        """
        data_dumper = self.data_dumper_class(output_filename, **self.data_dumper_settings)

        output_file_count = 2
        event_count = 0
        for filename in file_list:
            if self.event_source_class:
                event_source = event_source_class(filename, **self.event_source_settings)
            else:
                event_source = io.event_source(filename)

            # Write all file-level data once
            example_event = next(event_source._generator())
            try:
                data_dumper.dump_header_info()
                data_dumper.dump_instrument_info(example_event.inst)
                data_dumper.dump_mc_header_info(example_event.mcheader)
            except IOError:
                continue

            # Write all events sequentially
            for event in event_source:
                self.calibrator.calibrate(event)
                if (self.preselection_cut_function is not None) and not self.preselection_cut_function(event):
                    continue

                try:
                    data_dumper.dump_event(event)
                    event_count += 1
                except IOError:
                    break

                # Check whether to create another file
                if (self.events_per_file and (event_count >= self.events_per_file)) or (self.output_file_size and (os.path.getsize(data_dumper.output_filename) > self.output_file_size)):
                    # Append a trailing digit to get next filename in sequence
                    output_filename, *extensions = os.path.basename(data_dumper.output_filename).split('.')
                    if re.search(r'\_[0-9]+$', output_filename):
                        output_filename = re.sub(r'\_[0-9]+$', "_" + str(output_file_count), output_filename)
                    else:
                        output_filename = output_filename + "_" + str(output_file_count)

                    for ext in extensions:
                        output_filename = output_filename + '.' + ext

                    # Create a new Data Dumper pointing at a new file
                    # and write file-level data
                    data_dumper = self.data_dumper_class(output_filename, **self.data_dumper_settings)
                    data_dumper.dump_header_info()
                    data_dumper.dump_instrument_info(event.inst)
                    data_dumper.dump_mc_header_info(event.mcheader)
                    # Reset event count and increment file count
                    event_count = 0
                    output_file_count += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=("Read DL1 data via event source into ctapipe containers, then write to a desired output file format." )
        )
    parser.add_argument(
        'runlist',
        help='YAML file containing matched groups of input filenames and output filenames.')
    parser.add_argument(
        '--output_dir',
        help='Path to directory to create output files.',
        default='.')
    parser.add_argument(
        '--config_file',
        help='YAML configuration file for settings.')
    parser.add_argument(
        "--debug",
        help="Print all debug logger messages",
        action="store_true")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG) # Set logger to print all messages

    with open(args.runlist, 'r') as runlist_file:
        runlist = yaml.load(runlist_file)

    for run in runlist:
        run['target'] = os.path.join(args.output_dir, run['target']) # Join output filenames with output directory path

    num_input_files = len([input_file for run in runlist for input_file in run['inputs']])
    num_output_files = len(runlist)

    logger.info("Number of input files in runlist: {}".format(num_input_files))
    logger.info("Number of output files requested: {}".format(num_output_files))

    # load options from config file and create DL1 Data Writer object
    if args.config_file is not None:
        logger.info("Reading config file {}...".format(args.config_file))
        with open(args.config_file, 'r') as config_file:
            config = yaml.load(config_file)

        logger.info("Config file {} loaded.".format(args.config_file))
        logger.info(yaml.dump(config, default_flow_style=False, default_style='' ))

        writer_settings = config['Data Writer']['Settings']

        event_source_settings = config['Event Source']['Settings']

        dumper_name = config['Data Dumper']['Type']
        dumper_settings = config['Data Dumper']['Settings']

        # Locate DL1DataDumper subclass to use for writing output
        dumpers = {i.__name__: i for i in DL1DataDumper.__subclasses__()}
        if dumper_name in dumpers:
            data_dumper_class = dumpers[dumper_name]
        else:
            raise ValueError("No subclass of DL1DataDumper with name: {}".format(dumper_name))

        data_writer = DL1DataWriter(event_source_class=None, event_source_settings=event_source_settings, data_dumper_class=data_dumper_class, data_dumper_settings=dumper_settings, **writer_settings)
    else:
        logger.info("No config file provided, using default settings")
        data_writer = DL1DataWriter()

    data_writer.process_data(runlist)
