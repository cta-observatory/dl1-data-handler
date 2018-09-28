# -*- coding: utf-8 -*-
"""
Module for main ImageExtractor class and command line tool.
"""
import pkg_resources
import random
import argparse
import pickle as pkl
import logging
import os
import math
import yaml

import numpy as np
import tables

import ctapipe.instrument
import ctapipe.io
import ctapipe.calib

import image
import row_types

logger = logging.getLogger(__name__)


class ImageExtractor:
    """

    Class to handle reading in simtel files and writing to HDF5.

    Requires an output file path (.h5) to write/append to, and a
    pickled bins cuts dictionary to apply bins and cuts.

    Configuration settings can be initialized with a config object
    (from validating a ConfigObj configuration file), which is recommended,
    or by using the normal constructor and passing parameters manually (not
    recommended because settings are not validated and must be manually
    checked for agreement with allowed options.

    Once initialized, can read in simtel files and write to the output file
    in accordance with the provided configuration settings.


    """

    OPTICS_TYPES =  set(['{}-{}'.format(v[0], v[1]) if v[1] else v[0]
        for v in ctapipe.instrument.optics._FOCLEN_TO_TEL_INFO.values()]
        + [v[0] for v in ctapipe.instrument.optics._FOCLEN_TO_TEL_INFO.values()])
    CAM_TYPES = image.CAM_TYPES
    CAM_NUM_PIXELS = image.CAM_NUM_PIXELS
    IMAGE_SHAPES = image.IMAGE_SHAPES

    METADATA_FIELDS = {
        "CORSIKA_ver": "mcheader.corsika_version",
        "simtel_ver": "mcheader.simtel_version",
        "prod_site_alt": "mcheader.prod_site_alt",
        "prod_site_coord": "mcheader.prod_site_coord",
        "prod_site_B_field": "mcheader.prod_site_B_total",
        "prod_site_array": "mcheader.prod_site_array",
        "prod_site_subarray": "mcheader.prod_site_subarray",
        "particle_type": "event.mc.shower_primary_id",
        "zenith": "mcheader.run_array_direction[1]",
        "azimuth": "mcheader.run_array_direction[0]",
        "view_cone": None,
        "spectral_index": "mcheader.spectral_index",
        "E_min": "mcheader.energy_range_min",
        "E_max": "mcheader.energy_range_min",
    }

    ALLOWED_CUT_PARAMS = {}
    DEFAULT_CUTS_DICT = {}

    DEFAULT_IMGS_PER_EVENT = 1.0

    def __init__(self,
            output_path,
            ED_cuts_dict=None,
            storage_mode='tel_type',
            tel_type_list=['LST:LSTCam', 'MST:FlashCam', 'MST:NectarCam', 'MST-SCT:SCTCam', 'SST:DigiCam', 'SST:ASTRICam', 'SST:CHEC'],
            img_mode='1D',
            img_channels=1,
            include_timing=True,
            img_scale_factors={'NectarCam': 1},
            img_dtypes={'LSTCam': 'float32', 'NectarCam': 'float32', 'FlashCam': 'float32', 'SCTCam': 'float32',
                        'DigiCam': 'float32', 'ASTRICam': 'float32', 'CHEC': 'float32'},
            img_dim_order='channels_last',
            cuts_dict=DEFAULT_CUTS_DICT,
            comp_lib='lzo',
            comp_lvl=1,
            expected_tel_types=10,
            expected_tels=300,
            expected_events=7000,
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
            ('/Event_Info', 'mc_energy'),
            ('/Event_Info', 'alt'),
            ('/Event_Info', 'az'),
            ('/LST:LSTCam', 'event_index'),
            ('/MST:NectarCam', 'event_index'),
            ('/MST:FlashCam', 'event_index'),
            ('/MST-SCT:SCTCam', 'event_index'),
            ('/SST:DigiCam', 'event_index'),
            ('/SST:ASTRICam', 'event_index'),
            ('/SST:CHEC', 'event_index')
            ]):

        """Constructor for ImageExtractor
        """

        if os.path.isdir(os.path.dirname(output_path)):
            self.output_path = output_path
        else:
            raise ValueError('Output file directory does not exist: {}.'.format(os.path.dirname(output_path)))

        self.ED_cuts_dict = ED_cuts_dict

        if storage_mode in ['tel_type','tel_id']:
            self.storage_mode = storage_mode
        else:
            raise ValueError('Invalid storage mode: {}.'.format(storage_mode))

        for tel_type in tel_type_list:
            optics_type, cam_type = tel_type.split(":")
            if optics_type not in self.OPTICS_TYPES:
                raise ValueError('Invalid optics type: {}'.format(optics_type))
            if cam_type not in self.CAM_TYPES:
                raise ValueError('Invalid camera type: {}'.format(cam_type))
        self.tel_type_list = tel_type_list

        if img_mode in ['1D', '2D']:
            self.img_mode = img_mode
        else:
            raise ValueError('Invalid img_mode: {}.'.format(img_mode))

        if img_channels > 0:
            self.img_channels = img_channels
        else:
            raise ValueError('Image channels must be > 0.')

        self.include_timing = include_timing

        self.img_scale_factors = img_scale_factors
        self.img_dtypes = img_dtypes

        if img_dim_order in ['channels_first', 'channels_last']:
            self.img_dim_order = img_dim_order
        else:
            raise ValueError('Invalid dimension ordering: {}.'.format(img_dim_order))

        self.trace_converter= image.TraceConverter(
            self.img_dtypes,
            self.img_dim_order,
            self.img_channels,
            self.img_scale_factors)

        self.cuts_dict = cuts_dict

        self.filters = tables.Filters(complevel=comp_lvl, complib=comp_lib)

        self.expected_tel_types = expected_tel_types
        self.expected_tels = expected_tels
        self.expected_events = expected_events
        self.expected_images_per_event = expected_images_per_event

        self.index_columns = index_columns

    def select_telescopes(self, data_file):
        """Method to read telescope info from a given simtel file
        and select the desired telescopes indicated by self.tel_type_mode.

        Parameters
        ----------
        data_file: str
            The string path (relative or absolute) to the input simtel.gz
            file.

        Returns
        -------
        tuple(dict(str:list(int)),int)
            Returns a dictionary of selected telescopes of format
            dict(<tel_type>:list(<tel_id>)) and an int number of total
            telescopes

        """

        logger.info("Collecting telescope types...")

        event = next(iter(ctapipe.io.event_source(data_file)))

        self.all_tels = {tel_type: sorted(event.inst.subarray.get_tel_ids_for_type(tel_type)) for tel_type in event.inst.subarray.telescope_types}

        # select telescopes by type
        logger.info("Selected telescope types: [")
        for tel_type in self.tel_type_list:
            logger.info("{},".format(tel_type))
        logger.info("]")

        selected_tels = {}
        for tel_type in self.tel_type_list:
            if tel_type not in event.inst.subarray.telescope_types:
                logger.warning('Telescope type {} not found in data. Ignoring...'.format(tel_type))
            else:
                selected_tels[tel_type] = self.all_tels[tel_type]

        total_num_tel_selected = 0
        for tel_type in self.all_tels:
            if tel_type in selected_tels:
                num_tel_selected = len(selected_tels[tel_type])
            else:
                num_tel_selected = 0
            logger.info(tel_type + ": " + str(num_tel_selected) +
                        " out of " + str(len(self.all_tels[tel_type])) +
                        " telescopes selected.")
            total_num_tel_selected += num_tel_selected

        return selected_tels, total_num_tel_selected

    def write_metadata(self, HDF5_file, data_file):

        logger.info("Checking/writing metadata...")

        event = next(iter(ctapipe.io.event_source(data_file)))

        attributes = HDF5_file.root._v_attrs

        # Add major software versions
        attributes['image_extractor_ver'] = pkg_resources.get_distribution('image-extractor').version
        attributes['ctapipe_ver'] = pkg_resources.get_distribution('ctapipe').version

        # Add shower metadata fields
        for field in self.METADATA_FIELDS:
            try:
                value = getattr(event, field) if field else None

                # If not present in HDF5 file header, add attribute. Else, compare for equality
                if not attributes.__contains__(field):
                    attributes[field] = value
                else:
                    if attributes[field] != value:
                        raise ValueError("Metadata field {} for current simtel file does not match output file: {} vs {}" \
                                         .format(field, value, attributes[field]))
            except:
                pass

        run_file_name = os.path.basename(data_file)
        if not attributes.__contains__("runlist"):
            attributes.runlist = [run_file_name]
        else:
            runlist = attributes.runlist
            runlist.append(run_file_name)
            attributes.runlist = runlist

    def process_data(self, data_file, max_events=None):
        """Main method to read a simtel.gz data file, process the event data,
        and write it to a formatted HDF5 data file.
        """
        logger.info("Preparing HDF5 file structure...")

        with tables.open_file(self.output_path, mode="a", title="Output File") as f:
            try:

                self.write_metadata(f, data_file)

                selected_tels, num_tel = self.select_telescopes(data_file)

                event = next(iter(ctapipe.io.event_source(data_file)))

                # create and fill array information table
                if not f.__contains__('/Array_Info'):
                    arr_table = f.create_table(f.root,
                                    'Array_Info',
                                    row_types.Array,
                                    ("Table of array data"),
                                    filters=self.filters,
                                    expectedrows=self.expected_tels)

                    arr_row = arr_table.row

                    for tel_type in selected_tels:
                        for tel_id in selected_tels[tel_type]:
                            arr_row["tel_id"] = tel_id
                            arr_row["tel_x"] = event.inst.subarray.positions[tel_id].value[0]
                            arr_row["tel_y"] = event.inst.subarray.positions[tel_id].value[1]
                            arr_row["tel_z"] = event.inst.subarray.positions[tel_id].value[2]
                            arr_row["tel_type"] = tel_type
                            arr_row["run_array_direction"] = event.mcheader.run_array_direction
                            arr_row.append()

                # create and fill telescope information table
                if not f.__contains__('/Telescope_Info'):
                    tel_table = f.create_table(f.root,
                                    'Telescope_Info',
                                    row_types.Tel,
                                    ("Table of telescope data"),
                                    filters=self.filters,
                                    expectedrows=self.expected_tel_types)

                    descr = tel_table.description._v_colobjects
                    descr2 = descr.copy()

                    max_npix = self.CAM_NUM_PIXELS[max(self.CAM_NUM_PIXELS, key=self.CAM_NUM_PIXELS.get)]
                    descr2["pixel_pos"] = tables.Float32Col(shape=(2, max_npix))

                    tel_table2 = f.create_table(
                            f.root,
                            'temp',
                            descr2,
                            "Table of telescope data",
                            filters=self.filters,
                            expectedrows=self.expected_tel_types)
                    tel_table.attrs._f_copy(tel_table2)
                    tel_table.remove()
                    tel_table2.move(f.root, 'Telescope_Info')

                    tel_row = tel_table2.row

                    # add units to table attributes
                    random_tel_type = random.choice(list(selected_tels.keys()))
                    random_tel_id = random.choice(selected_tels[random_tel_type])
                    tel_table2.attrs.tel_pos_units = str(event.inst.subarray.positions[random_tel_id].unit)

                    for tel_type in selected_tels:
                        random_tel_id = random.choice(selected_tels[tel_type])
                        tel_row["tel_type"] = tel_type
                        tel_row["num_pixels"] = len(event.inst.subarray.tel[random_tel_id].camera.pix_id)
                        posx = np.hstack([event.inst.subarray.tel[random_tel_id].camera.pix_x.value,
                                          np.zeros(max_npix - len(event.inst.subarray.tel[random_tel_id].camera.pix_x))])
                        posy = np.hstack([event.inst.subarray.tel[random_tel_id].camera.pix_y.value,
                                          np.zeros(max_npix - len(event.inst.subarray.tel[random_tel_id].camera.pix_y))])
                        tel_row["pixel_pos"] = [posx, posy]
                        tel_row.append()

                # create event table
                if not f.__contains__('/Event_Info'):
                    table = f.create_table(f.root,
                            'Event_Info',
                            row_types.Event,
                            "Table of Event metadata",
                            filters=self.filters,
                            expectedrows=self.expected_events)

                    descr = table.description._v_colobjects
                    descr2 = descr.copy()

                    if self.storage_mode == 'tel_type':
                        for tel_type in selected_tels:
                            descr2[tel_type + '_indices'] = tables.Int32Col(shape=(len(selected_tels[tel_type])))
                    elif self.storage_mode == 'tel_id':
                        descr2["indices"] = tables.Int32Col(shape=(num_tel))

                    table2 = f.create_table(
                            f.root,
                            'temp',
                            descr2,
                            "Table of Events",
                            filters=self.filters,
                            expectedrows=self.expected_events)
                    table.attrs._f_copy(table2)
                    table.remove()
                    table2.move(f.root, 'Event_Info')

                    # add units to table attributes
                    table2.attrs.core_pos_units = str(event.mc.core_x.unit)
                    table2.attrs.h_first_int_units = str(event.mc.h_first_int.unit)
                    table2.attrs.mc_energy_units = str(event.mc.energy.unit)
                    table2.attrs.alt_az_units = str(event.mc.alt.unit)

                # create image tables
                for tel_type in selected_tels:
                    cam_type = tel_type.split(':')[1]
                    if self.img_mode == '2D':
                        img_width = self.IMAGE_SHAPES[cam_type][0] * self.img_scale_factors[cam_type]
                        img_length = self.IMAGE_SHAPES[cam_type][1] * self.img_scale_factors[cam_type]

                        if self.img_dim_order == 'channels_first':
                            array_shape = (self.img_channels, img_width, img_length)
                        elif self.img_dim_order == 'channels_last':
                            array_shape = (img_width, img_length, self.img_channels)

                        np_type = np.dtype(np.dtype(self.img_dtypes[cam_type]), array_shape)
                        columns_dict = {"image": tables.Col.from_dtype(np_type), "event_index": tables.Int32Col()}

                    elif self.img_mode == '1D':
                        array_shape = (self.CAM_NUM_PIXELS[cam_type],)
                        np_type = np.dtype((np.dtype(self.img_dtypes[cam_type]), array_shape))

                        columns_dict = {"image_charge": tables.Col.from_dtype(np_type), "event_index": tables.Int32Col()}
                        if self.include_timing:
                            columns_dict["image_peak_times"] = tables.Col.from_dtype(np_type)

                    description = type('description', (tables.IsDescription,), columns_dict)

                    if self.storage_mode == 'tel_type':
                        if not f.__contains__('/' + tel_type):

                            if tel_type in self.expected_images_per_event:
                                expected_rows = self.expected_images_per_event[tel_type] * self.expected_events
                            else:
                                expected_rows = DEFAULT_IMGS_PER_EVENT * self.expected_events

                            table = f.create_table(
                                    f.root,
                                    tel_type,
                                    description,
                                    "Table of {} images".format(tel_type),
                                    filters=self.filters,
                                    expectedrows=expected_rows)

                            # append blank image at index 0
                            image_row = table.row

                            if self.img_mode == '2D':
                                image_row['image'] = self.trace_converter.convert(None, None, cam_type)

                            elif self.img_mode == '1D':
                                shape = (image.CAM_NUM_PIXELS[cam_type],)
                                image_row['image_charge'] = np.zeros(shape, dtype=self.img_dtypes[cam_type])
                                image_row['event_index'] = -1
                                if self.include_timing:
                                    image_row['image_peak_times'] = np.zeros(shape, dtype=self.img_dtypes[cam_type])

                            image_row.append()
                            table.flush()

                    elif self.storage_mode == 'tel_id':
                        for tel_id in selected_tels[tel_type]:
                            if not f.__contains__('T' + str(tel_id)):

                                if tel_type in self.expected_images_per_event:
                                    expected_rows = self.expected_images_per_event[tel_type] * self.expected_events / len(self.all_tels[tel_type])
                                else:
                                    expected_rows = DEFAULT_IMGS_PER_EVENT * self.expected_events / len(self.all_tels[tel_type])

                                table = f.create_table(f.root,
                                        'T' + str(tel_id),
                                        description,
                                        "Table of T{} images".format(str(tel_id)),
                                        filters=self.filters,
                                        expectedrows=expected_rows)

                                # append blank image at index 0
                                image_row = table.row

                                if self.img_mode == '2D':
                                    image_row['image'] = self.trace_converter.convert(None, None, cam_type)

                                elif self.img_mode == '1D':
                                    shape = (image.CAM_NUM_PIXELS[tel_type],)
                                    image_row['image_charge'] = np.zeros(shape, dtype=self.img_dtypes[cam_type])
                                    image_row['event_index'] = -1
                                    if self.include_timing:
                                        image_row['image_peak_times'] = np.zeros(shape, dtype=self.img_dtypes[cam_type])

                                image_row.append()
                                table.flush()

                # specify calibration and other processing options
                cal = ctapipe.calib.CameraCalibrator(None, None)

                logger.info("Processing events...")

                event_count = 0
                passing_count = 0

                source = ctapipe.io.event_source(data_file, allowed_tels=[j for i in selected_tels for j in selected_tels[i]],
                                             max_events=max_events)

                for event in source:
                    event_count += 1

                    if self.cuts_dict:
                        if self.ED_cuts_dict is not None:
                            logger.warning(
                                'Warning: Both ED_cuts_dict and cuts dictionary found. Using cuts dictionary instead.')

                        if test_cuts(event, cuts_dict):
                            passing_count += 1
                        else:
                            continue
                    else:
                        if self.ED_cuts_dict is not None:
                            if (event.r0.run_id, event.r0.event_id) in self.ED_cuts_dict:
                                bin_number, reconstructed_energy = self.ED_cuts_dict[
                                    (event.r0.run_id, event.r0.event_id)]
                                passing_count += 1
                            else:
                                continue

                    cal.calibrate(event)

                    table = f.root.Event_Info
                    table.flush()
                    event_row = table.row
                    event_index = table.nrows

                    if self.storage_mode == 'tel_type':
                        tel_index_vectors = {tel_type: [] for tel_type in selected_tels}
                    elif self.storage_mode == 'tel_id':
                        all_tel_index_vector = []

                    for tel_type in selected_tels.keys():
                        for tel_id in sorted(selected_tels[tel_type]):
                            if self.storage_mode == 'tel_type':
                                index_vector = tel_index_vectors[tel_type]
                            elif self.storage_mode == 'tel_id':
                                index_vector = all_tel_index_vector

                            if tel_id in event.r0.tels_with_data:
                                pixel_vector = event.dl1.tel[tel_id].image[0]
                                logger.debug('Storing image from tel_type {} ({} pixels)'.format(tel_type, len(pixel_vector)))
                                if self.include_timing:
                                    peaks_vector = event.dl1.tel[tel_id].peakpos[0]
                                else:
                                    peaks_vector = None

                                if self.storage_mode == 'tel_type':
                                    table = f.get_node('/' + tel_type, classname='Table')
                                elif self.storage_mode == 'tel_id':
                                    table = f.get_node('/T' + str(tel_id) , classname='Table')
                                next_index = table.nrows
                                image_row = table.row

                                if self.img_mode == '2D':
                                    image_row['image'] = self.trace_converter.convert(pixel_vector, peaks_vector, cam_type)

                                elif self.img_mode == '1D':
                                    image_row['image_charge'] = pixel_vector
                                    if self.include_timing:
                                        image_row['image_peak_times'] = peaks_vector

                                image_row["event_index"] = event_index

                                image_row.append()
                                index_vector.append(next_index)
                                table.flush()
                            else:
                                index_vector.append(0)

                    if self.storage_mode == 'tel_type':
                        for tel_type in tel_index_vectors:
                            event_row[tel_type + '_indices'] = tel_index_vectors[tel_type]
                    elif self.storage_mode == 'tel_id':
                        event_row['indices'] = all_tel_index_vector

                    event_row['event_number'] = event.r0.event_id
                    event_row['run_number'] = event.r0.obs_id
                    event_row['particle_id'] = event.mc.shower_primary_id
                    event_row['core_x'] = event.mc.core_x.value
                    event_row['core_y'] = event.mc.core_y.value
                    event_row['h_first_int'] = event.mc.h_first_int.value
                    event_row['mc_energy'] = event.mc.energy.value
                    event_row['alt'] = event.mc.alt.value
                    event_row['az'] = event.mc.az.value

                    event_row.append()
                    table.flush()
                    
                f.root.Event_Info.flush()
                total_num_events = f.root.Event_Info.nrows

                # Add all indexes
                for location, col_name in self.index_columns:
                    try:
                        table = f.get_node(location, classname='Table')
                        table.cols._f_col(col_name).create_index()
                    except:
                        pass


                f.close()

                logger.info("{} events read in file".format(event_count))
                logger.info("{} total events in output file.".format(total_num_events))
                if self.cuts_dict or self.ED_cuts_dict:
                    logger.info("{} events passed cuts/written to file".format(passing_count))
                logger.info("Done!")
            except:
                os.remove(self.output_path)
                raise

    def set_cuts(self, cuts_dict):
        for i in cuts_dict:
            if i not in ALLOWED_CUT_PARAMS:
                raise ValueError("Invalid cut parameter: {}.".format(i))

        self.cuts_dict = cuts_dict

    def test_cuts(event, cuts):
        for cut in cuts:
            cut_param = eval('event.' + ALLOWED_CUT_PARAMS[cut])
            cut_min = cuts[cut][0]
            cut_max = cuts[cut][1]

            if cut_min is not None and cut_param < cut_min:
                return False

            if cut_max is not None and cut_param >= cut_max:
                return False

        return True

    def shuffle_data(self, h5_file, random_seed):

        # open input hdf5 file
        f = tables.open_file(h5_file, mode="r+", title="Input file")

        table = f.root.Events
        descr = table.description

        num_events = table.shape[0]
        new_indices = [i for i in range(num_events)]
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(new_indices)

        table_new = f.create_table(
            f.root,
            'Events_temp',
            descr,
            "Table of events",
            filters=self.filters,
            expectedrows=self.expected_events)

        for i in range(num_events):
            table_new.append([tuple(table[new_indices[i]])])

        table.remove()
        table_new.move(f.root, 'Events')
        f.close()

    def split_data(self, h5_file, splits):

        assert len(splits) == 3, "Must provide 3 values for train, validation, test"

        split_sum = 0
        for i in splits:
            split_sum += i
        assert math.isclose(split_sum, 1.0, rel_tol=1e-5), "Split fractions do not add up to 1"

        # open input hdf5 file
        f = tables.open_file(h5_file, mode="r+", title="Input file")

        table = f.root.Events
        descr = table.description

        num_events = table.shape[0]
        indices = range(num_events)
        i = 0

        split_names = ['Training', 'Validation', 'Test']

        for j in range(len(splits)):
            if splits[j] != 0.0:
                table_new = f.create_table(
                    f.root,
                    'Events_' +
                    split_names[j],
                    descr,
                    "Table of " +
                    split_names[j] +
                    " Events",
                    filters=self.filters,
                    expectedrows=self.expected_events*splits[j])

                split_fraction = splits[j]

                if i + int(split_fraction * num_events) <= num_events:
                    split_indices = indices[
                                    i:i + int(split_fraction * num_events)]
                else:
                    split_indices = indices[i:num_events]

                for k in split_indices:
                    table_new.append([tuple(table[k])])

                i += int(split_fraction * num_events)

        table.remove()
        f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=("Load image data and event parameters from a simtel file"
                     "into a formatted HDF5 file."))
    parser.add_argument(
        'run_list',
        help='text file run list containing a list of simtel files to process (1 per line)')
    parser.add_argument(
        'hdf5_path',
        help='path of output HDF5 file, or currently existing file to append to')
    parser.add_argument(
        '--config_file_path',
        help='path of configuration file for settings')
    parser.add_argument(
        '--ED_cuts_dict_file',
        help='path of .pkl file containing cuts dictionary from EventDisplay')
    parser.add_argument(
        "--info",
        help="print info messages",
        action="store_true")
    parser.add_argument(
        "--debug",
        help="print debug messages",
        action="store_true")
    parser.add_argument(
        "--max_events",
        help="set a maximum number of events to process from each file",
        type=int)
    parser.add_argument(
        "--shuffle",
        help="shuffle output data file. Can pass optional random seed", nargs='?', const='default', action='store',
        default=None)
    parser.add_argument(
        "--split",
        help="Split output data file into separate event tables. Pass optional list of splits (training, validation, test)",
        nargs='?', const=[0.9, 0.1, 0.0], action='store', default=None)

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.info:
        logger.setLevel(logging.INFO)

    # load options from config file
    if args.config_file_path is not None:
        with open(args.config_file_path, 'r') as config_file:
            config = yaml.load(config_file)
        options = {k: v for x in config.values() for k, v in x.items()}

        if options['ED_cuts_dict'] is not None:
            options['ED_cuts_dict'] = pkl.load(open(options['ED_cuts_dict'], "rb"))

        extractor = ImageExtractor(args.hdf5_path, **options)
    else:
        # default arguments
        extractor = ImageExtractor(args.hdf5_path)

    run_list = []
    with open(args.run_list) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                run_list.append(line)

    logger.info("Number of data files in runlist: {}".format(len(run_list)))

    for i, data_file in enumerate(run_list):
        logger.info("Processing file {}/{}".format(i + 1, len(run_list)))
        extractor.process_data(data_file, args.max_events)

    if args.shuffle:
        extractor.shuffle_data(args.hdf5_path, args.shuffle)

    if args.split:
        extractor.split_data(args.hdf5_path, args.split)
