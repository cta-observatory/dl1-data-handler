"""
image_extractor for writing processed data from ctapipe
containers to HDF5 files.
Also includes helper function make_sct_image_array()
and Pytables custom row description classes Event, Tel.
"""

import argparse
import pickle as pkl
import logging

from configobj import ConfigObj
from validate import Validator
import numpy as np
from tables import open_file, IsDescription, Int16Atom, Int16Col, UInt8Col, UInt16Col, UInt32Col, Float32Col, Float64Col, StringCol
from ctapipe.io.hessio import hessio_event_source
from ctapipe.calib import CameraCalibrator

import config

__all__ = ['make_sct_image_array',
           'image_extractor']

logger = logging.getLogger(__name__)

config_spec = config.config_spec

# harcoded telescope constants
# TODO: find more natural way of handling this?
# TODO: find correct values
LST_IMAGE_WIDTH = 0
LST_IMAGE_LENGTH = 0
SCT_IMAGE_WIDTH = 120
SCT_IMAGE_LENGTH = 120
SST_IMAGE_WIDTH = 0
SST_IMAGE_LENGTH = 0
LST_NUM_PIXELS = 1855
SCT_NUM_PIXELS = 11328
SST_NUM_PIXELS = 0  # update


class Event(IsDescription):
    """
    Row descriptor class for Pytables event table.
    """
    event_number = UInt32Col()
    run_number = UInt32Col()
    gamma_hadron_label = UInt8Col()
    MC_energy = Float64Col()
    reconstructed_energy = Float32Col()


class Tel(IsDescription):
    """
    Row descriptor class for Pytables telescope data table.
    """
    tel_id = UInt8Col()
    tel_x = Float32Col()
    tel_y = Float32Col()
    tel_z = Float32Col()
    tel_type = StringCol(8)


def make_sct_image_array(
        pixels_vector,
        peaks_vector,
        scale_factor,
        img_dtype,
        dim_order,
        channels):
    """
    Converter from ctapipe image pixel vector, peak position vector to numpy array format.
    """
    # hardcoded to correct SCT values
    # TODO: find more natural way to handle these?
    NUM_PIXELS = 11328
    CAMERA_DIM = (120, 120)
    ROWS = 15
    MODULE_DIM = (8, 8)
    MODULE_SIZE = MODULE_DIM[0] * MODULE_DIM[1]
    MODULES_PER_ROW = [5, 9, 11, 13, 13, 15, 15, 15, 15, 15, 13, 13, 11, 9, 5]

    # counting from the bottom row, left to right
    MODULE_START_POSITIONS = [
        (((CAMERA_DIM[0] -
           MODULES_PER_ROW[j] *
           MODULE_DIM[0]) /
          2) +
         (
            MODULE_DIM[0] *
            i),
            j *
            MODULE_DIM[1]) for j in range(ROWS) for i in range(
            MODULES_PER_ROW[j])]

    if dim_order == 'channels_first':
        im_array = np.zeros([channels, CAMERA_DIM[0] * scale_factor,
                             CAMERA_DIM[1] * scale_factor], dtype=img_dtype)
    elif dim_order == 'channels_last':
        im_array = np.zeros([CAMERA_DIM[0] *
                             scale_factor, CAMERA_DIM[1] *
                             scale_factor, channels], dtype=img_dtype)

    if pixels_vector is not None:
        try:
            assert len(pixels_vector) == NUM_PIXELS
        except AssertionError as err:
            logger.exception(
                "Length of pixel vector does not match SCT num pixels.")
            raise err
        image_exists = True
    else:
        image_exists = False

    if peaks_vector is not None:
        try:
            assert len(peaks_vector) == NUM_PIXELS
        except AssertionError as err:
            logger.exception(
                "Length of peaks vector does not match SCT num pixels.")
            raise err
        include_timing = True
    else:
        include_timing = False

    if image_exists:
        pixel_index = 0
        for (x_start, y_start) in MODULE_START_POSITIONS:
            for i in range(MODULE_SIZE):
                x = (int(x_start + int(i / MODULE_DIM[0]))) * scale_factor
                y = (y_start + i % MODULE_DIM[1]) * scale_factor

                scaled_region = [(x + i, y + j) for i in range(0, scale_factor)
                                 for j in range(0, scale_factor)]

                for (x_coord, y_coord) in scaled_region:
                    if dim_order == 'channels_first':
                        im_array[0, x_coord,
                                 y_coord] = pixels_vector[pixel_index]
                        if include_timing:
                            im_array[1, x_coord,
                                     y_coord] = peaks_vector[pixel_index]
                    elif dim_order == 'channels_last':
                        im_array[x_coord, y_coord,
                                 0] = pixels_vector[pixel_index]
                        if include_timing:
                            im_array[x_coord, y_coord,
                                     1] = peaks_vector[pixel_index]

                pixel_index += 1

    return im_array


def image_extractor(data_file_path, output_file_path, bins_cuts_dict, config):
    """
    Function to read and write data from ctapipe containers to HDF5
    """
    MODE = config['mode']
    IMG_MODE = config['image']['mode']
    STORAGE_MODE = config['storage_mode']
    IMG_SCALE_FACTOR = config['image']['scale_factor']
    IMG_DTYPE = config['image']['dtype']
    IMG_DIM_ORDERING = config['image']['dim_ordering']
    ENERGY_BIN_UNITS = config['energy_bins']['units']

    logger.info("Mode: ", MODE)
    logger.info("Image mode: ", IMG_MODE)
    logger.info("File storage mode: ", STORAGE_MODE)
    logger.info("Image scale factor: ", IMG_SCALE_FACTOR)
    logger.info("Image array type: ", IMG_DTYPE)
    logger.info("Image dim order: ", IMG_DIM_ORDERING)

    logger.info("Getting telescope types...")

    # collect telescope lists
    source_temp = hessio_event_source(data_file_path, max_events=1)

    LST_list = []
    SCT_list = []
    SST_list = []

    for event in source_temp:
        for i in event.inst.telescope_ids:
            if event.inst.num_pixels[i] == SCT_NUM_PIXELS:
                SCT_list.append(i)
            elif event.inst.num_pixels[i] == LST_NUM_PIXELS:
                LST_list.append(i)
            else:
                SST_list.append(i)

        all_tels = {'SST': SST_list, 'SCT': SCT_list, 'LST': LST_list}

    # select telescopes by type
    selected = {}
    TEL_MODE = config['telescope']['type_mode']
    if TEL_MODE == 'SST':
        selected['SST'] = SST_list
    elif TEL_MODE == 'SCT':
        selected['SCT'] = SCT_list
    elif TEL_MODE == 'LST':
        selected['LST'] = LST_list
    elif TEL_MODE == 'SCT+LST':
        selected['LST'] = LST_list
        selected['SCT'] = SCT_list
    elif TEL_MODE == 'SST+SCT':
        selected['SST'] = SST_list
        selected['SCT'] = SCT_list
    elif TEL_MODE == 'SST+LST':
        selected['LST'] = LST_list
        selected['SST'] = SST_list
    elif TEL_MODE == 'ALL':
        selected['LST'] = LST_list
        selected['SCT'] = SCT_list
        selected['SST'] = SST_list
    else:
        logger.error("Telescope selection mode invalid.")
        raise ValueError('Telescope selection mode not recognized.')

    logger.info("Telescope Mode: ", TEL_MODE)

    NUM_TEL = 0

    for i in selected.keys():
        logger.info(i + ": " + str(len(selected[i])) + " out of " + str(
            len(all_tels[i])) + " telescopes selected.")
        NUM_TEL += len(selected[i])

    logger.info("Loading additional configuration settings...")

    if MODE == 'gh_class':

        e_min = float(config['energy_bins']['min'])
        e_max = float(config['energy_bins']['max'])
        e_bin_size = float(config['energy_bins']['bin_size'])
        num_e_bins = int((e_max - e_min) / e_bin_size)
        energy_bins = [(e_min + i * e_bin_size, e_min + (i + 1) *
                        e_bin_size) for i in range(num_e_bins)]

    elif MODE == 'energy_recon':

        erec_min = float(config['energy_recon']['bins']['min'])
        erec_max = float(config['energy_recon']['bins']['max'])
        erec_bin_size = float(config['energy_recon']['bins']['bin_size'])
        num_erec_bins = int((erec_max - erec_min) / erec_bin_size)
        energy_recon_bins = [(erec_min +
                              i *
                              erec_bin_size, erec_min +
                              (i +
                               1) *
                              erec_bin_size) for i in range(num_erec_bins)]

    if IMG_MODE == 'PIXELS_3C':
        IMAGE_CHANNELS = 3
        INCLUDE_TIMING = False
    elif IMG_MODE == 'PIXELS_1C':
        IMAGE_CHANNELS = 1
        INCLUDE_TIMING = False
    elif IMG_MODE == 'PIXELS_TIMING_2C':
        IMAGE_CHANNELS = 2
        INCLUDE_TIMING = True
    elif IMG_MODE == 'PIXELS_TIMING_3C':
        IMAGE_CHANNELS = 3
        INCLUDE_TIMING = True
    else:
        logger.error("Invalid image format (IMG_MODE).")
        raise ValueError('Image processing mode not recognized.')

    logger.info("Preparing HDF5 file structure...")

    f = open_file(output_file_path, mode="a", title="Data File")

    # prep data into bins based on mode (by energy bin for g/h classification,
    # 1 group for energy reconstruction)
    if MODE == 'gh_class':
        # create groups for each energy bin, if not already existing
        datasets = []
        for i in range(len(energy_bins)):
            if not f.__contains__("/E" + str(i)):
                group = f.create_group(
                    "/", "E" + str(i), 'Energy bin group' + str(i))
                group._v_attrs.min_energy = energy_bins[i][0]
                group._v_attrs.max_energy = energy_bins[i][1]
                group._v_attrs.units = ENERGY_BIN_UNITS
            row_str = 'f.root.E{}'.format(str(i))
            where = eval(row_str)
            datasets.append(where)
    elif MODE == 'energy_recon':
        datasets = [f.root]

    dataset_tables = []

    # create and fill telescope table

    # create table for telescope positions
    if not f.__contains__('/Tel_Table'):
        tel_pos_table = f.create_table(
            "/", 'Tel_Table', Tel, "Table of telescope ids, positions, and types")
        tel_row = tel_pos_table.row

        source_temp = hessio_event_source(data_file_path, max_events=1)

        for event in source_temp:

            for i in selected.keys():
                for j in selected[i]:
                    tel_row["tel_id"] = j
                    tel_row["tel_x"] = event.inst.tel_pos[j].value[0]
                    tel_row["tel_y"] = event.inst.tel_pos[j].value[1]
                    tel_row["tel_z"] = event.inst.tel_pos[j].value[2]
                    tel_row["tel_type"] = i

                    tel_row.append()

    # create table for events
    for d in datasets:
        if not d.__contains__('Events'):
            table = f.create_table(
                d, 'Events', Event, "Table of event records")
            descr = table.description._v_colobjects
            descr2 = descr.copy()

            # dynamically add correct label type

            if MODE == 'energy_recon':
                descr2['energy_reconstruction_bin_label'] = UInt8Col()

            # dynamically add columns for telescopes
            if STORAGE_MODE == 'all':
                for tel_type in selected.keys():

                    if tel_type == 'SST':
                        IMAGE_WIDTH = SST_IMAGE_WIDTH * IMG_SCALE_FACTOR
                        IMAGE_LENGTH = SST_IMAGE_LENGTH * IMG_SCALE_FACTOR
                    elif tel_type == 'SCT':
                        IMAGE_WIDTH = SCT_IMAGE_WIDTH * IMG_SCALE_FACTOR
                        IMAGE_LENGTH = SCT_IMAGE_LENGTH * IMG_SCALE_FACTOR
                    elif tel_type == 'LST':
                        IMAGE_WIDTH = LST_IMAGE_WIDTH * IMG_SCALE_FACTOR
                        IMAGE_LENGTH = LST_IMAGE_LENGTH * IMG_SCALE_FACTOR

                    for tel_id in selected[tel_type]:
                        descr2["T" + str(tel_id)] = UInt16Col(
                            shape=(IMAGE_WIDTH, IMAGE_LENGTH, IMAGE_CHANNELS))

                    descr2["trig_list"] = UInt8Col(shape=(NUM_TEL))

            elif STORAGE_MODE == 'mapped':
                descr2["tel_map"] = Int16Col(shape=(NUM_TEL))

            table2 = f.create_table(
                d, 'temp', descr2, "Table of event records")
            table.attrs._f_copy(table2)
            table.remove()
            table2.move(d, 'Events')

        dataset_tables.append(d.Events)

    # for mapped storage, create 1 Array per telescope
    # telescope_arrays = []
    if STORAGE_MODE == 'mapped':
        for d in datasets:
            for tel_type in selected.keys():
                if tel_type == 'SST':
                    IMAGE_WIDTH = SST_IMAGE_WIDTH * IMG_SCALE_FACTOR
                    IMAGE_LENGTH = SST_IMAGE_LENGTH * IMG_SCALE_FACTOR
                elif tel_type == 'SCT':
                    IMAGE_WIDTH = SCT_IMAGE_WIDTH * IMG_SCALE_FACTOR
                    IMAGE_LENGTH = SCT_IMAGE_LENGTH * IMG_SCALE_FACTOR
                elif tel_type == 'LST':
                    IMAGE_WIDTH = LST_IMAGE_WIDTH * IMG_SCALE_FACTOR
                    IMAGE_LENGTH = LST_IMAGE_LENGTH * IMG_SCALE_FACTOR

                # img_atom = Atom.from_dtype(np.dtype((np.int16, (IMAGE_WIDTH, IMAGE_LENGTH,IMAGE_CHANNELS))))
                # img_atom = Atom.from_type('int16', shape=(IMAGE_WIDTH,IMAGE_LENGTH,IMAGE_CHANNELS))
                img_atom = Int16Atom()

                for tel_id in selected[tel_type]:
                    if not d.__contains__('T' + str(tel_id)):
                        # print("creating T{}".format(tel_id))
                        array = f.create_earray(
                            d, 'T' + str(tel_id), img_atom, (0, IMAGE_WIDTH, IMAGE_LENGTH, IMAGE_CHANNELS))
                        # telescope_arrays.append(array)

    # define/specify calibration and other processing
    cal = CameraCalibrator(None, None)

    logger.info("Processing events...")

    event_count = 0
    passing_count = 0

    # load all SCT data from simtel file
    source = hessio_event_source(
        data_file_path, allowed_tels=[
            j for i in selected.keys() for j in selected[i]])

    for event in source:
        event_count += 1

        # get energy bin and reconstructed energy
        if config['use_pkl_dict']:
            if (event.r0.run_id, event.r0.event_id) in bins_cuts_dict:
                bin_number, reconstructed_energy = bins_cuts_dict[(
                    event.r0.run_id, event.r0.event_id)]
                passing_count += 1
            else:
                continue
        else:
            # if pass cuts (applied locally):
            bin_number, reconstructed_energy = 0
            # else:
            # continue

        # calibrate camera (charge extraction + pedestal subtraction + trace integration)
        # NOTE: MUST BE MOVED UP ONCE ENERGY RECONSTRUCTION AND BIN NUMBERS ARE
        # CALCULATED LOCALLY
        cal.calibrate(event)

        # compute correct energy reconstruction bin label
        if MODE == 'energy_recon':
            for i in range(len(energy_recon_bins)):
                if event.mc.energy.value >= 10**(
                        energy_recon_bins[i][0]) and event.mc.energy.value < 10**(energy_recon_bins[i][1]):
                    erec_bin_label = i
                    passing_count += 1
                    break

        # process and write data
        if MODE == 'energy_recon':
            event_row = dataset_tables[0].row
        elif MODE == 'gh_class':
            event_row = dataset_tables[bin_number].row

        # collect telescope data and create trig_list and tel_map
        if STORAGE_MODE == 'all':
            trig_list = []
        if STORAGE_MODE == 'mapped':
            tel_map = []
        for tel_type in selected.keys():
            for tel_id in selected[tel_type]:
                if tel_id in event.r0.tels_with_data:
                    image = event.dl1.tel[tel_id].image
                    # truncate at 0
                    image[image < 0] = 0
                    # round float values to hundredths + save as int
                    image = [round(i * 100) for i in image[0]]
                    # compute peak position
                    if INCLUDE_TIMING:
                        peaks = event.dl1.tel[tel_id].peakpos[0]
                    else:
                        peaks = None

                    image_array = make_sct_image_array(
                        image, peaks, IMG_SCALE_FACTOR, IMG_DTYPE, IMG_DIM_ORDERING, IMAGE_CHANNELS)

                    if STORAGE_MODE == 'all':
                        trig_list.append(1)
                        event_row["T" + str(tel_id)] = image_array
                    elif STORAGE_MODE == 'mapped':
                        array_str = 'f.root.E{}.T{}'.format(bin_number, tel_id)
                        array = eval(array_str)
                        next_index = array.nrows
                        array.append(np.expand_dims(image_array, axis=0))
                        tel_map.append(next_index)

                else:
                    if STORAGE_MODE == 'all':
                        trig_list.append(0)
                        event_row["T" + str(tel_id)] = make_sct_image_array(
                            None, None, IMG_SCALE_FACTOR, IMG_DTYPE, IMG_DIM_ORDERING, IMAGE_CHANNELS)
                    elif STORAGE_MODE == 'mapped':
                        tel_map.append(-1)

        if STORAGE_MODE == 'all':
            event_row['trig_list'] = trig_list
        elif STORAGE_MODE == 'mapped':
            event_row['tel_map'] = tel_map

        if event.mc.shower_primary_id == 0:
            gamma_hadron_label = 1
        elif event.mc.shower_primary_id == 101:
            gamma_hadron_label = 0

        # other parameter data
        event_row['event_number'] = event.r0.event_id
        event_row['run_number'] = event.r0.run_id
        event_row['gamma_hadron_label'] = gamma_hadron_label
        event_row['MC_energy'] = event.mc.energy.value
        event_row['reconstructed_energy'] = reconstructed_energy

        if MODE == 'energy_recon':
            event_row['energy_reconstruction_bin_label'] = erec_bin_label

        # write data to table
        event_row.append()

    for table in dataset_tables:
        table.flush()

    logger.info("{} events processed".format(event_count))
    logger.info("{} events passed cuts/written to file".format(passing_count))
    logger.info("Done!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Load image data and event parameters from a simtel file into a formatted HDF5 file.')
    parser.add_argument('data_file', help='path to input .simtel file')
    parser.add_argument(
        'hdf5_path',
        help='path of output HDF5 file, or currently existing file to append to')
    parser.add_argument(
        'bins_cuts_dict',
        help='dictionary containing bins/cuts in .pkl format')
    parser.add_argument(
        'config_file',
        help='configuration file')
    parser.add_argument(
        "--debug", help="print debug/logger messages", action="store_true")
    args = parser.parse_args()

    # Configuration file, load + validate
    spc = config_spec.split('\n')
    config = ConfigObj(args.config_file, configspec=spc)
    validator = Validator()
    val_result = config.validate(validator)

    if val_result:
        logger.info("Config file validated.")
    else:
        logger.error("Invalid config file.")
        raise ValueError('Invalid config file.')

    # load bins/cuts file
    if config['use_pkl_dict']:
        logger.info("Loading bins/cuts dictionary...")
        bins_dict = pkl.load(open(args.bins_cuts_dict, "rb"))
    else:
        bins_dict = None

    image_extractor(args.data_file, args.hdf5_path, bins_dict, config)
