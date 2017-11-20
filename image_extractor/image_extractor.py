# -*- coding: utf-8 -*-
"""
Module for main ImageExtractor class and command line tool.
"""

import random
import argparse
import pickle as pkl
import logging
import glob
import os
import math

import numpy as np
import tables
from ctapipe.io.hessio import hessio_event_source
from ctapipe.calib import pedestals, CameraCalibrator

import image
import row_types

logger = logging.getLogger(__name__)

class ImageExtractor:
    """Class to handle reading in simtel files and writing to HDF5.

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

    TEL_TYPES = {'LST':51940, 'MSTF':29680, 'MSTN':28224, 'MSTS':63282, 'SST1':7258, 'SSTA':5091, 'SSTC':4676}

    IMAGE_SHAPES = image.IMAGE_SHAPES

    ALLOWED_CUT_PARAMS = {}
    DEFAULT_CUTS_DICT = {}

    def __init__(self,
                 output_path,
                 ED_cuts_dict=None,
                 storage_mode='tel_type',
                 tel_type_list=['MSTS'],
                 img_mode='2D',
                 img_channels=1,
                 include_timing=False,
                 img_scale_factors={'MSTS':1},
                 img_dtype='uint16',
                 img_dim_order='channels_last',
                 cuts_dict=DEFAULT_CUTS_DICT):

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
            if tel_type not in self.TEL_TYPES:
                raise ValueError('Invalid telescope type: {}.'.format(tel_type))
        self.tel_type_list = tel_type_list

        if img_mode in ['1D','2D']:
            self.img_mode = img_mode
        else:
            raise ValueError('Invalid img_mode: {}.'.format(img_mode))
       
        if img_channels > 0:
            self.img_channels = img_channels
        else:
            raise ValueError('Image channels must be > 0.')

        self.include_timing = include_timing

        self.img_scale_factors = img_scale_factors
        self.img_dtype = img_dtype

        if img_dim_order in ['channels_first','channels_last']:
            self.img_dim_order = img_dim_order
        else: 
            raise ValueError('Invalid dimension ordering: {}.'.format(img_dim_order))
        
        self.trace_converter= image.TraceConverter(
            self.img_dtype,
            self.img_dim_order,
            self.img_channels,
            self.img_scale_factors)

        self.cuts_dict = cuts_dict

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

        source_temp = hessio_event_source(data_file, max_events=1)

        all_tels = {tel_type:[] for tel_type in self.TEL_TYPES}

        dict_to_tel_type = {value: tel_type for tel_type, value in TEL_TYPES.items()}

        for event in source_temp:
            for tel_id in sorted(event.inst.telescope_ids):
                if round(event.inst.num_pixels[tel_id]*event.inst.optical_foclen[tel_id].value) in dict_to_tel_type:
                    tel_type = dict_to_tel_type[round(event.inst.num_pixels[tel_id]*event.inst.optical_foclen[tel_id].value)] 
                    all_tels[tel_type].append(tel_id)

        # select telescopes by type
        logger.info("Selected telescope types: [")
        for tel_type in self.tel_type_list:
            logger.info("{},".format(tel_type))
        logger.info("]")
       
        selected_tels = {tel_type: all_tels[tel_type] for tel_type in self.tel_type_list}

        total_num_tel_selected = 0
        for tel_type in all_tels:
            if tel_type in selected_tels:
                num_tel_selected = len(selected_tels[tel_type])
            else:
                num_tel_selected = 0
            logger.info(tel_type + ": " + str(num_tel_selected) +
                        " out of " + str(len(all_tels[tel_type])) +
                        " telescopes selected.")
            total_num_tel_selected += num_tel_selected

        return selected_tels, total_num_tel_selected

    def process_data(self, data_file, max_events):
        """Main method to read a simtel.gz data file, process the event data,
        and write it to a formatted HDF5 data file.
        """
        logger.info("Preparing HDF5 file structure...")

        f = tables.open_file(self.output_path, mode="a", title="Output File")

        selected_tels, num_tel = self.select_telescopes(data_file)

        # create and fill telescope information table
        if not f.__contains__('/Tel_Table'):
            tel_pos_table = f.create_table(f.root, 'Tel_Table',
                                           row_types.Tel,
                                           ("Table of telescope data"))
            tel_row = tel_pos_table.row

            source_temp = hessio_event_source(data_file, max_events=1)
            for event in source_temp:
                for tel_type in selected_tels:
                    for tel_id in selected_tels[tel_type]:
                        tel_row["tel_id"] = tel_id
                        tel_row["tel_x"] = event.inst.tel_pos[tel_id].value[0]
                        tel_row["tel_y"] = event.inst.tel_pos[tel_id].value[1]
                        tel_row["tel_z"] = event.inst.tel_pos[tel_id].value[2]
                        tel_row["tel_type"] = tel_type
                        tel_row["run_array_direction"] = event.mcheader.run_array_direction
                        tel_row["optical_foclen"] = event.inst.optical_foclen[tel_id].value
                        tel_row["num_pixels"] = event.inst.num_pixels[tel_id]
                        tel_row.append()


        #create event table
        if not f.__contains__('/Events'):
            table = f.create_table(f.root, 'Events',
                                   row_types.Event,
                                   "Table of Event metadata")
           
            descr = table.description._v_colobjects
            descr2 = descr.copy()

            if self.storage_mode == 'tel_type':
                for tel_type in selected_tels:
                    descr2[tel_type+'_indices'] = tables.Int32Col(shape=(len(selected_tels[tel_type])))
            elif self.storage_mode == 'tel_id':
                descr2["indices"] = tables.Int32Col(shape=(num_tel))

            table2 = f.create_table(f.root, 'temp', descr2,"Table of Events")
            table.attrs._f_copy(table2)
            table.remove()
            table2.move(f.root, 'Events')

        #create image arrays
        for tel_type in selected_tels:
            if self.img_mode == '2D':
                img_width = self.IMAGE_SHAPES[tel_type][0]*self.img_scale_factors[tel_type]
                img_length = self.IMAGE_SHAPES[tel_type][1]*self.img_scale_factors[tel_type]
    
                if self.img_dim_order == 'channels_first':
                    array_shape = (0,self.img_channels,img_width,img_length)
                elif self.img_dim_order == 'channels_last':
                    array_shape = (0,img_width,img_length,self.img_channels)
            elif self.img_mode == '1D':
                if self.img_dim_order == 'channels_first':
                    array_shape = (0,self.img_channels,self.TEL_NUM_PIXELS[tel_type])
                elif self.img_dim_order == 'channels_last':
                    array_shape = (0,self.TEL_NUM_PIXELS[tel_type],self.img_channels)
           
            if self.storage_mode == 'tel_type':
                if not f.__contains__('/' + tel_type):
                    array = f.create_earray(f.root,tel_type,
                                            tables.Atom.from_dtype(np.dtype(self.img_dtype)),
                                            array_shape)
           
            elif self.storage_mode == 'tel_id':
                for tel_id in selected_tels[tel_type]:
                    if not f.__contains__('T' + str(tel_id)):
                            array = f.create_earray(f.root,'T'+str(tel_id),
                                                    tables.Atom.from_dtype(np.dtype(self.img_dtype)),
                                                    array_shape)
                
        # specify calibration and other processing options
        cal = CameraCalibrator(None, None)

        logger.info("Processing events...")

        event_count = 0
        passing_count = 0

        source = hessio_event_source(data_file,allowed_tels=[j for i in selected_tels for j in selected_tels[i]],max_events=max_events)

        for event in source:
            event_count += 1

            if self.cuts_dict:
                if self.ED_cuts_dict is not None:
                    logger.warning('Warning: Both ED_cuts_dict and cuts dictionary found. Using cuts dictionary instead.')

                if test_cuts(event,cuts_dict):
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

            table = f.root.Events
            event_row = table.row

            if self.storage_mode == 'tel_type':
                tel_index_vectors = {tel_type:[] for tel_type in selected_tels}
            elif self.storage_mode == 'tel_id':
                all_tel_index_vector = []

            for tel_type in sorted(selected_tels.keys()):
                for tel_id in selected_tels[tel_type]:
                    if self.storage_mode == 'tel_type':
                        index_vector = tel_index_vectors[tel_type]
                    elif self.storage_mode == 'tel_id':
                        index_vector = all_tel_index_vector
                    
                    if tel_id in event.r0.tels_with_data:
                        pixel_vector = event.dl1.tel[tel_id].image  
                        if self.include_timing:
                            peaks_vector = event.dl1.tel[tel_id].peakpos[0]
                        else:
                            peaks_vector = None

                        if self.img_mode == '2D':
                            image = self.trace_converter.convert_SCT(pixel_vector, peaks_vector)
                        elif self.img_mode == '1D':
                            if self.img_dim_order == 'channels_first':
                                image = np.stack([pixel_vector,peaks_vector],axis=0)
                            elif self.img_dim_order == 'channels_last':
                                image = np.stack([pixel_vector,peaks_vector],axis=1)
                    
                        if self.storage_mode == 'tel_type':
                            array = eval('f.root.{}'.format(tel_type))
                        elif self.storage_mode == 'tel_id':
                            array = eval('f.root.T{}'.format(tel_id))
                        next_index = array.nrows
                        array.append(np.expand_dims(image, axis=0))
                        
                        index_vector.append(next_index)

                    else:
                        index_vector.append(-1)

            if self.storage_mode == 'tel_type':
                for tel_type in tel_index_vectors:
                    event_row[tel_type+'_indices'] = tel_index_vectors[tel_type]
            elif self.storage_mode == 'tel_id':
                event_row['indices'] = all_tel_index_vector
            
            event_row['event_number'] = event.r0.event_id
            event_row['run_number'] = event.r0.run_id
            event_row['gamma_hadron_label'] = event.mc.shower_primary_id
            event_row['core_x'] = event.mc.core_x.value
            event_row['core_y'] = event.mc.core_y.value
            event_row['h_first_int'] = event.mc.h_first_int.value
            event_row['mc_energy'] = event.mc.energy.value
            event_row['alt'] = event.mc.alt.value
            event_row['az'] = event.mc.az.value

            event_row.append()
            table.flush()

        f.close()

        logger.info("{} events read in file".format(event_count))
        if self.cuts_dict or self.ED_cuts_dict:
            logger.info("{} events passed cuts/written to file".format(passing_count))
        logger.info("Done!")

    def set_cuts(self,cuts_dict):
        for i in cuts_dict:
            if i not in ALLOWED_CUT_PARAMS:
                raise ValueError("Invalid cut parameter: {}.".format(i))

        self.cuts_dict = cuts_dict

    def test_cuts(event,cuts):
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
            "Table of events")

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
        assert math.isclose(split_sum,1.0,rel_tol=1e-5), "Split fractions do not add up to 1"

        # open input hdf5 file
        f = tables.open_file(h5_file, mode="r+", title="Input file")

        table = f.root.Events
        descr = table.description

        num_events = table.shape[0]
        indices = range(num_events)
        i = 0

        split_names = ['Training','Validation','Test']

        for j in range(len(splits)):
            if splits[j] != 0.0:
                table_new = f.create_table(
                    f.root,
                    'Events_' +
                    split_names[j],
                    descr,
                    "Table of " +
                    split_names[j] +
                    " Events")

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
        'data_files',
        help='wildcard path to input .simtel files')
    parser.add_argument(
        'hdf5_path',
        help=('path of output HDF5 file, or currently existing file to append to'))
    parser.add_argument(
        '--config_file',
        help=('configuration file specifying the selected telescope ids '
              'from simtel file, the desired energy bins, and the correct '
              'image output dimensions/dtype.'))
    parser.add_argument(
        '--ED_cuts_dict_file',
        help='path of .pkl file containing cuts dictionary from EventDisplay')
    parser.add_argument(
        "--debug",
        help="print debug/logger messages",
        action="store_true")
    parser.add_argument(
        "--max_events",
        help="set a maximum number of events to process from each file",
        type=int)
    parser.add_argument(
        "--shuffle",
        help="shuffle output data file. Can pass optional random seed",nargs='?',const='default',action='store',default=None)
    parser.add_argument(
        "--split",
        help="Split output data file into separate event tables. Pass optional list of splits (training, validation, test)",
        nargs='?',const = [0.9,0.1,0.0],action='store',default=None)

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # load bins cuts dictionary from file
    if args.ED_cuts_dict_file is not None:
        ED_cuts_dict = pkl.load(open(args.ED_cuts_dict_file, "rb"))
    else:
        ED_cuts_dict = None

    extractor = ImageExtractor(args.hdf5_path,ED_cuts_dict=ED_cuts_dict)

    data_files = glob.glob(args.data_files)

    for data_file in data_files:
        extractor.process_data(data_file, args.max_events)

    if args.shuffle:
        extractor.shuffle_data(args.hdf5_path, args.shuffle)

    if args.split:
        extractor.split_data(args.hdf5_path, args.split)
