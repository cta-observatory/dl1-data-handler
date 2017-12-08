# -*- coding: utf-8 -*-
"""
Module for main ImageExtractor class and command line tool.
"""
import pkg_resources
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
    TEL_NUM_PIXELS = image.TEL_NUM_PIXELS
    IMAGE_SHAPES = image.IMAGE_SHAPES

    METADATA_FIELDS = {
        "ImageExtractor_ver": ("pkg_resources.get_distribution('image-extractor').version",),
        "ctapipe_ver": ("pkg_resources.get_distribution('ctapipe').version",),
        "CORSIKA_ver": None,
        "simtel_ver": None,
        "prod_site_alt": None,
        "prod_site_coord": None,
        "prod_site_B_field": None,
        "prod_site_array": None,
        "prod_site_subarray": None,
        "particle_type": ("event.mc.shower_primary_id",),
        "zenith": ("event.mcheader.run_array_direction[1]",),
        "azimuth": ("event.mcheader.run_array_direction[0]",),
        "spectral_index": None,
        "E_min": None,
        "E_max": None}

    ALLOWED_CUT_PARAMS = {}
    DEFAULT_CUTS_DICT = {}

    def __init__(self,
                 output_path,
                 ED_cuts_dict=None,
                 storage_mode='tel_type',
                 tel_type_list=['MSTS','LST','MSTF','MSTN','MSTS','SST1','SSTA','SSTC'],
                 img_mode='1D',
                 img_channels=1,
                 include_timing=True,
                 img_scale_factors={'MSTS':1},
                 img_dtypes={'MSTS':'float32','LST':'float32'},
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
        self.img_dtypes = img_dtypes

        if img_dim_order in ['channels_first','channels_last']:
            self.img_dim_order = img_dim_order
        else: 
            raise ValueError('Invalid dimension ordering: {}.'.format(img_dim_order))
        
        self.trace_converter= image.TraceConverter(
            self.img_dtypes,
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

        event = next(hessio_event_source(data_file,max_events=1))

        all_tels = {tel_type:[] for tel_type in self.TEL_TYPES}

        dict_to_tel_type = {value: tel_type for tel_type, value in self.TEL_TYPES.items()}

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

    def write_metadata(self,HDF5_file,data_file):

        logger.info("Checking/writing metadata...")

        event = next(hessio_event_source(data_file))

        attributes = HDF5_file.root._v_attrs

        for field in self.METADATA_FIELDS:
            if isinstance(self.METADATA_FIELDS[field],tuple):
                value = eval(self.METADATA_FIELDS[field][0])
            else:
                value = self.METADATA_FIELDS[field]
            if not attributes.__contains__(field):
                exec("attributes." + field + " = value")
            else:
                if eval("attributes." + field) != value:
                    raise ValueError("Metadata field {} for current simtel file does not match output file: {} vs {}".format(field,value,eval("attributes."+field)))

        run_file_path = os.path.abspath(data_file)
        if not attributes.__contains__("runlist"):
            attributes.runlist = [run_file_path]
        else:
            runlist = attributes.runlist
            runlist.append(run_file_path)
            attributes.runlist = runlist

    def process_data(self, data_file, max_events):
        """Main method to read a simtel.gz data file, process the event data,
        and write it to a formatted HDF5 data file.
        """
        logger.info("Preparing HDF5 file structure...")

        f = tables.open_file(self.output_path, mode="a", title="Output File")

        self.write_metadata(f,data_file)

        selected_tels, num_tel = self.select_telescopes(data_file)

        event = next(hessio_event_source(data_file))

        # create and fill telescope information table
        if not f.__contains__('/Telescope_Info'):
            tel_table = f.create_table(f.root, 'Telescope_Info',
                                           row_types.Tel,
                                           ("Table of telescope data"))
            tel_row = tel_table.row

            #add units to table attributes
            random_tel_type = random.choice(list(selected_tels.keys()))
            random_tel_id = random.choice(selected_tels[random_tel_type])
            tel_table.attrs.tel_pos_units = str(event.inst.tel_pos[random_tel_id].unit)
            tel_table.attrs.optical_foclen_units = str(event.inst.optical_foclen[random_tel_id].unit)

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
        if not f.__contains__('/Event_Info'):
            table = f.create_table(f.root, 'Event_Info',
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
            table2.move(f.root, 'Event_Info')

            #add units to table attributes
            table2.attrs.core_pos_units = str(event.mc.core_x.unit)
            table2.attrs.h_first_int_units = str(event.mc.h_first_int.unit)
            table2.attrs.mc_energy_units = str(event.mc.energy.unit)
            table2.attrs.alt_az_units = str(event.mc.alt.unit)

        #create image tables
        for tel_type in selected_tels:
            if self.img_mode == '2D':
                img_width = self.IMAGE_SHAPES[tel_type][0]*self.img_scale_factors[tel_type]
                img_length = self.IMAGE_SHAPES[tel_type][1]*self.img_scale_factors[tel_type]
    
                if self.img_dim_order == 'channels_first':
                    array_shape = (self.img_channels,img_width,img_length)
                elif self.img_dim_order == 'channels_last':
                    array_shape = (img_width,img_length,self.img_channels)

                np_type = np.dtype(np.dtype(self.img_dtypes[tel_type]), array_shape)
                columns_dict = {"image":tables.Col.from_dtype(np_type),"event_index":tables.Int32Col()}

            elif self.img_mode == '1D':
                array_shape = (self.TEL_NUM_PIXELS[tel_type],)  
                np_type = np.dtype((np.dtype(self.img_dtypes[tel_type]), array_shape))

                columns_dict = {"image_charge":tables.Col.from_dtype(np_type),"event_index":tables.Int32Col()}
                if self.include_timing:
                    columns_dict["image_peak_times"] = tables.Col.from_dtype(np_type)

            description = type('description', (tables.IsDescription,), columns_dict)

            if self.storage_mode == 'tel_type':
                if not f.__contains__('/' + tel_type):
                    table = f.create_table(f.root,tel_type,description,"Table of {} images".format(tel_type))

                    #append blank image at index 0
                    image_row = table.row
                    
                    if self.img_mode == '2D':
                        image_row['image'] = self.trace_converter.convert(None,None,tel_type)  
                    
                    elif self.img_mode == '1D':
                        shape = (image.TEL_NUM_PIXELS[tel_type],) 
                        image_row['image_charge'] = np.zeros(shape,dtype=self.img_dtypes[tel_type])
                        image_row['event_index'] = -1
                        if self.include_timing:
                            image_row['image_peak_times'] = np.zeros(shape,dtype=self.img_dtypes[tel_type])

                    image_row.append()
                    table.flush()
           
            elif self.storage_mode == 'tel_id':
                for tel_id in selected_tels[tel_type]:
                    if not f.__contains__('T' + str(tel_id)):
                        table = f.create_table(f.root,'T'+str(tel_id),description,"Table of T{} images".format(str(tel_id)))
                
                        #append blank image at index 0
                        image_row = table.row
                        
                        if self.img_mode == '2D':
                            image_row['image'] = self.trace_converter.convert(None,None,tel_type)  
                        
                        elif self.img_mode == '1D':
                            shape = (image.TEL_NUM_PIXELS[tel_type],) 
                            image_row['image_charge'] = np.zeros(shape,dtype=self.img_dtypes[tel_type])
                            image_row['event_index'] = -1
                            if self.include_timing:
                                image_row['image_peak_times'] = np.zeros(shape,dtype=self.img_dtypes[tel_type])

                        image_row.append()
                        table.flush()

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

            table = f.root.Event_Info
            table.flush()
            event_row = table.row
            event_index = table.nrows

            if self.storage_mode == 'tel_type':
                tel_index_vectors = {tel_type:[] for tel_type in selected_tels}
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
                        if self.include_timing:
                            peaks_vector = event.dl1.tel[tel_id].peakpos[0]
                        else:
                            peaks_vector = None

                        if self.storage_mode == 'tel_type':
                            table = eval('f.root.{}'.format(tel_type))
                        elif self.storage_mode == 'tel_id':
                            table = eval('f.root.T{}'.format(tel_id))
                        next_index = table.nrows
                        image_row = table.row

                        if self.img_mode == '2D':
                            image_row['image'] = self.trace_converter.convert(pixel_vector,peaks_vector,tel_type)  

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
                    event_row[tel_type+'_indices'] = tel_index_vectors[tel_type]
            elif self.storage_mode == 'tel_id':
                event_row['indices'] = all_tel_index_vector
            
            event_row['event_number'] = event.r0.event_id
            event_row['run_number'] = event.r0.run_id
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

        f.close()

        logger.info("{} events read in file".format(event_count))
        logger.info("{} total events in output file.".format(total_num_events))
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
        'run_list',
        help='text file run list containing a list of simtel files to process (1 per line)')
    parser.add_argument(
        'hdf5_path',
        help=('path of output HDF5 file, or currently existing file to append to'))
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

    extractor = ImageExtractor(args.hdf5_path,tel_type_list=['MSTS','LST'])

    run_list = []
    with open(args.run_list) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                run_list.append(line)

    logger.info("Number of data files in runlist: {}".format(len(run_list)))

    for i,data_file in enumerate(run_list):
        logger.info("Processing file {}/{}".format(i,len(run_list)))
        extractor.process_data(data_file, args.max_events)

    if args.shuffle:
        extractor.shuffle_data(args.hdf5_path, args.shuffle)

    if args.split:
        extractor.split_data(args.hdf5_path, args.split)
