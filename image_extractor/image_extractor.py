"""
ImageExtractor class for writing processed data from ctapipe containers to HDF5 files.
"""

import argparse
import math
import pickle as pkl
import logging
import glob
import shutil

from configobj import ConfigObj
from validate import Validator
import numpy as np
from astropy import units as u
from tables import *
from ctapipe.io.hessio import hessio_event_source
from ctapipe.calib import pedestals,CameraCalibrator

import config
import row_types
import trace_convertor 

__all__ = ['Image_Extractor']

logger = logging.getLogger(__name__)


SCT_IMAGE_WIDTH = 120
SCT_IMAGE_LENGTH = 120

class ImageExtractor

    def __init__(self, output_path,config_file,bins_cuts_dict):
        self.output_path = output_path
        
        if val_result:
            logger.info("Config file validated.")

            self.mode = config['mode']
            self.img_mode = config['image']['mode']
            self.tel_type_mode = config['telescope']['type_mode']
            self.storage_mode = config['storage_mode']
            self.img_scale_factor = config['image']['scale_factor']
            self.img_dtype = config['image']['dtype']
            self.img_dim_order = config['image']['dim_ordering']
            self.energy_bin_units = config['energy_bins']['units']
            self.use_bins_cuts_dict = config['use_pkl_dict']

        else:
            logger.error("Invalid config file.")
            raise ValueError('Invalid config file.')

        #load bins_cuts_dict
        if self.use_bins_cuts_dict and bins_cuts_dict is None:
            logger.error("Cuts enabled in config file but dictionary missing.")
            raise ValueError("Cuts enabled in config file but dictionary missing.")
            
        self.bins_cuts_dict = bins_cuts_dict

    def select_telescopes(self,data_file):

        #harcoded telescope constants
        #TODO: find more natural way of handling this?
        LST_NUM_PIXELS = 1855
        SCT_NUM_PIXELS = 11328
        SST_NUM_PIXELS = 0 #update

        logger.info("Getting telescope types...")

        #collect telescope lists 
        source_temp = hessio_event_source(data_file,max_events=1)

        all_tels = {'SST' : [], 'SCT' : SCT_list, 'LST' : LST_list}

        for event in source_temp: 
            for i in event.inst.telescope_ids:
                if event.inst.num_pixels[i] == SCT_NUM_PIXELS:
                    all_tels['SCT'].append(i)
                elif event.inst.num_pixels[i] == LST_NUM_PIXELS:
                    all_tels['LST'].append(i)
                elif event.inst.num_pixels[i] == SST_NUM_PIXELS:
                    all_tels['SST'].append(i)
                else:
                    logger.error("Unknown telescope type (invalid num_pixels).")
                    raise ValueError("Unknown telescope type (invalid num_pixels).")

        #select telescopes by type
        logger.info("Telescope Mode: ",self.tel_type_mode)

        if self.tel_type_mode == 'SST':
            selected_tel_types = ['SST']
        elif self.tel_type_mode == 'SCT':
            selected_tel_types = ['SCT']
        elif self.tel_type_mode == 'LST':
            selected_tel_types = ['LST']
        elif self.tel_type_mode == 'SCT+LST':
            selected_tel_types = ['SCT','LST']
        elif self.tel_type_mode == 'SST+SCT':
            selected_tel_types = ['SST','SCT']
        elif self.tel_type_mode == 'SST+LST':
            selected_tel_types = ['SST','LST']
        elif self.tel_type_mode == 'ALL':
            selected_tel_types = ['SST','SCT','LST']
        else:
            logger.error("Telescope selection mode invalid.")
            raise ValueError('Telescope selection mode not recognized.') 

        selected_tels = {key:all_tels[key] for key in selected_tel_types}

        num_tel = 0

        for tel_type in selected_tels.keys():
            logger.info(tel_type + ": " + str(len(selected_tels[tel_type])) + " out of " + str(len(all_tels[tel_type])) + " telescopes selected.")
            num_tel += len(selected_tels[tel_type])

        return selected_tels,num_tel

    def process_data(self,data_file,max_events):
        """
        Function to read and write data from ctapipe containers to HDF5 
        """
       
        logger.info("Mode: ",MODE)
        logger.info("Image mode: ",IMG_MODE)
        logger.info("File storage mode: ",STORAGE_MODE)
        logger.info("Image scale factor: ",IMG_SCALE_FACTOR)
        logger.info("Image array type: ",IMG_DTYPE)
        logger.info("Image dim order: ",IMG_DIM_ORDERING)

        logger.info("Loading additional configuration settings...")

        if MODE == 'gh_class':

            e_min = float(config['energy_bins']['min'])
            e_max = float(config['energy_bins']['max'])
            e_bin_size = float(config['energy_bins']['bin_size'])
            num_e_bins = int((e_max - e_min)/e_bin_size)
            energy_bins = [(e_min + i*e_bin_size, e_min + (i+1)*e_bin_size) for i in range(num_e_bins)]

        elif MODE == 'energy_recon':

            erec_min = float(config['energy_recon']['bins']['min'])
            erec_max = float(config['energy_recon']['bins']['max'])
            erec_bin_size = float(config['energy_recon']['bins']['bin_size'])
            num_erec_bins = int((erec_max - erec_min)/erec_bin_size)
            energy_recon_bins = [(erec_min+i*erec_bin_size,erec_min + (i+1)*erec_bin_size) for i in range(num_erec_bins)]   

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

        f = open_file(output_file_path, mode = "a", title = "Data File") 

        #prep data into bins based on mode (by energy bin for g/h classification, 1 group for energy reconstruction)
        if MODE == 'gh_class':
            #create groups for each energy bin, if not already existing
            datasets = []
            for i in range(len(energy_bins)):
                if not f.__contains__("/E" + str(i)):
                    group = f.create_group("/", "E" + str(i), 'Energy bin group' + str(i))
                    group._v_attrs.min_energy = energy_bins[i][0]
                    group._v_attrs.max_energy = energy_bins[i][1]
                    group._v_attrs.units = ENERGY_BIN_UNITS
                row_str = 'f.root.E{}'.format(str(i))
                where = eval(row_str)
                datasets.append(where)
        elif MODE == 'energy_recon':
            datasets = [f.root]

        dataset_tables = []

        #create table for telescope positions
        if not f.__contains__('/Tel_Table'):
            tel_pos_table = f.create_table("/",'Tel_Table', Tel, "Table of telescope ids, positions, and types")
            tel_row = tel_pos_table.row

            source_temp = hessio_event_source(data_file_path,max_events=1)

            for event in source_temp: 

                for i in selected.keys():
                    for j in selected[i]:
                        tel_row["tel_id"] = j
                        tel_row["tel_x"] = event.inst.tel_pos[j].value[0]
                        tel_row["tel_y"] = event.inst.tel_pos[j].value[1]
                        tel_row["tel_z"] = event.inst.tel_pos[j].value[2]
                        tel_row["tel_type"] = i
                        tel_row["run_array_direction"] = \
                            event.mcheader.run_array_direction

                        tel_row.append()

        #create table for events
        for d in datasets:
            if not d.__contains__('Events'):
                table = f.create_table(d, 'Events', Event, "Table of event records") 
                descr = table.description._v_colobjects
                descr2 = descr.copy()

                #dynamically add correct label type

                if MODE == 'energy_recon':
                    descr2['reconstructed_energy'] = UInt8Col()
                    descr2['energy_reconstruction_bin_label'] = UInt8Col()

                #dynamically add columns for telescopes
                if STORAGE_MODE == 'all':
                    for tel_type in selected.keys():

                        if tel_type == 'SST':
                            IMAGE_WIDTH = SST_IMAGE_WIDTH*IMG_SCALE_FACTOR
                            IMAGE_LENGTH = SST_IMAGE_LENGTH*IMG_SCALE_FACTOR
                        elif tel_type == 'SCT':
                            IMAGE_WIDTH = SCT_IMAGE_WIDTH*IMG_SCALE_FACTOR
                            IMAGE_LENGTH = SCT_IMAGE_LENGTH*IMG_SCALE_FACTOR
                        elif tel_type == 'LST':
                            IMAGE_WIDTH = LST_IMAGE_WIDTH*IMG_SCALE_FACTOR
                            IMAGE_LENGTH = LST_IMAGE_LENGTH*IMG_SCALE_FACTOR

                        for tel_id in selected[tel_type]:
                            descr2["T" + str(tel_id)] = UInt16Col(shape=(IMAGE_WIDTH,IMAGE_LENGTH,IMAGE_CHANNELS))

                        descr2["trig_list"] = UInt8Col(shape=(NUM_TEL))
     
                elif STORAGE_MODE == 'mapped':
                    descr2["tel_map"] = Int32Col(shape=(NUM_TEL))
        
                table2 = f.create_table(d, 'temp', descr2, "Table of event records")
                table.attrs._f_copy(table2)
                table.remove()
                table2.move(d, 'Events')
            
            dataset_tables.append(d.Events)

        #for mapped storage, create 1 Array per telescope
        #telescope_arrays = []
        if STORAGE_MODE == 'mapped':
            for d in datasets:
                for tel_type in selected.keys():
                    if tel_type == 'SST':
                        IMAGE_WIDTH = SST_IMAGE_WIDTH*IMG_SCALE_FACTOR
                        IMAGE_LENGTH = SST_IMAGE_LENGTH*IMG_SCALE_FACTOR
                    elif tel_type == 'SCT':
                        IMAGE_WIDTH = SCT_IMAGE_WIDTH*IMG_SCALE_FACTOR
                        IMAGE_LENGTH = SCT_IMAGE_LENGTH*IMG_SCALE_FACTOR
                    elif tel_type == 'LST':
                        IMAGE_WIDTH = LST_IMAGE_WIDTH*IMG_SCALE_FACTOR
                        IMAGE_LENGTH = LST_IMAGE_LENGTH*IMG_SCALE_FACTOR
        
                    #img_atom = Atom.from_dtype(np.dtype((np.int16, (IMAGE_WIDTH, IMAGE_LENGTH,IMAGE_CHANNELS))))
                    #img_atom = Atom.from_type('int16', shape=(IMAGE_WIDTH,IMAGE_LENGTH,IMAGE_CHANNELS)) 
                    img_atom = Int16Atom()

                    for tel_id in selected[tel_type]:
                        if not d.__contains__('T'+str(tel_id)):
                            #print("creating T{}".format(tel_id))
                            array = f.create_earray(d, 'T'+str(tel_id), img_atom, (0,IMAGE_WIDTH,IMAGE_LENGTH,IMAGE_CHANNELS))
                            #telescope_arrays.append(array)

        #define/specify calibration and other processing
        cal = CameraCalibrator(None,None)

        logger.info("Processing events...")

        event_count = 0
        passing_count = 0

        #load all SCT data from simtel file
        source = hessio_event_source(data_file_path,allowed_tels=[j for i in selected.keys() for j in selected[i]], max_events=max_events)

        for event in source:
            event_count += 1

            #get energy bin and reconstructed energy
            if config['use_pkl_dict']:
                if (event.r0.run_id,event.r0.event_id) in bins_cuts_dict:
                    bin_number, reconstructed_energy = bins_cuts_dict[(event.r0.run_id,event.r0.event_id)]
                    passing_count +=1
                else:
                    continue
            else:
                #if pass cuts (applied locally):
                bin_number, reconstructed_energy = [0, 0]
                #else:
                #continue

            #calibrate camera (charge extraction + pedestal subtraction + trace integration)
            #NOTE: MUST BE MOVED UP ONCE ENERGY RECONSTRUCTION AND BIN NUMBERS ARE CALCULATED LOCALLY
            cal.calibrate(event)

            #compute correct energy reconstruction bin label
            if MODE == 'energy_recon':
                for i in range(len(energy_recon_bins)):
                    if event.mc.energy.value >= 10**(energy_recon_bins[i][0]) and event.mc.energy.value < 10**(energy_recon_bins[i][1]):
                        erec_bin_label = i
                        passing_count += 1
                        break

            #process and write data
            if MODE == 'energy_recon':
                event_row = dataset_tables[0].row
            elif MODE == 'gh_class':
                event_row = dataset_tables[bin_number].row

            #collect telescope data and create trig_list and tel_map
            if STORAGE_MODE == 'all':
                trig_list = []
            if STORAGE_MODE == 'mapped':
                tel_map = []
            for tel_type in selected.keys():
                for tel_id in selected[tel_type]:
                    if tel_id in event.r0.tels_with_data:
                        image = event.dl1.tel[tel_id].image
                        #truncate at 0
                        image[image < 0] = 0
                        #round float values to hundredths + save as int
                        image = [round(i*100) for i in image[0]]
                        #compute peak position
                        if INCLUDE_TIMING:
                            peaks = event.dl1.tel[tel_id].peakpos[0]
                        else:
                            peaks = None 

                        image_array = make_sct_image_array(image,peaks,IMG_SCALE_FACTOR,IMG_DTYPE,IMG_DIM_ORDERING,IMAGE_CHANNELS)
     
                        if STORAGE_MODE == 'all':
                            trig_list.append(1)
                            event_row["T" + str(tel_id)] = image_array          
                        elif STORAGE_MODE == 'mapped':
                            array_str = 'f.root.E{}.T{}'.format(bin_number,tel_id)
                            array = eval(array_str)
                            next_index = array.nrows
                            array.append(np.expand_dims(image_array,axis=0))
                            tel_map.append(next_index)

                    else:
                        if STORAGE_MODE == 'all':
                            trig_list.append(0)
                            event_row["T" + str(tel_id)] = make_sct_image_array(None,None,IMG_SCALE_FACTOR,IMG_DTYPE,IMG_DIM_ORDERING,IMAGE_CHANNELS)
                        elif STORAGE_MODE == 'mapped':
                            tel_map.append(-1)
                            
            if STORAGE_MODE == 'all':
                event_row['trig_list'] = trig_list 
            elif STORAGE_MODE == 'mapped':
                event_row['tel_map'] = tel_map

            #other parameter data
            event_row['event_number'] = event.r0.event_id
            event_row['run_number'] = event.r0.run_id
            event_row['gamma_hadron_label'] = event.mc.shower_primary_id
            event_row['core_x'] = event.mc.core_x.value
            event_row['core_y'] = event.mc.core_y.value
            event_row['h_first_int'] = event.mc.h_first_int.value
            event_row['mc_energy'] = event.mc.energy.value
            event_row['alt'] = event.mc.alt.value
            event_row['az'] = event.mc.az.value


            if MODE == 'energy_recon':
                event_row['reconstructed_energy'] = reconstructed_energy
                event_row['energy_reconstruction_bin_label'] = erec_bin_label

            #write data to table
            event_row.append()

        for table in dataset_tables:
            table.flush()

        logger.info("{} events processed".format(event_count))
        logger.info("{} events passed cuts/written to file".format(passing_count))
        logger.info("Done!")

    def shuffle_data(self,h5_file):

        temp_filename = os.path.splitext(h5_file)[0] + "_temp.h5"

        #open input hdf5 file
        f_in = open_file(h5_file, mode = "r", title = "Input file")

        #create new file, duplicating file structure
        f_shuffled =  open_file(temp_filename, mode ="w", title = "Output file") 

        #copy tel_table
        if f_in.__contains__('/Tel_Table'): 
            tel_table = f_in.root.Tel_Table
            new_tel_table = tel_table.copy(f_shuffled.root, 'Tel_Table')

        if MODE == 'gh_class':
            for group in f_in.walk_groups("/"):
                if not group == f_in.root:
                    #copy energy bin groups
                    group_new = f_shuffled.create_group("/", group._v_name, group._v_attrs.TITLE)

                    #copy event table, but shuffle
                    table = group.Events
                    descr = table.description

                    num_events = table.shape[0]
                    new_indices = [i for i in range(num_events)]
                    random.shuffle(new_indices)

                    table_new = f_shuffled.create_table(group_new,'Events', descr,"Table of events")

                    for i in range(num_events):
                        table_new.append([tuple(table[new_indices[i]])]) 

                    #copy tel arrays
                    for child in group._v_children:
                        if re.match("T[0-9]+",child):
                            new_array = group._f_get_child(child).copy(group_new,child)


       shutil.move(temp_filename, h5_file) 

    def split_data(self,h5_file,split_dict):

        split_sum = 0
        for i in split_dict:
            split_sum += split_dict[i]
        assert split_sum == 1,"Split fractions do not add up to 1"

        temp_filename = os.path.splitext(h5_file)[0] + "_temp.h5"

        #open input hdf5 file
        f_in = open_file(h5_file, mode = "r", title = "Input file")

        #create new file, duplicating file structure
        f_shuffled =  open_file(temp_filename, mode ="w", title = "Output file") 

        #copy tel_table
        if f_in.__contains__('/Tel_Table'): 
            tel_table = f_in.root.Tel_Table
            new_tel_table = tel_table.copy(f_shuffled.root, 'Tel_Table')

        tables = []
        new_tables = []
   
        if MODE == 'gh_class':
            for group in f_in.walk_groups("/"):
                if not group == f_in.root:
                    #copy energy bin groups
                    group_new = f_shuffled.create_group("/", group._v_name, group._v_attrs.TITLE)

                    #copy tel arrays
                    for child in group._v_children:
                        if re.match("T[0-9]+",child):
                            new_array = group._f_get_child(child).copy(group_new,child)

                    #move events into separate tables
                    table = group.Events
                    descr = table.description

                    num_events = table.shape[0]
                    indices = range(num_events)
                    i = 0

                    for split in list(split_dict.keys()):
                        table_new = f_shuffled.create_table(group_new, 'Events_' + split, descr, "Table of " + split + " Events")

                        split_fraction = split_dict[split]
                        
                        if i+int(split_fraction*num_events) <= num_events:
                            split_indices = indices[i:i+int(split_fraction*num_events)]
                        else:
                            split_indices = indices[i:num_events]

                        i += int(split_fraction*num_events)

                        for j in split_indices:
                            table_new.append([tuple(table[split_indices[j]])])

       shutil.move(temp_filename, h5_file)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Load image data and event parameters from a simtel file into a formatted HDF5 file.')
    parser.add_argument('data_files', help='wildcard path to input .simtel files')
    parser.add_argument('hdf5_path', help='path of output HDF5 file, or currently existing file to append to')
    parser.add_argument('config_file',help='configuration file specifying the selected telescope ids from simtel file, the desired energy bins, the correst image output dimensions/dtype, ')
    parser.add_argument('--bins_cuts_dict_file',help='path of .pkl file containing bins/cuts dictionary')
    parser.add_argument("--debug", help="print debug/logger messages", action="store_true")
    parser.add_argument("--max_events", help="set a maximum number of events to process from each file",type=int)
    parser.add_argument("--shuffle",help="shuffle output data file", action="store_true")
    parser.add_argument("--split",help="split output data file into separate event tables", action="store_true")

    args = parser.parse_args()

    #Configuration file, load + validate
    spc = config.config_spec.split('\n')
    config = ConfigObj(args.config_file,configspec=spc)
    validator = Validator()
    val_result = config.validate(validator)

    #load bins cuts dictionary from file
    if args.bins_cuts_dict_file is not None:
        bins_cuts_dict = pkl.load(open(bins_cuts_dict_file, "rb" ))
    else:
        bins_cuts_dict = None

    extractor = ImageExtractor(args.hdf5_path,config,bins_cuts_dict)

    data_files = glob.glob(args.data_files)

    for data_file in data_files:
        extractor.process_data(data_file,args.max_events)

    if args.shuffle:
        extractor.shuffle_data(args.hdf5_path)

    if args.split:
        extractor.split_data(args.hdf5_path,split_dict)


