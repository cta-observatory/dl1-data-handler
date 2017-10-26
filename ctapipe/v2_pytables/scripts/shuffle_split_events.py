import argparse
import math
from random import shuffle
import re

from configobj import ConfigObj
import pickle as pkl
from glob import glob
import numpy as np
from tables import *
from ctapipe.utils.datasets import get_dataset
from ctapipe.io.hessio import hessio_event_source
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.image.charge_extractors import GlobalPeakIntegrator
#from ctapipe.image
from matplotlib import pyplot as plt
from astropy import units as u
#from PIL import Image

TRAINING_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

MODE = 'gh_class'

if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Takes existing HDF5 file formatted for Tensorflow/pytables and shuffles the events in each bin independently. Then splits dataset into training, validation, test sets and writes to new file.')
    parser.add_argument('hdf5_path', help='path to HDF5 file to be shuffled')
    parser.add_argument('output_file',help='path to output HDF5 file (shuffled). Must not already exist.')
    parser.add_argument('-shuffle', action = 'store_true')
    args = parser.parse_args()

    shuffle = args.shuffle

    #open input hdf5 file
    f_in = open_file(args.hdf5_path, mode = "r", title = "Input file") 

    #create new file, duplicating file structure
    f_shuffled =  open_file(args.output_file, mode ="w", title = "Output file") 

    tables = []
    new_tables = []

    data_splits = ['Training','Validation','Test']

    #copy tel_table
    if f_in.__contains__('/Tel_Table'): 
        tel_table = f_in.root.Tel_Table
        new_tel_table = tel_table.copy(f_shuffled.root, 'Tel_Table')
        
    if MODE == 'gh_class':
        for group in f_in.walk_groups("/"):

            if not group == f_in.root:
                table = group.Events
                tables.append(table)
                descr = table.description
                group_new = f_shuffled.create_group("/", group._v_name, group._v_attrs.TITLE)

                #copy tel arrays
                for child in group._v_children:
                    if re.match("T[0-9]+",child):
                        new_array = group._f_get_child(child).copy(group_new,child)

                new_datasets = []
                for i in data_splits:
                    table_new = f_shuffled.create_table(group_new, 'Events_' + i, descr, "Table of " + i + " Inputs")
                    new_datasets.append(table_new)
                new_tables.append(new_datasets)
    elif MODE == 'energy_recon':
        table = f_shuffled.root.Events
        tables.append(table)
        descr = table.description
        new_datasets = []
        for i in data_splits:
            table_new = f_shuffled.create_table(f_shuffled.root, 'Events_' + i, descr, "Table of " + i + " Inputs")
            new_datasets.append(table_new)
        new_tables.append(new_datasets)
    
    for table,new_datasets in zip(tables,new_tables):
        #calculate total number of events from eventNumber dataset
        num_events = table.shape[0]
 
        #create list of indices per group and shuffle them
        new_indices = [i for i in range(num_events)]
        if shuffle:
            shuffle(new_indices)

        train_start = 0
        train_end = train_start + int(num_events*TRAINING_SPLIT)
        val_start = train_end + 1
        val_end = val_start + int(num_events*VALIDATION_SPLIT)
        test_start = val_end + 1
        test_end = num_events

        for i in range(num_events):
            if i >= train_start and i <= train_end:
                new_datasets[0].append([tuple(table[new_indices[i]])]) 
            elif i >= val_start and i <= val_end:
                new_datasets[1].append([tuple(table[new_indices[i]])]) 
            elif i >= test_start and i < test_end:
                new_datasets[2].append([tuple(table[new_indices[i]])]) 
            else:
                print("Index error")
                quit()
                
    #when done, delete old HDF5 file (???)
    #os.remove(args.hdf5_path)

