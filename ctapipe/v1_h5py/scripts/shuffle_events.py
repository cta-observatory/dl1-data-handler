import argparse
import math
from random import shuffle

from configobj import ConfigObj
import pickle as pkl
from ROOT import TChain,TFile
from glob import glob
import numpy as np
import h5py
from ctapipe.utils.datasets import get_dataset
from ctapipe.io.hessio import hessio_event_source
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.image.charge_extractors import GlobalPeakIntegrator
#from ctapipe.image
from matplotlib import pyplot as plt
from astropy import units as u
#from PIL import Image

if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Takes existing HDF5 file formatted for Keras (/file/energy_bin/tel_dataset) and shuffles the events in each bin independently.')
    parser.add_argument('hdf5_path', help='path to HDF5 file to be shuffled')
    parser.add_argument('output_file',help='path to output HDF5 file (shuffled). Must not already exist.')
    parser.add_argument('config_file', help='configuration file, to get correct format for provided h5 file')
    args = parser.parse_args()

    #load config file
    config = ConfigObj(args.config_file)

    MODE = config['mode']
    IMG_MODE = config['image']['mode']
    STORAGE_MODE = config['storage_mode']
    IMG_SCALE_FACTOR = int(config['image']['scale_factor'])
    IMG_DTYPE = config['image']['dtype']

    #open input hdf5 file
    f_in = h5py.File(args.hdf5_path,'r+')

    #create new file, duplicating file structure
    f_shuffled = h5py.File(args.output_file,'w-')

    if MODE == 'gh_class':
        groups = []
        new_groups = []
        for i in f_in.keys():
            groups.append(f_in[i])
            f_shuffled.create_group(i)
            f_shuffled.create_group(i + "/tel_data")
            new_groups.append(f_shuffled[i])
    elif MODE == 'energy_recon':
        groups = [f_in]
        f_shuffled.create_group("/tel_data")
        new_groups = [f_shuffled]

    for group,new_group in zip(groups,new_groups):
        #calculate total number of events from eventNumber dataset
        num_events = group['event_number'].len()
 
        #create list of indices per group and shuffle them
        indices = [i for i in range(num_events)]
        shuffled_indices = [i for i in range(num_events)]
        shuffle(shuffled_indices)

        for i in group.keys():
            if isinstance(group[i], h5py.Dataset):
                #duplicate all metadata datasets
                assert group[i].len() == num_events
                if i == 'tel_map':
                    new_group.create_dataset(i,group[i].shape, dtype=group[i].dtype,maxshape=(None,) + group[i].shape[1:],chunks=True)
                else:    
                    new_group.create_dataset(i,group[i].shape, dtype=group[i].dtype,maxshape=(None,),chunks=True)
                #copy metadata 
                for old_index,new_index in zip(indices,shuffled_indices):
                    new_group[i][new_index] = group[i][old_index]
            else:
                #duplicate all telescope datasets
                for j in group[i].keys():
                    if STORAGE_MODE == 'all':
                        assert group[i][j].len() == num_events
                    new_group[i].create_dataset(j,group[i][j].shape, dtype=group[i][j].dtype,maxshape=(None,) + group[i][j].shape[1:],chunks=True)
                    #copy telescope data
                    if STORAGE_MODE == 'mapped':
                        for k in range(len(group[i][j])):
                            new_group[i][j][k,:,:,:] = group[i][j][k,:,:,:]
                    elif STORAGE_MODE == 'all':
                        for old_index,new_index in zip(indices,shuffled_indices):
                            new_group[i][j][old_index,:,:,:] = group[i][j][old_index,:,:,:]

    #when done, delete old HDF5 file (???)
    #os.remove(args.hdf5_path)

