import argparse
import math

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

#hardcoded image dimensions
IMG_SCALE_FACTOR = 2
IMG_CHANNELS = 3
SCT_IMAGE_WIDTH = 120
SCT_IMAGE_LENGTH = 120
IMG_WIDTH = SCT_IMAGE_WIDTH * IMG_SCALE_FACTOR
IMG_LENGTH = SCT_IMAGE_LENGTH * IMG_SCALE_FACTOR
IMG_DTYPE = 'int16'

def makeSCTImageArray(pixels,channels,scale_factor):

    CAMERA_DIM = (120,120)
    ROWS = 15
    MODULE_DIM = (8,8)
    MODULE_SIZE = MODULE_DIM[0]*MODULE_DIM[1]
    
    #counting from bottom
    MODULES_PER_ROW = [5,9,11,13,13,15,15,15,15,15,13,13,11,9,5]
    MODULE_START_POSITIONS = [(((CAMERA_DIM[0]-MODULES_PER_ROW[j]*MODULE_DIM[0])/2)+(MODULE_DIM[0]*i),j*MODULE_DIM[1]) for j in range(ROWS) for i in range(MODULES_PER_ROW[j])]

    im_array = np.zeros([channels,CAMERA_DIM[0]*IMG_SCALE_FACTOR,CAMERA_DIM[1]*IMG_SCALE_FACTOR], dtype=IMG_DTYPE)
    
    pixel_count = 0

    for (x_start,y_start) in MODULE_START_POSITIONS:
        for i in range(MODULE_SIZE):
            x = int(x_start + int(i/8))
            y = y_start + i % 8

            x *= scale_factor
            y *= scale_factor

            scaled_region = [(x+i,y+j) for i in range(0,scale_factor) for j in range(0,scale_factor)]

            for (x_coord,y_coord) in scaled_region:
                im_array[0,x_coord,y_coord] = pixels[pixel_count]

            pixel_count += 1

    #pil_im = Image.fromarray(im_array[0,:,:],mode='I')
    #pil_im.show()
   
    return im_array

if __name__ == '__main__':

    #np.set_printoptions(threshold=np.inf) 

    parser = argparse.ArgumentParser(description='Load image data and event parameters from a simtel file into a formatted HDF5 file.')
    parser.add_argument('data_file', help='path to input .simtel file')
    #parser.add_argument('tel_file',help='FITS telescope config file')
    parser.add_argument('hdf5_path', help='path of output HDF5 file, or currently existing file to append to')
    parser.add_argument('bins_cuts_dict',help='dictionary containing bins/cuts in .pkl format')
    #parser.add_argument('config_file',help='configuration file specifying the selected telescope ids from simtel file, the desired energy bins, the correst image output dimensions/dtype, ')
    args = parser.parse_args()

    #open hdf5 file
    f = h5py.File(args.hdf5_path,'a')

    #tel ids of SCTs in simtel file (currently hardcoded -> move to config file)
    selected_tel_ids = [i for i in range(5,19)]

    #energy bins for output file (current hardcoded -> move to config file)
    energy_bins = [(0.1,0.31),(0.31,1),(1,10)]
    num_energy_bins = len(energy_bins)

    #create groups for each energy bin, if not already existing
    for i in range(num_energy_bins):
        if str(i) not in f.keys():
            f.create_group(str(i))

    #within each energy bin group, create datasets for all telescopes if not already existing
    for group in f.keys():
        for tel_id in selected_tel_ids:
            if str(tel_id) not in f[group].keys():
                f[group].create_dataset(str(tel_id),(1,IMG_CHANNELS,IMG_WIDTH,IMG_LENGTH), dtype=IMG_DTYPE,maxshape=(None,IMG_CHANNELS,IMG_WIDTH,IMG_LENGTH),chunks=True)

    #load data (100 events) from simtel file
    source = hessio_event_source(args.data_file)

    #trace integrator
    integ = GlobalPeakIntegrator(None,None)

    #camera geometry (currently hardcoded -> move to config/use guess)
    #geom = CameraGeometry.from_name('SCTCam')

    #load bins/cuts file
    bins_dict = pkl.load(open(args.bins_cuts_dict, "rb" ))

    #iterate through events, loading data into h5 file
    for event in source:
        #find correct bin
        if (event.r0.run_id,event.r0.event_id) in bins_dict:
            bin_number, reconstructed_energy = bins_dict[(event.r0.run_id,event.r0.event_id)]
        else:
            continue

        #process and write telescope data
        for tel_id in selected_tel_ids:
            if tel_id in event.r0.tels_with_data:
                trace = np.array(event.r0.tel[tel_id].adc_samples)
                #apply trace integration
                trace_integrated = integ.extract_charge(trace)[0][0]
                #apply image cleaning (missing)
                image_array = makeSCTImageArray(trace_integrated,IMG_CHANNELS,IMG_SCALE_FACTOR)
            else:
                image_array = np.zeros([IMG_CHANNELS, IMG_WIDTH, IMG_LENGTH], dtype=IMG_DTYPE)

            #append new image to each telescope dataset
            f[str(bin_number)][str(tel_id)].resize(len(f[str(bin_number)][str(tel_id)])+1,axis=0)
            f[str(bin_number)][str(tel_id)][len(f[str(bin_number)][str(tel_id)])-1,:,:,:] = image_array


        #process and write other parameter data


