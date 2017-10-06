import argparse
import math
import pickle as pkl

from configobj import ConfigObj
from validate import Validator
import numpy as np
import h5py
from ctapipe.io.hessio import hessio_event_source
#from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.image.charge_extractors import NeighbourPeakIntegrator
#from ctapipe.reco import EnergyRegressor
from ctapipe.calib import pedestals,CameraCalibrator
#from matplotlib import pyplot as plt
from astropy import units as u
#from PIL import Image

#spec/template for configuration file validation
config_spec = """
mode = option('gh_class','energy_recon', default='gh_class')
storage_mode = option('all','mapped', default='all')
use_pkl_dict = boolean(default=True)
[image]
mode = option('PIXELS_3C','PIXELS_1C','PIXELS_TIMING_2C',PIXELS_TIMING_3C',default='PIXELS_3C')
scale_factor = integer(min=1,default=2)
dtype = option('uint32', 'int16', 'int32', 'uint16',default='uint16')
[telescope]
type_mode = option('SST', 'SCT', 'LST', 'SST+SCT','SST+LST','SCT+LST','ALL', default='SCT')
[energy_bins]
units = option('eV','MeV','GeV','TeV',default='TeV')
scale = option('linear','log10',default='log10')
min = float(default=-1.0)
max = float(default=1.0)
bin_size = float(default=0.5)
[preselection_cuts]
MSCW = tuple(default=list(-2.0,2.0))
MSCL = tuple(default=list(-2.0,5.0))
EChi2S = tuple(default=list(0.0,None))
ErecS = tuple(default=list(0.0,None))
EmissionHeight = tuple(default=list(0.0,50.0))
MC_Offset = mixed_list(string,string,float,float,default=list('MCxoff','MCyoff',0.0,3.0))
NImages = tuple(default=list(3,None))
dES = tuple(default=list(0.0,None))
[energy_recon]
    gamma_only = boolean(default=True)
    [[bins]]
    units = option('eV','MeV','GeV','TeV',default='TeV')
    scale = option('linear','log10',default='log10')
    min = float(default=-2.0)
    max = float(default=2.0)
    bin_size = float(default=0.05)
"""

#telescope constants
SCT_IMAGE_WIDTH = 120
SCT_IMAGE_LENGTH = 120
LST_NUM_PIXELS = 1855
SCT_NUM_PIXELS = 11328
SST_NUM_PIXELS = 0 #update

def makeSCTImageArray(pixels_vector,peaks_vector,image_mode,scale_factor,img_dtype):

    if image_mode == 'PIXELS_3C':
        channels = 3
        include_timing = False
    elif image_mode == 'PIXELS_1C':
        channels = 1
        include_timing = False
    elif image_mode == 'PIXELS_TIMING_2C':
        channels = 2
        include_timing = True
    elif image_mode == 'PIXELS_TIMING_3C':
        channels = 3 
        include_timing = True
    else:
        raise ValueError('Image processing mode not recognized. Exiting...')
        quit()

    #hardcoded to correct SCT values
    NUM_PIXELS = 11328
    CAMERA_DIM = (120,120)
    ROWS = 15
    MODULE_DIM = (8,8)
    MODULE_SIZE = MODULE_DIM[0]*MODULE_DIM[1]
    MODULES_PER_ROW = [5,9,11,13,13,15,15,15,15,15,13,13,11,9,5]
    
    assert len(pixels_vector) == NUM_PIXELS
    if image_mode == 'PIXELS_TIMING_2C' or image_mode == 'PIXELS_TIMING_3C':
        assert len(peaks_vector) == NUM_PIXELS

    MODULE_START_POSITIONS = [(((CAMERA_DIM[0]-MODULES_PER_ROW[j]*MODULE_DIM[0])/2)+(MODULE_DIM[0]*i),j*MODULE_DIM[1]) for j in range(ROWS) for i in range(MODULES_PER_ROW[j])] #counting from the bottom row, left to right

    im_array = np.zeros([channels,CAMERA_DIM[0]*scale_factor,CAMERA_DIM[1]*scale_factor], dtype=img_dtype)
    
    pixel_num = 0

    for (x_start,y_start) in MODULE_START_POSITIONS:
        for i in range(MODULE_SIZE):
            x = (int(x_start + int(i/MODULE_DIM[0])))*scale_factor
            y = (y_start + i % MODULE_DIM[1])*scale_factor

            scaled_region = [(x+i,y+j) for i in range(0,scale_factor) for j in range(0,scale_factor)]

            for (x_coord,y_coord) in scaled_region:
                im_array[0,x_coord,y_coord] = pixels_vector[pixel_num]
                if include_timing:
                    im_array[1,x_coord,y_coord] = peaks_vector[pixel_num]

            pixel_num += 1

    return im_array

#@profile
def imageExtractor():

    parser = argparse.ArgumentParser(description='Load image data and event parameters from a simtel file into a formatted HDF5 file.')
    parser.add_argument('data_file', help='path to input .simtel file')
    parser.add_argument('hdf5_path', help='path of output HDF5 file, or currently existing file to append to')
    parser.add_argument('bins_cuts_dict',help='dictionary containing bins/cuts in .pkl format')
    parser.add_argument('config_file',help='configuration file specifying the selected telescope ids from simtel file, the desired energy bins, the correst image output dimensions/dtype, ')
    parser.add_argument("--debug", help="print debug/logger messages",action="store_true")
    args = parser.parse_args()
 
    #Configuration file, load + validate
    spc = config_spec.split('\n')
    config = ConfigObj(args.config_file,configspec=spc)
    validator = Validator()
    val_result = config.validate(validator)

    if val_result:
        print("Config file validated.")

    MODE = config['mode']
    IMG_MODE = config['image']['mode']
    STORAGE_MODE = config['storage_mode']
    IMG_SCALE_FACTOR = config['image']['scale_factor']
    IMG_DTYPE = config['image']['dtype']

    if MODE == 'gh_class' or MODE == 'energy_recon':
        print("Mode: ",MODE)
    else:
        raise ValueError('Data processing mode not recognized (Choose gh_class or energy_recon). Exiting...')
        quit()

    print("Image mode: ",IMG_MODE)
    print("File storage mode: ",STORAGE_MODE)

    print("Getting telescope types...")

    source_temp = hessio_event_source(args.data_file,max_events=1)

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
            #elif event.inst.num_pixels[i] == 

    TEL_MODE = config['telescope']['type_mode'] 
    if TEL_MODE == 'SST':
        selected = SST_list
    elif TEL_MODE == 'SCT':
        selected = SCT_list
    elif TEL_MODE == 'LST':
        selected = LST_list
    elif TEL_MODE == 'SCT+LST':
        selected = LST_list + SCT_list
    elif TEL_MODE == 'SST+SCT':
        selected = SCT_list + SST_list
    elif TEL_MODE == 'SST+LST':
        selected = LST_list + SST_list
    elif TEL_MODE == 'ALL':
        selected = LST_list + SCT_list + SST_list
    else:
        raise ValueError('Telescope selection mode not recognized. Exiting...')
        quit()

    print("Loading configuration settings...")

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
        IMG_CHANNELS = 3
    elif IMG_MODE == 'PIXELS_1C':
        IMG_CHANNELS = 1
    elif IMG_MODE == 'PIXELS_TIMING_2C':
        IMG_CHANNELS = 2
    elif IMG_MODE == 'PIXELS_TIMING_3C':
        IMG_CHANNELS = 3 
    else:
        raise ValueError('Image processing mode not recognized. Exiting...')
        quit()

    print("Preparing HDF5 file structure...")

    f = h5py.File(args.hdf5_path,'a')

    #prep data into bins based on mode (by energy bin for g/h classification, 1 group for energy reconstruction)
    if MODE == 'gh_class':
        #create groups for each energy bin, if not already existing
        bins = []
        for i in range(len(energy_bins)):
            if str(i) not in f.keys():
                f.create_group(str(i))
                bins.append(f[str(i)])
    elif MODE == 'energy_recon':
        bins = [f]

    #add more event parameters here:
    #types hardcoded based on parameter
    event_params={
            'run_number': 'int32',
            'event_number': 'int32',
            'MC_energy': 'd',
            'Reconstructed_energy': 'd'
            }

    if MODE == 'gh_class':
        event_params['Gamma_hadron_label'] = 'int32'

    elif MODE == 'energy_recon':
        event_params['Energy_bin_label'] = 'int32'
        if not config['energy_recon']['gamma_only'] == "True":
            event_params['Gamma_hadron_label'] = 'int32'

    for b in bins:
        if 'tel_data' not in b.keys():
            b.create_group('tel_data')

        for tel_id in selected:
            if str(tel_id) not in b['tel_data'].keys():
                b['tel_data'].create_dataset(str(tel_id),(0,IMG_CHANNELS,SCT_IMAGE_WIDTH*IMG_SCALE_FACTOR,SCT_IMAGE_LENGTH*IMG_SCALE_FACTOR), dtype=IMG_DTYPE,maxshape=(None,IMG_CHANNELS,SCT_IMAGE_WIDTH*IMG_SCALE_FACTOR,SCT_IMAGE_LENGTH*IMG_SCALE_FACTOR),chunks=True)

        for i in event_params.keys():
            if i[0] not in b.keys():
                b.create_dataset(i,(0,), dtype=event_params[i],maxshape=(None,),chunks=True)

        if STORAGE_MODE == 'mapped':
            if 'tel_map' not in b.keys():
                b.create_dataset('tel_map',(0,len(selected)),dtype='int32',maxshape=(None,len(selected)),chunks=True)
                b['tel_map'].attrs.create("tel_ids",selected)

    print("Preparing...")
    #load bins/cuts file
    if config['use_pkl_dict']:
        print("Loading bins/cuts dictionary...")
        bins_dict = pkl.load(open(args.bins_cuts_dict, "rb" ))

    #load all SCT data from simtel file
    source = hessio_event_source(args.data_file,allowed_tels=selected)

    #TEMPORARY FIX - determine gamma hadron label from simtel file name
    if 'gamma' in args.data_file:
        gamma_hadron_label = 1
        if MODE == 'energy_recon' and config['energy_recon']['gamma_only'] == "True":
            raise ValueError('Gamma simtel file: {} skipped'.format(args.data_file))
            quit()

    elif 'proton' in args.data_file:
        gamma_hadron_label = 0
    else:
        raise ValueError('Unable to determine gamma_hadron label from filename')
        quit()

    #select calibration and other tools
    cal = CameraCalibrator(None,None)
    geom = CameraGeometry.from_name('SCTCam')
    integ = NeighbourPeakIntegrator(None,None)
    integ.neighbours = geom.neighbors
    #energy_reco = EnergyRegressor(cam_id_list=map(str,SCT_list))
    #hillas_reco = 
    #impact reconstructor = 
    
    print("Processing events...")

    event_count = 0
    passing_count = 0

    for event in source:
        event_count += 1

        #print("Calibrating/parameterizing raw data...")

        #calibrate camera (charge extraction + pedestal subtraction + trace integration)
        cal.calibrate(event)
        
        #image cleaning (??)

        #get energy bin and reconstructed energy
        if config['use_pkl_dict']:
            if (event.r0.run_id,event.r0.event_id) in bins_dict:
                bin_number, reconstructed_energy = bins_dict[(event.r0.run_id,event.r0.event_id)]
                passing_count +=1
            else:
                continue
        else:
            #if pass cuts (applied locally):
            bin_number, reconstructed_energy = 0
            #else:
            #continue

        #compute correct energy reconstruction bin label
        if MODE == 'energy_recon':
            for i in range(len(energy_recon_bins)):
                if event.mc.energy.value >= 10**(energy_recon_bins[i][0]) and event.mc.energy.value < 10**(energy_recon_bins[i][1]):
                    erec_bin_label = i
                    passing_count +=1
                    break

        if STORAGE_MODE == 'mapped':
            map_row = np.full(len(selected), -1,dtype='int32')

        #process and write telescope data
        for j,tel_id in zip(range(len(selected)),selected):
            if tel_id in event.r0.tels_with_data:
                image = event.dl1.tel[tel_id].image
                #truncate at 0
                image[image < 0] = 0
                #round float values to hundredths + save as int
                image = [round(i*100) for i in image[0]]
                #compute peak position
                if IMG_MODE == 'PIXELS_TIMING_2C' or IMG_MODE == 'PIXELS_TIMING_3C':
                    peaks = event.dl1.tel[tel_id].peakpos[0]
                else:
                    peaks = None

                #write to image array
                image_array = makeSCTImageArray(image,peaks,IMG_MODE,IMG_SCALE_FACTOR,IMG_DTYPE)
                
                #append new image to each telescope dataset
                f[str(bin_number)]['tel_data'][str(tel_id)].resize(len(f[str(bin_number)]['tel_data'][str(tel_id)])+1,axis=0)
                f[str(bin_number)]['tel_data'][str(tel_id)][-1,:,:,:] = image_array

                if STORAGE_MODE == 'mapped':
                    map_row[j] = len(f[str(bin_number)]['tel_data'][str(tel_id)]) - 1

            else:
                if STORAGE_MODE == 'all':
                    image_array = np.zeros([IMG_CHANNELS, SCT_IMAGE_WIDTH*IMG_SCALE_FACTOR, SCT_IMAGE_LENGTH*IMG_SCALE_FACTOR], dtype=IMG_DTYPE)
                    #append new image to each telescope dataset
                    f[str(bin_number)]['tel_data'][str(tel_id)].resize(len(f[str(bin_number)]['tel_data'][str(tel_id)])+1,axis=0)
                    f[str(bin_number)]['tel_data'][str(tel_id)][-1,:,:,:] = image_array


        if STORAGE_MODE == 'mapped':
            f[str(bin_number)]['tel_map'].resize(len(f[str(bin_number)]['tel_map'])+1,axis=0)
            f[str(bin_number)]['tel_map'][-1,:] = map_row
 
        #process and write other parameter data
        for i in event_params.keys():
            if i == 'run_number':
                value = event.r0.run_id
            elif i == 'event_number':
                value = event.r0.event_id
            elif i == 'MC_energy':
                value = event.mc.energy.value
            elif i == 'Reconstructed_energy':
                value = reconstructed_energy
            elif i == 'Gamma_hadron_label':
                value = gamma_hadron_label
            elif i == 'Energy_bin_label':
                value = erec_bin_label
            else:
                raise ValueError('Invalid event parameter')
                quit()

            if MODE == 'gh_class':
                f[str(bin_number)][i].resize(len(f[str(bin_number)][i])+1,axis=0)
                f[str(bin_number)][i][-1] = value
            
            elif MODE == 'energy_recon':
                f[i].resize(len(f[str(bin_number)][i])+1,axis=0)
                f[i][-1] = value

    f.close()

    print("Done!")
    print("{} events processed".format(event_count))
    print("{} events passed cuts/written to file".format(passing_count))

if __name__ == '__main__':
    imageExtractor()
