import argparse
import math
import pickle as pkl

from configobj import ConfigObj
from validate import Validator
import numpy as np
from tables import *
from ctapipe.io.hessio import hessio_event_source
#from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.image.charge_extractors import NeighbourPeakIntegrator
#from ctapipe.reco import EnergyRegressor
from ctapipe.calib import pedestals,CameraCalibrator
from astropy import units as u
from PIL import Image

#spec/template for configuration file validation
config_spec = """
mode = option('gh_class','energy_recon', default='gh_class')
storage_mode = option('all','mapped', default='all')
use_pkl_dict = boolean(default=True)
[image]
mode = option('PIXELS_3C','PIXELS_1C','PIXELS_TIMING_2C',PIXELS_TIMING_3C',default='PIXELS_3C')
scale_factor = integer(min=1,default=2)
dtype = option('uint32', 'int16', 'int32', 'uint16',default='uint16')
dim_order = option('channels_first','channels_last',default='channels_last')
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

class Event(IsDescription):
    event_number = UInt32Col()   
    run_number = UInt32Col() 
    gamma_hadron_label = UInt8Col()
    MC_energy = Float64Col()
    reconstructed_energy = Float32Col()

def makeSCTImageArray(pixels_vector,peaks_vector,image_mode,scale_factor,img_dtype,dim_order,channels):

    #hardcoded to correct SCT values
    NUM_PIXELS = 11328
    CAMERA_DIM = (120,120)
    ROWS = 15
    MODULE_DIM = (8,8)
    MODULE_SIZE = MODULE_DIM[0]*MODULE_DIM[1]
    MODULES_PER_ROW = [5,9,11,13,13,15,15,15,15,15,13,13,11,9,5]

    MODULE_START_POSITIONS = [(((CAMERA_DIM[0]-MODULES_PER_ROW[j]*MODULE_DIM[0])/2)+(MODULE_DIM[0]*i),j*MODULE_DIM[1]) for j in range(ROWS) for i in range(MODULES_PER_ROW[j])] #counting from the bottom row, left to right

    if dim_order == 'channels_first':
        im_array = np.zeros([channels,CAMERA_DIM[0]*scale_factor,CAMERA_DIM[1]*scale_factor], dtype=img_dtype)
    elif dim_order == 'channels_last':
        im_array = np.zeros([CAMERA_DIM[0]*scale_factor,CAMERA_DIM[1]*scale_factor,channels], dtype=img_dtype)

    if pixels_vector is not None:
        pixel_num = 0

        assert len(pixels_vector) == NUM_PIXELS
        if image_mode == 'PIXELS_TIMING_2C' or image_mode == 'PIXELS_TIMING_3C':
            assert len(peaks_vector) == NUM_PIXELS

        for (x_start,y_start) in MODULE_START_POSITIONS:
            for i in range(MODULE_SIZE):
                x = (int(x_start + int(i/MODULE_DIM[0])))*scale_factor
                y = (y_start + i % MODULE_DIM[1])*scale_factor

                scaled_region = [(x+i,y+j) for i in range(0,scale_factor) for j in range(0,scale_factor)]

                for (x_coord,y_coord) in scaled_region:
                    if dim_order == 'channels_first':
                        im_array[0,x_coord,y_coord] = pixels_vector[pixel_num]
                        if include_timing:
                            im_array[1,x_coord,y_coord] = peaks_vector[pixel_num]
                    elif dim_order == 'channels_last':
                        im_array[x_coord,y_coord,0] = pixels_vector[pixel_num]
                        if include_timing:
                            im_array[x_coord,y_coord,1] = peaks_vector[pixel_num]

                pixel_num += 1

    #image = Image.fromarray(im_array[:,:,0],'I')
    #image.save("test.png")

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
    else:
        raise ValueError('Invalid config file.')

    MODE = config['mode']
    IMG_MODE = config['image']['mode']
    STORAGE_MODE = config['storage_mode']
    IMG_SCALE_FACTOR = config['image']['scale_factor']
    IMG_DTYPE = config['image']['dtype']
    IMG_DIM_ORDERING = config['image']['dim_ordering']
    ENERGY_BIN_UNITS = config['energy_bins']['units']

    print("Mode: ",MODE)
    print("Image mode: ",IMG_MODE)
    print("File storage mode: ",STORAGE_MODE)
    print("Image scale factor: ",IMG_SCALE_FACTOR)
    print("Image array type: ",IMG_DTYPE)
    print("Image dim order: ",IMG_DIM_ORDERING)

    print("Getting telescope types...")

    #collect telescope lists
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

    all_tels = {'SST' : SST_list, 'SCT' : SCT_list, 'LST' : LST_list}

    #select telescopes by type
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
        raise ValueError('Telescope selection mode not recognized.') 

    print("Telescope Mode: ",TEL_MODE)
    
    for i in selected.keys():
        print(i + ": " + str(len(selected[i])) + " out of " + str(len(all_tels[i])) + " telescopes selected.")

    print("Loading additional configuration settings...")

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
        raise ValueError('Image processing mode not recognized.')

    print("Preparing HDF5 file structure...")

    f = open_file(args.hdf5_path, mode = "a", title = "Data File") 

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
        datasets = [f]

    dataset_tables = []

    for d in datasets:
        if not d.__contains__('Events'):
            table = f.create_table(d, 'Events', Event, "Table of event records") 
            descr = table.description._v_colobjects
            descr2 = descr.copy()

            #dynamically add correct label type

            if MODE == 'energy_recon':
                descr2['energy_reconstruction_bin_label'] = UInt8Col()

            #dynamically add columns for telescopes
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
     
            table2 = f.create_table(d, 'temp', descr2, "Table of event records")
            table.attrs._f_copy(table2)
            table.remove()
            table2.move(d, 'Events')
        
        dataset_tables.append(d.Events)

    #load bins/cuts file
    if config['use_pkl_dict']:
        print("Loading bins/cuts dictionary...")
        bins_dict = pkl.load(open(args.bins_cuts_dict, "rb" ))

    #TEMPORARY FIX - determine gamma hadron label from simtel file name
    if 'gamma' in args.data_file:
        gamma_hadron_label = 1
    elif 'proton' in args.data_file:
        gamma_hadron_label = 0
        if MODE == 'energy_recon' and config['energy_recon']['gamma_only'] == "True":
            raise ValueError('Proton simtel file: {} skipped'.format(args.data_file))
            quit()
    else:
        raise ValueError('Unable to determine gamma_hadron label from filename')
        quit()

    #select calibration and other tools
    cal = CameraCalibrator(None,None)
    #geom = CameraGeometry.from_name('SCTCam')
    #integ = NeighbourPeakIntegrator(None,None)
    #integ.neighbours = geom.neighbors
    #energy_reco = EnergyRegressor(cam_id_list=map(str,SCT_list))
    #hillas_reco = 
    #impact reconstructor = 

    print("Processing events...")

    event_count = 0
    passing_count = 0

    #load all SCT data from simtel file
    source = hessio_event_source(args.data_file,allowed_tels=[j for i in selected.keys() for j in selected[i]])

    for event in source:
        event_count += 1

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

        #collect telescope data
        for tel_id in event.r0.tels_with_data:
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

            #append new image to each telescope dataset, or a blank image if no data from telescope 
            if tel_id in [j for i in selected.keys() for j in selected[i]]:
                image_array = makeSCTImageArray(image,peaks,IMG_MODE,IMG_SCALE_FACTOR,IMG_DTYPE,IMG_DIM_ORDERING,IMAGE_CHANNELS)
            else:
                image_array = makeSCTImageArray(None,peaks,IMG_MODE,IMG_SCALE_FACTOR,IMG_DTYPE,IMG_DIM_ORDERING,IMAGE_CHANNELS)
            
            event_row["T" + str(tel_id)] = image_array

        #other parameter data
        event_row['event_number'] = event.r0.event_id
        event_row['run_number'] = event.r0.run_id
        event_row['gamma_hadron_label'] = gamma_hadron_label
        event_row['MC_energy'] = event.mc.energy.value
        event_row['reconstructed_energy'] = reconstructed_energy
        
        if MODE == 'energy_recon':
            event_row['energy_reconstruction_bin_label'] = erec_bin_label

        #write data to table
        event_row.append()

    for table in dataset_tables:
        table.flush()

    print("{} events processed".format(event_count))
    print("{} events passed cuts/written to file".format(passing_count))
    print("Done!")

if __name__ == '__main__':
    imageExtractor()
