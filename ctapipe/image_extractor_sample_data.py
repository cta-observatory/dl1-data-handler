import h5py

from ctapipe.utils.datasets import get_dataset
from ctapipe.io.hessio import hessio_event_source
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.image.charge_extractors import GlobalPeakIntegrator
from matplotlib import pyplot as plt
from astropy import units as u

import argparse
import numpy as np

#NOTE: currently hardcoded to load data from ctapipe samples

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load image data and event parameters from a simtel file into a formatted HDF5 file.')
    #parser.add_argument('data_file', help='path to input .simtel file')
    #parser.add_argument('--weights', help='path to saved model weights')

    args = parser.parse_args()

    #open hdf5 file
    f = h5py.File('output.h5','a')

    sample_sct_ids = [i for i in range(101,127)]

    energy_bins = [(0.1,0.31),(0.31,1),(1,10)]
    num_energy_bins = len(energy_bins)

    energy_bin_grps = []

    #prepare groups for each energy bin
    for i in range(num_energy_bins):
        #print(i+1)
        if str(i+1) in f.keys():
            energy_bin_grps.append(f[str(i+1)])
        else:
            energy_bin_grps.append(f.create_group(str(i+1)))

    #within each energy bin group, create datasets for all telescopes
    for group in energy_bin_grps:
        for tel_id in sample_sct_ids:
            if str(tel_id) in group.keys():
                continue
            else:
                group.create_dataset(str(tel_id),(1,3,240,240), dtype='int16')

    #load data (100 events) from ctapipe examples
    source = hessio_event_source(get_dataset("gamma_test.simtel.gz"), max_events=100)

    #trace integrator
    integ = GlobalPeakIntegrator(None,None)

    #iterate through events, loading data into h5 file
    for event in source:
        #load telescope data
        for tel_id in event.r0.tels_with_data:
            if tel_id in sample_sct_ids:
                trace = np.array(event.r0.tel[tel_id].adc_samples)
                #trace integration
                trace_integrated = integ.extract_charge(trace)
                #create camera display image
                pix_x, pix_y= event.inst.pixel_pos[tel_id]
                print(pix_x)
                print(pix_y)




                
   
