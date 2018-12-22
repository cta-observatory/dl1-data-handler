import argparse
import math

import pickle as pkl
from ROOT import TChain, TFile
from glob import glob
import numpy as np
import h5py
from ctapipe.utils.datasets import get_dataset
from ctapipe.io.hessio import hessio_event_source
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.image.charge_extractors import GlobalPeakIntegrator
from ctapipe.image.hillas import hillas_parameters_4
from ctapipe.reco import EnergyRegressor, HillasReconstructor
from ctapipe.calib import CameraCalibrator
from matplotlib import pyplot as plt
from astropy import units as u
from PIL import Image

IMG_DTYPE = 'int32'
GLOBAL_COUNT = 0


def makeSCTImageArray(
        pixels_vector,
        peaks_vector,
        img_mode,
        scale_factor,
        img_dtype,
        dim_order):

    if img_mode == 'PIXELS_3C':
        channels = 3
        include_timing = False
    elif img_mode == 'PIXELS_1C':
        channels = 1
        include_timing = False
    elif img_mode == 'PIXELS_TIMING_2C':
        channels = 2
        include_timing = True
    elif img_mode == 'PIXELS_TIMING_3C':
        channels = 3
        include_timing = True
    else:
        raise ValueError('Image processing mode not recognized. Exiting...')
        quit()

    # hardcoded to correct SCT values
    NUM_PIXELS = 11328
    CAMERA_DIM = (120, 120)
    ROWS = 15
    MODULE_DIM = (8, 8)
    MODULE_SIZE = MODULE_DIM[0] * MODULE_DIM[1]
    MODULES_PER_ROW = [5, 9, 11, 13, 13, 15, 15, 15, 15, 15, 13, 13, 11, 9, 5]

    # starting modules from the bottom row, left to right
    MODULE_START_POSITIONS = [
            (((CAMERA_DIM[0] - MODULES_PER_ROW[j] * MODULE_DIM[0]) / 2) +
             (MODULE_DIM[0] * i), j * MODULE_DIM[1])
            for j in range(ROWS) for i in range(MODULES_PER_ROW[j])]

    if dim_order == 'channels_first':
        im_array = np.zeros(
            [channels,
             CAMERA_DIM[0] * scale_factor,
             CAMERA_DIM[1] * scale_factor],
            dtype=img_dtype)
    elif dim_order == 'channels_last':
        im_array = np.zeros(
            [CAMERA_DIM[0] * scale_factor,
             CAMERA_DIM[1] * scale_factor,
             channels],
            dtype=img_dtype)

    if pixels_vector is not None:
        pixel_num = 0

        assert len(pixels_vector) == NUM_PIXELS
        if img_mode == 'PIXELS_TIMING_2C' or img_mode == 'PIXELS_TIMING_3C':
            assert len(peaks_vector) == NUM_PIXELS

        for (x_start, y_start) in MODULE_START_POSITIONS:
            for i in range(MODULE_SIZE):
                x = (int(x_start + int(i / MODULE_DIM[0]))) * scale_factor
                y = (y_start + i % MODULE_DIM[1]) * scale_factor

                scaled_region = [(x + i, y + j) for i in range(0, scale_factor)
                                 for j in range(0, scale_factor)]

                for (x_coord, y_coord) in scaled_region:
                    if dim_order == 'channels_first':
                        im_array[0, x_coord, y_coord] = \
                            pixels_vector[pixel_num]
                        if include_timing:
                            im_array[1, x_coord, y_coord] = \
                                peaks_vector[pixel_num]
                    elif dim_order == 'channels_last':
                        im_array[x_coord, y_coord, 0] = \
                            pixels_vector[pixel_num]
                        if include_timing:
                            im_array[x_coord, y_coord, 1] = \
                                peaks_vector[pixel_num]

                pixel_num += 1

        image = Image.fromarray(im_array[:, :, 0], 'I')
        image.save("test_images/test_" + str(GLOBAL_COUNT) + ".png")
        global GLOBAL_COUNT
        GLOBAL_COUNT += 1

    return im_array

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Reads event parameters directly from simtel files.')
    parser.add_argument('data_file', help='path to input .simtel file')
    args = parser.parse_args()

    # load camera types from first event, take note of
    source = hessio_event_source(args.data_file, max_events=1)

    LST_list = []
    SCT_list = []
    SST_list = []

    for event in source:
        for i in event.inst.telescope_ids:
            if event.inst.num_pixels[i] == 11328:
                SCT_list.append(i)
            elif event.inst.num_pixels[i] == 1855:
                LST_list.append(i)
            # elif event.inst.num_pixels[i] ==

    # camera calibrator
    cal = CameraCalibrator(None, None)

    # trace integrator
    integ = GlobalPeakIntegrator(None, None)

    # energy reconstructor
    energy_reco = EnergyRegressor(cam_id_list=map(str, SCT_list))

    # direction reconstructor
    hillas_reco = HillasReconstructor()

    # impact reconstructor

    # load all SCT data from simtel file
    source = hessio_event_source(
        args.data_file,
        allowed_tels=SCT_list,
        max_events=100)

    pedestal_start = 48
    pedestal_end = None

    for event in source:
        if event.mc.energy.value > 10 and len(event.r0.tels_with_data) >= 3:

            # print(event.inst)
            # print(event.mc)
            # print(event.r0)
            # print(event.r1)
            # print(event.dl0)
            # print(event.dl1)
            # print(event.dl2)
            # print(event.mcheader)
            # print(event.trig)
                # trace = event.r0.tel[tel_id].adc_samples[0]

                # peds, pedvars =
                # pedestals.calc_pedestals_from_traces(trace,start,end)

                # print("PEDS:",peds)
                # print("VARS:",pedvars)

            # event_id = event.r0.event_id

            cal.calibrate(event)

            # energy reconstruction input
            # X = {}

            # hillas_params_dict = {}
            # tel_phi_dict = {}
            # tel_theta_dict = {}

            for tel_id in event.r0.tels_with_data:

                # print(event.dl1.tel[tel_id].peakpos)

                # trace = np.array(event.r0.tel[tel_id].adc_samples)
                # apply trace integration
                # trace_integrated = integ.extract_charge(trace)[0][0]
                # print(trace_integrated)

                # image_array =
                # makeSCTImageArray(trace_integrated,IMG_CHANNELS,IMG_SCALE_FACTOR)
                image = event.dl1.tel[tel_id].image

                image[image < 0] = 0
                image = [round(i * 100) for i in image[0]]
                # print(max(image))
                # print(image)
                image_array = makeSCTImageArray(
                    image,
                    None,
                    'PIXELS_3C',
                    2,
                    IMG_DTYPE,
                    'channels_last')

                # pil_im = Image.fromarray(image_array[0,:,:],mode='I')
                # pil_im.show()
                # pil_im.save("images_2/" + str(event_id) + "_T" + str(tel_id)
                # + ".png")

                # pix_x, pix_y= event.inst.pixel_pos[tel_id]
                # image_orig = event.dl1.tel[tel_id].image[0]

                # hillas_params_dict[tel_id] = hillas_parameters_4(pix_x,
                #                                                  pix_y,
                #                                                  image_orig)
                # tel_phi_dict[tel_id] = event.mc.tel[tel_id]['altitude_cor']
                # tel_theta_dict[tel_id] = event.mc.tel[tel_id]['azimuth_cor']

                # X[tel_id] = [ ]
                # print(hillas_params_dict[tel_id])
                # print(X[tel_id])

            # hillas_reconstruction = hillas_reco.predict(
            #                              hillas_dict=hillas_params_dict,
            #                              tel_phi=tel_phi_dict,
            #                              tel_theta=tel_theta_dict,
            #                              inst=event.inst)
            # print(hillas_reconstruction)
