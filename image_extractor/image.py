import logging

import numpy as np

import image_extractor

TEL_NUM_PIXELS = {'LST':1855,'MSTF':1855, 'MSTN':1764,'MSTS':11328,'SST1':1296, 'SSTA':2368, 'SSTC':2048}
IMAGE_SHAPES = {'MSTS': (120, 120)}

logger = logging.getLogger(__name__)


class TraceConverter:

    def __init__(self, img_dtype, dim_order, num_channels, scale_factors):
        self.img_dtype = img_dtype
        self.dim_order = dim_order
        self.num_channels = num_channels
        self.scale_factors = scale_factors

        #create injunction tables for each telescope type
        self.injunction_tables = {}
        self.injunction_tables['MSTS'] = __generate_table_MSTS()

    def convert(self,pixels_vector,peaks_vector,tel_type):
        """
        Converter from ctapipe image pixel vector,
        peak position vector to numpy array format.
        """
        
        image_shape = IMAGE_SHAPES[tel_type]
        scale_factor = self.scale_factors[tel_type]
        injunction_table = self.injunction_tables[tel_type]

        if self.dim_order == 'channels_first':
            shape = [self.num_channels,
                    image_shape[0] * scale_factor,
                    image_shape[1] * scale_factor]
        elif self.dim_order == 'channels_last':
            shape = [image_shape[0] * scale_factor,
                    image_shape[1] * scale_factor,
                    self.num_channels]
    
        if pixels_vector is None:
            return np.zeros(shape,dtype=self.img_dtype)
        else:
            
            img_array = np.empty(shape,dtype=self.img_dtype)

            #preprocess image vector - truncate at 0, scale by 100, round to integer
            pixels_vector = np.around(np.multiply(pixels_vector.clip(min=0),100))

            if len(pixels_vector) != TEL_NUM_PIXELS[tel_type]:
                raise ValueError("Size of image vector does not match telescope type: {} {}".format(len(pixels_vector),TEL_NUM_PIXELS[tel_type]))

            if peaks_vector is not None:
                if len(pixels_vector) != len(peaks_vector):
                    raise ValueError("Size of image vector and peak time vector do not match: {} {}".format(len(pixels_vector),len(peaks_vector)))
                if self.img_channels < 2:
                    raise ValueError('To include timing information, num channels must be >= 2.')

                for i, (charge, peak) in enumerate(zip(pixels_vector,peaks_vector)):
                    x,y = injunction_table[i]
                    if self.dim_order == 'channels_first':
                        img_array[0,x,y] = charge
                        img_array[1,x,y] = peak
                    elif self.dim_order == 'channels_last':
                        img_array[x,y,0] = charge
                        img_array[x,y,1] = peak
            else:
                for i,charge in enumerate(pixels_vector):
                    x,y = injunction_table[i]
                    if self.dim_order == 'channels_first':
                        img_array[0,x,y] = charge
                    elif self.dim_order == 'channels_last':
                        img_array[x,y,0] = charge
 
            return img_array

    @staticmethod
    def __generate_table_MSTS():
        """
        Function returning MSTS injunction table
        """
        
        ROWS = 15
        MODULE_DIM = (8, 8)
        MODULE_SIZE = MODULE_DIM[0] * MODULE_DIM[1]
        MODULES_PER_ROW = [
            5,
            9,
            11,
            13,
            13,
            15,
            15,
            15,
            15,
            15,
            13,
            13,
            11,
            9,
            5]

        # counting from the bottom row, left to right
        MODULE_START_POSITIONS = [(((IMAGE_SHAPES['MSTS'][0] - MODULES_PER_ROW[j] *
                                     MODULE_DIM[0]) / 2) +
                                   (MODULE_DIM[0] * i), j * MODULE_DIM[1])
                                  for j in range(ROWS)
                                  for i in range(MODULES_PER_ROW[j])]

        injunction_table = [(int(x_0 + int(i / MODULE_DIM[0])),y_0 + i % MODULE_DIM[1]) 
                            for (x_0,y_0) in MODULE_START_POSITIONS
                            for i in range(MODULE_SIZE)]

        return injunction_table


