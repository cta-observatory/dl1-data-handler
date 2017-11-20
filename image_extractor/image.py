import numpy as np

IMAGE_SHAPES = {'MSTS': (120, 120)}


class TraceConverter:

    def __init__(self, img_dtype, dim_order, num_channels, scale_factors):
        self.img_dtype = img_dtype
        self.dim_order = dim_order
        self.num_channels = num_channels
        self.scale_factors = scale_factors

    def convert_MSTS(self, pixels_vector, peaks_vector):
        """
        Converter from ctapipe image pixel vector,
        peak position vector to numpy array format.
        """
        
        image_shape = IMAGE_SHAPES['MSTS']
        scale_factor = self.scale_factors['MSTS']

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
        MODULE_START_POSITIONS = [(((image_shape[0] - MODULES_PER_ROW[j] *
                                     MODULE_DIM[0]) / 2) +
                                   (MODULE_DIM[0] * i), j * MODULE_DIM[1])
                                  for j in range(ROWS)
                                  for i in range(MODULES_PER_ROW[j])]

        if self.dim_order == 'channels_first':
            shape = [self.num_channels,
                    image_shape[0] * scale_factor,
                    image_shape[1] * scale_factor]
        elif self.dim_order == 'channels_last':
            shape = [image_shape[0] * scale_factor,
                    image_shape[1] * scale_factor,
                    self.num_channels]
    
        img_array = np.zeros(shape,dtype=self.img_dtype)

        image_exists = True if pixels_vector is not None else False
        include_timing = True if peaks_vector is not None else False

        if include_timing and self.img_channels < 2:
            raise ValueError('To include timing information, num channels must be >=2.')

        #preprocess pixels vector
        # truncate at 0, scale by 100
        pixels_vector[pixels_vector < 0] = 0
        pixels_vector = np.around(np.multiply(pixels_vector[0],100))

        if image_exists:
            pixel_index = 0
            for (x_start, y_start) in MODULE_START_POSITIONS:
                for i in range(MODULE_SIZE):
                    x = (int(x_start + int(i / MODULE_DIM[0]))) * scale_factor
                    y = (y_start + i % MODULE_DIM[1]) * scale_factor

                    scaled_region = [(x+i, y+j) for i in range(0, scale_factor)
                                     for j in range(0, scale_factor)]

                    for (x_coord, y_coord) in scaled_region:
                        if self.dim_order == 'channels_first':
                            charge_pos = (0,x_coord,y_coord)
                            timing_pos = (1,x_coord,y_coord)
                        elif self.dim_order == 'channels_last':
                            charge_pos = (x_coord,y_coord,0)
                            timing_pos = (x_coord,y_coord,1)
                        
                        img_array[charge_pos] = pixels_vector[pixel_index]
                        if include_timing:
                            peaks_vector = peaks_vector.astype
                            img_array[timing_pos] = peaks_vector[pixel_index]
                pixel_index += 1

        return img_array

    def convert_LST(self, pixels_vector, peaks_vector):
        raise NotImplementedError

    def convert_MSTF(self, pixels_vector, peaks_vector):
        raise NotImplementedError

    def convert_MSTN(self, pixels_vector, peaks_vector):
        raise NotImplementedError

    def convert_SST1(self, pixels_vector, peaks_vector):
        raise NotImplementedError

    def convert_SSTA(self, pixels_vector, peaks_vector):
        raise NotImplementedError

    def convert_SSTC(self, pixels_vector, peaks_vector):
        raise NotImplementedError

