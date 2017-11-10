import numpy as np

from .image_extractor import ImageExtractor


class TraceConverter:

    def __init__(self, img_dtype, dim_order, num_channels, scale_factors):
        self.img_dtype = img_dtype
        self.dim_order = dim_order
        self.num_channels = num_channels
        self.scale_factors = scale_factors

    def convert_SCT(self, pixels_vector, peaks_vector):
        """
        Converter from ctapipe image pixel vector,
        peak position vector to numpy array format.
        """
        # hardcoded to correct SCT values
        # TODO: find more natural way to handle these?
        image_shape = ImageExtractor.IMAGE_SHAPE['SCT']
        scale_factor = self.scale_factors['SCT']

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
            im_array = np.zeros(
                [self.num_channels,
                 image_shape[0] * scale_factor,
                 image_shape[1] * scale_factor],
                dtype=self.img_dtype)
        elif self.dim_order == 'channels_last':
            im_array = np.zeros(
                [image_shape[0] * scale_factor,
                 image_shape[1] * scale_factor,
                 self.num_channels],
                dtype=self.img_dtype)

        if pixels_vector is not None:
            image_exists = True
        else:
            image_exists = False

        if peaks_vector is not None:
            include_timing = True
        else:
            include_timing = False

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
                            im_array[
                                0,
                                x_coord,
                                y_coord] = pixels_vector[
                                    pixel_index]
                            if include_timing:
                                im_array[
                                    1,
                                    x_coord,
                                    y_coord] = peaks_vector[
                                        pixel_index]
                        elif self.dim_order == 'channels_last':
                            im_array[
                                x_coord,
                                y_coord,
                                0] = pixels_vector[
                                    pixel_index]
                            if include_timing:
                                im_array[
                                    x_coord,
                                    y_coord,
                                    1] = peaks_vector[
                                        pixel_index]

                    pixel_index += 1

        return im_array

    def convert_LST(self, pixels_vector, peaks_vector):
        raise NotImplementedError

    def convert_SST(self, pixels_vector, peaks_vector):
        raise NotImplementedError
