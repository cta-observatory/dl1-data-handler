

class TraceConverter:

    NUM_PIXELS = {"SCT":11328}
    IMAGE_SHAPE = {"SCT":(120,120)}



    def convert_SCT(pixels_vector,peaks_vector,scale_factor,img_dtype,dim_order,channels):
        """
        Converter from ctapipe image pixel vector, peak position vector to numpy array format.
        """ 
        #hardcoded to correct SCT values
        #TODO: find more natural way to handle these?
        NUM_PIXELS = 11328
        CAMERA_DIM = (120,120)
        ROWS = 15
        MODULE_DIM = (8,8)
        MODULE_SIZE = MODULE_DIM[0]*MODULE_DIM[1]
        MODULES_PER_ROW = [5,9,11,13,13,15,15,15,15,15,13,13,11,9,5]

        #counting from the bottom row, left to right
        MODULE_START_POSITIONS = [(((CAMERA_DIM[0]-MODULES_PER_ROW[j]*MODULE_DIM[0])/2)+(MODULE_DIM[0]*i),j*MODULE_DIM[1]) for j in range(ROWS) for i in range(MODULES_PER_ROW[j])]

        if dim_order == 'channels_first':
            im_array = np.zeros([channels,CAMERA_DIM[0]*scale_factor,CAMERA_DIM[1]*scale_factor], dtype=img_dtype)
        elif dim_order == 'channels_last':
            im_array = np.zeros([CAMERA_DIM[0]*scale_factor,CAMERA_DIM[1]*scale_factor,channels], dtype=img_dtype)

        if pixels_vector is not None:
            try:
                assert len(pixels_vector) == NUM_PIXELS
            except AssertionError as err:
                logger.exception("Length of pixel vector does not match SCT num pixels.")
                raise err
            image_exists = True
        else:
            image_exists = False

        if peaks_vector is not None:
            try:
                assert len(peaks_vector) == NUM_PIXELS
            except AssertionError as err:
                logger.exception("Length of peaks vector does not match SCT num pixels.")
                raise err
            include_timing = True
        else:
            include_timing = False

        if image_exists:
            pixel_index = 0
            for (x_start,y_start) in MODULE_START_POSITIONS:
                for i in range(MODULE_SIZE):
                    x = (int(x_start + int(i/MODULE_DIM[0])))*scale_factor
                    y = (y_start + i % MODULE_DIM[1])*scale_factor

                    scaled_region = [(x+i,y+j) for i in range(0,scale_factor) for j in range(0,scale_factor)]

                    for (x_coord,y_coord) in scaled_region:
                        if dim_order == 'channels_first':
                            im_array[0,x_coord,y_coord] = pixels_vector[pixel_index]
                            if include_timing:
                                im_array[1,x_coord,y_coord] = peaks_vector[pixel_index]
                        elif dim_order == 'channels_last':
                            im_array[x_coord,y_coord,0] = pixels_vector[pixel_index]
                            if include_timing:
                                im_array[x_coord,y_coord,1] = peaks_vector[pixel_index]

                    pixel_index += 1

        return im_array


