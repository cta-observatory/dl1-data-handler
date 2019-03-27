import numpy as np

class ImageMapper():

    def __init__(self, mapping_method, channels):#", settings=defaults, ..."):
        self.mapping_method = mapping_method
        self.channels = channels
        self.mapping_tables = {}
        self.image_shape = {
            'LST_LSTCam': (110, 110, 1),
            'MST_FlashCam': (112, 112, 1),
            'MST_NectarCam': (110, 110, 1),
            'SST1M_DigiCam': (96, 96, 1),
            }

    # TODO: Replace this placeholder method with valid image mapping methods
    def map_image(self, vector, tel_type):
        # Fill an array of the right size with copies of the vector
        return np.resize(vector, self.image_shape[tel_type])
