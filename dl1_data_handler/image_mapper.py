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
   
    """Returns: image (WHC np.float32 array)"""
    def map_image(self, vector, tel_type):
        return np.resize(vector, self.image_shape[tel_type])
