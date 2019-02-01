def get_camera_type(tel_type):
    return tel_type.split(':')[1]

class ImageMapper():
    
    def __init__(self, mapping_method, channels):#", settings=defaults, ..."):
        self.mapping_method = mapping_method
        self.channels = channels
        self.mapping_tables = {}
        self.image_shape = {
                'LST_LSTCam': (108, 108, 1),
                'MST_FlashCam': (110, 110, 1),
                'SST1M_DigiCam': (94, 94, 1),
                }
   
    """Returns: image (WHC np.float32 array)"""
    def map_image(self, vector, tel_type):
        camera_type = get_camera_type(tel_type)
        raise NotImplementedError


