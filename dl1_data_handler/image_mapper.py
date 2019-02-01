def get_camera_type(tel_type):
    return tel_type.split(':')[1]

class ImageMapper():
    
    def __init__(mapping_method, channels", settings=defaults, ..."):
        self.mapping_method = mapping_method
        self.channels = channels
        self.mapping_tables = {}
   
    """Returns: WHC np.float32 array"""
    def map_image(1D_image, tel_type):
        camera_type = get_camera_type(tel_type)
        raise NotImplementedError


