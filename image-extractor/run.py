
class Dataset:

class SuperRun:

    def __init__(self,
                 output_path,
                 ED_cuts_dict=None,
                 storage_mode='tel_type',
                 tel_type_list=['LST','MSTF','MSTN','MSTS','SST1','SSTA','SSTC'],
                 img_mode='1D',
                 img_channels=1,
                 include_timing=True,
                 img_scale_factors={'MSTS':1},
                 img_dtypes={'LST':'float32','MSTS':'float32','MSTF':'float32','MSTN':'float32','SST1':'float32',
                 'SSTA':'float32','SSTC':'float32',},
                 img_dim_order='channels_last',
                 cuts_dict=DEFAULT_CUTS_DICT):




class Run:
    
    def __init__(self,
                 output_path,
                 ED_cuts_dict=None,
                 storage_mode='tel_type',
                 tel_type_list=['LST','MSTF','MSTN','MSTS','SST1','SSTA','SSTC'],
                 img_mode='1D',
                 img_channels=1,
                 include_timing=True,
                 img_scale_factors={'MSTS':1},
                 img_dtypes={'LST':'float32','MSTS':'float32','MSTF':'float32','MSTN':'float32','SST1':'float32',
                 'SSTA':'float32','SSTC':'float32',},
                 img_dim_order='channels_last',
                 cuts_dict=DEFAULT_CUTS_DICT):



