import argparse
import os
import logging

from configobj import ConfigObj
from validate import Validator

config_spec = """
mode = option('tel_id','tel_type', default='tel_type')
[image]
mode = option('vector','array',default='array')
format = option('charge_only','charge_timing','charge_timing_padding',default='charge_only')
array_scale_factor = integer(min=1,default=2)
dtype = option('uint32', 'int16', 'int32', 'uint16',default='uint16')
array_dim_order = option('channels_first','channels_last',default='channels_last')
[telescope]
type_list = list(default=list('SCT'))
[energy_bins]
apply_energy_bins = boolean(default=False)
units = option('eV','MeV','GeV','TeV',default='TeV')
scale = option('linear','log10',default='log10')
min = float(default=-1.0)
max = float(default=1.0)
bin_size = float(default=0.5)
[preselection_cuts]
mode = option('pkl_dict','ctapipe','none',default='pkl_dict')
MSCW = tuple(default=list(-2.0,2.0))
MSCL = tuple(default=list(-2.0,5.0))
EChi2S = tuple(default=list(0.0,None))
ErecS = tuple(default=list(0.0,None))
EmissionHeight = tuple(default=list(0.0,50.0))
MC_Offset = mixed_list(string,string,float,float,default=list('MCxoff','MCyoff',0.0,3.0))
NImages = tuple(default=list(3,None))
dES = tuple(default=list(0.0,None))
[energy_reconstruction]
    include_label = boolean(default=False)
    gamma_only = boolean(default=True)
    [[bins]]
    units = option('eV','MeV','GeV','TeV',default='TeV')
    scale = option('linear','log10',default='log10')
    min = float(default=-2.0)
    max = float(default=2.0)
    bin_size = float(default=0.05)
[preprocessing]
shuffle = boolean(default=True)
    [[split]]
    apply_split = boolean(default=True)
    train = float(default=0.9)
    val = float(default=0.1)
    test = float(default=0.1)

"""

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate/validate config files')
    parser.add_argument('file', help='path/name of file to generate/validate')
    parser.add_argument("--mode", help='validate (val), generate config file (config), or generate configspec file (spec)', default='config')
    args = parser.parse_args()

    if args.mode == 'config':
        spec = s.split("\n")
        config = ConfigObj(args.file, configspec=spec)
        validator = Validator()
        config.validate(validator, copy=True)
        config.filename = args.file
        config.write()

    elif args.mode == 'val':
        spec = s.split('\n')
        config = ConfigObj(args.file, configspec=spec)
        validator = Validator()
        assert config.validate(validator, copy=True), "Validation failed on file: {}".format(args.file)

    elif args.mode == 'spec':
        with open(args.file, 'wb') as f:
            write(s)

    else:
        logger.error('Invalid command line mode: {}. Valid options: config, val, spec.'.format(args.mode))
        raise ValueError('Invalid command line mode: {}.'.format(args.mode))
