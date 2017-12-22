import numpy as np
import logging
import argparse
#import image_extractor

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser(
    description=("Reduces prod3b simtel datasets and stores calibrated data into "
                 "into a formatted HDF5 file."))
parser.add_argument(
    'simtel_list',
    help='text file containing a list of simtel files to process (1 per line)')
parser.add_argument(
        "--debug",
        help="print debug/logger messages",
        action="store_true")

args = parser.parse_args()

if args.debug:
    logger.setLevel(logging.DEBUG)

logger.info('Reading list and finding matched runs')

with open(args.simtel_list,'r') as f:
    nonsct_list=[]
    sct_list=[]
    matched_list=[]
    for line in f:
        if 'SCT' in line:
            sct_list.append(line.rstrip())
        else:
            nonsct_list.append(line.rstrip())
    for sct_file in sct_list:
        for nonsct_file in nonsct_list:
            if sct_file.split('run')[1].split('cta')[0] == nonsct_file.split('run')[1].split('cta')[0]:
                matched_list.append([nonsct_file,sct_file])

logger.info('{} non-SCT runs in list'.format(len(sct_list)))
logger.info('{} SCT runs in list'.format(len(sct_list)))
logger.info('{} matching runs'.format(len(sct_list)))

logger.info('Merging matching pairs of runs')

hessiosys = '/home/nieto/Software/hessioxxx/bin'
merge_cfg = 'my.cfg'

for matched_pair in matched_list:
    mergecom = '{}/merge_simtel {} {} {}'.format(hessiosys,matched_pair[0],matched_pair[1],merge_cfg)
    print(mergecom)

            #max events to read per file
#max_events = 10

#data_file = "/data2/deeplearning/test/proton_20deg_180deg_run101___cta-prod3_desert-2150m-Paranal-HB9-all.simtel.gz"
#output_path = "/data2/deeplearning/test/dummy.h5"
#create an ImageExtractor with default settings
#extractor = image_extractor.ImageExtractor(output_path,tel_type_list=['MSTS'])

#extractor.process_data(data_file,max_events)