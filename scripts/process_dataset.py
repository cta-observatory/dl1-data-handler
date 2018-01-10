import sys
import os
import logging
import argparse
import subprocess
import imp
import configparser
import warnings
import time

# Dirty hack to avoid
# ImportError: No module named 'image'
# Needs fixing
sys.path.append(imp.find_module("image_extractor")[1])

from image_extractor import image_extractor

DEVNULL = open(os.devnull, 'w')

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser(
    description=("Reduces prod3b simtel datasets and stores calibrated data"
                 "into a formatted HDF5 file."))
parser.add_argument(
    'simtel_list',
    help='text file containing a list of simtel files to process (1 per line)')
parser.add_argument(
    'config',
    help='configuration file')
parser.add_argument(
        "--debug",
        help="print debug/logger messages",
        action="store_true")
parser.add_argument(
        "--keep_merged",
        help="do not delete merged simtel files",
        action="store_true")
parser.add_argument(
        "--out",
        help="set output path")

args = parser.parse_args()

if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

if args.out:
    outpath=args.out
else:
    outpath='.'

logger.info('Reading configuration file')

config = configparser.ConfigParser()
print(args.config)
if os.path.isfile(args.config):
    config.read(args.config)
    hessiosys = config['DEFAULT']['hessiosys']
    merge_map = config['DEFAULT']['merge_map']
    tel_type_list = config['DEFAULT']['tel_type_list'].split(' ')
    srun_size = int(config['DEFAULT']['srun_size'])
    print()
else:
    print('Config file {} not found'.format(args.config))
    sys.exit(1)

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
            run = nonsct_file.split('run')[1].split('___cta')[0]
            if sct_file.split('run')[1].split('___cta')[0] == run:
                matched_list.append([nonsct_file,sct_file,run])

logger.info('{} non-SCT runs in list'.format(len(sct_list)))
logger.info('{} SCT runs in list'.format(len(sct_list)))
logger.info('{} matching runs'.format(len(sct_list)))

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

matched_list = sorted(matched_list, key=lambda x: int(x[2]))
matched_list_split = chunks(matched_list,srun_size)

for srun, matched_sublist in enumerate(matched_list_split):
    ts0 = time.time()
    logger.info('Processing super-run {} of {} ({:.1f}%)'.format(srun+1, len(matched_list),
                                                                100 * (srun + 1) / len(matched_list)))
    if srun_size == 1:
        iefile = matched_sublist[0][0].split('/')[-1].replace("merged", "HB9").replace("simtel.gz", "h5")
        iefile = '{}/{}'.format(outpath,iefile)
    else:
        iefile = matched_sublist[0][0].split('/')[-1].replace("merged", "HB9").replace("simtel.gz", "h5")
        iefile = '{}/{}srun{}-{}___cta{}'.format(outpath, iefile.split('run')[0], matched_sublist[0][2],
                                                 matched_sublist[-1][2],iefile.split('___cta')[-1])
    logger.debug(print('H5 file name: {}'.format(iefile)))

    for idx, matched_pair in enumerate(matched_sublist):
        mergedfile = '{}/{}'.format(outpath, matched_pair[0].split('/')[-1].replace("merged", "HB9"))
        logger.info('Processing run {} pair {}/{} ({:.1f}%)'.format(matched_pair[2], idx+1, len(matched_sublist),
                                                                    100*(idx+1)/len(matched_sublist)))
        mergecom = '{}/merge_simtel {} {} {} {}'.format(hessiosys,merge_map,matched_pair[0],matched_pair[1],mergedfile)
        logger.debug(mergecom)
        tm0 = time.time()
        child = subprocess.run(mergecom.split(' '),stdout=DEVNULL)
        if child.returncode == 0:
            logger.info('Run {} merged - Elapsed time {:.1f}s'.format(matched_pair[2],time.time()-tm0))
            extractor = image_extractor.ImageExtractor(iefile, tel_type_list=tel_type_list)
            te0 = time.time()
            extractor.process_data(mergedfile)
            logger.info('Run {} extracted - Elapsed time {:.1f}s'.format(matched_pair[2], time.time() - te0))
            if not args.keep_merged:
                rmcom = 'rm -f {}'.format(mergedfile)
                child = subprocess.run(rmcom.split(' '))
                if child.returncode == 0:
                    logger.info('Simtel merged file deleted')
        else:
            warnings.warn('Merger of run {} failed!'.format(matched_pair[2]))

    logger.info('Superrun {}-{} extracted - Elapsed time {:.1f}s'.format(matched_sublist[0][2],
                                                                         matched_sublist[-1][2], time.time() - ts0))