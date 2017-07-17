import argparse
import math
import sys

from ROOT import TChain,TFile
from glob import glob
import numpy as np
import pickle as pkl

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Count all events in MSCW files.')
    parser.add_argument('mscw_files',help='path of mscw files containing event-level reconstructed parameters for applying cuts/bins')
    args = parser.parse_args()

    #create TChain and add all relevant files
    chain = TChain('data')

    files = glob(args.mscw_files)
    
    for i in files:
        chain.AddFile(i)

    entries = chain.GetEntriesFast()

    print(entries)


