import argparse
import math
import sys

from ROOT import TChain,TFile
from glob import glob
import numpy as np
import pickle as pkl

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load mscw files to get event-level parameters and apply desired bins/cuts. Then writes dictionary mapping (runNumber,eventNumber) to (energy bin, reconstructed energy).')
    parser.add_argument('mscw_files',help='path of mscw files containing event-level reconstructed parameters for applying cuts/bins')
    args = parser.parse_args()

    #energy bins
    ENERGY_BINS = [(0.1,0.31),(0.31,1),(1,10)]
    NUM_ENERGY_BINS = len(ENERGY_BINS)

    #cuts

    NO_MAX_FLOAT = float('inf')
    NO_MAX_INT = sys.maxsize

    MSCW_CUT = (-2.0,2.0)
    MSCL_CUT = (-2.0,5.0)
    EChi2S_CUT = (0.0,NO_MAX_FLOAT)
    ErecS_CUT = (0.0,NO_MAX_FLOAT)
    EmissionHeight_CUT = (0.0,50.0)
    Offset_CUT = (0.0,3.0)
    NImages_CUT = (3,NO_MAX_INT)
    dES_CUT = (0.0,NO_MAX_FLOAT)

    CUT_VALUES = [MSCW_CUT,MSCL_CUT,EChi2S_CUT,ErecS_CUT,EmissionHeight_CUT,Offset_CUT,NImages_CUT,dES_CUT]

    #create TChain and add all relevant files
    chain = TChain('data')

    files = glob(args.mscw_files)
    
    for i in files:
        chain.AddFile(i)

    entries = chain.GetEntriesFast()

    print('Applying bins/cuts')

    #dict storing information on which events are in which bins and the reconstructed energy for each event
    event_bin_Erec_dict = { }

    for entry in chain:
        CUT_VARIABLES = [entry.MSCW, entry.MSCL, entry.EChi2S, entry.ErecS, entry.EmissionHeight, math.sqrt(entry.MCxoff**2 + entry.MCyoff**2), entry.NImages, entry.dES]
        assert len(CUT_VALUES) == len(CUT_VARIABLES)
        CUTS= []
        for i in range(len(CUT_VALUES)):
            CUTS.append(CUT_VARIABLES[i]>= CUT_VALUES[i][0] and CUT_VARIABLES[i] < CUT_VALUES[i][1])

        if all(CUTS):
            for j in range(NUM_ENERGY_BINS):
                if entry.ErecS>=ENERGY_BINS[j][0] and entry.ErecS<ENERGY_BINS[j][1]:
                    event_bin_Erec_dict[(entry.runNumber,entry.eventNumber)] = (j,entry.ErecS)
                    break

    print('Writing to .pkl file')

    pkl.dump(event_bin_Erec_dict, open( "bins_cuts_dict.pkl", "wb" ) )

    print('Done!')

