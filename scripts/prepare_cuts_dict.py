import argparse
import math
import sys

from ROOT import TChain, TFile
from glob import glob
import numpy as np
import pickle as pkl

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=("Load mscw files to get event-level parameters and "
                     "apply desired bins/cuts. Then writes dictionary "
                     "mapping (runNumber,eventNumber) to "
                     "(energy bin, reconstructed energy)."))
    parser.add_argument('mscw_files',
                        help=("path of mscw files containing event-level"
                              "reconstructed parameters for applying"
                              "cuts/bins"))
    # parser.add_argument('config_file',help='configuration file')
    parser.add_argument('output_filename',
                        help='name of output .pkl dictionary file')
    args = parser.parse_args()

    # energy bins
    ENERGY_BINS = [(0.1, 0.31), (0.31, 1), (1, 10)]

    # cuts
    MAX_FLOAT = float('inf')
    MAX_INT = sys.maxsize

    CUT_PARAMETERS = [('MSCW', (-2.0, 2.0)),
                      ('MSCL', (-2.0, 5.0)),
                      ('EChi2S', (0.0, MAX_FLOAT)),
                      ('ErecS', (0.0, MAX_FLOAT)),
                      ('EmissionHeight', (0.0, 50.0)),
                      (('MCxoff', 'MCyoff'), (0.0, 3.0)),
                      ('NImages', (3, MAX_INT)),
                      ('dES', (0.0, MAX_FLOAT))]

    # create TChain and add all relevant files
    chain = TChain('data')
    files = glob(args.mscw_files)
    for i in files:
        chain.AddFile(i)

    entries = chain.GetEntriesFast()
    passed_cuts_count = 0

    print('Applying bins/cuts')

    # dict storing information on which events are in which bins and the
    # reconstructed energy for each event
    event_bin_Erec_dict = {}

    for entry in chain:
        CUTS = []
        for i in CUT_PARAMETERS:
            if not isinstance(i[0], tuple):
                CUTS.append(
                    getattr(entry, i[0]) >= i[1][0] and
                    getattr(entry, i[0]) < i[1][1])
            else:
                CUTS.append(
                    math.sqrt(getattr(entry, i[0][0])**2 +
                              getattr(entry, i[0][1])**2) >= i[1][0] and
                    math.sqrt(getattr(entry, i[0][0])**2 +
                              getattr(entry, i[0][1])**2) < i[1][1])

        if all(CUTS):
            for j in range(len(ENERGY_BINS)):
                if entry.ErecS >= ENERGY_BINS[j][0] \
                        and entry.ErecS < ENERGY_BINS[j][1]:
                    event_bin_Erec_dict[(
                                        entry.runNumber,
                                        entry.eventNumber)] = (j,
                                                               entry.ErecS)
                    passed_cuts_count += 1
                    break

    print('Total number of events: ', entries)
    print('Events passing cuts: ', passed_cuts_count)

    print('Writing to .pkl file...')

    pkl.dump(event_bin_Erec_dict, open(args.output_filename, "wb"))

    print('Done!')
