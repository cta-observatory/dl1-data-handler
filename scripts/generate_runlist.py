"""Load data from ctapipe EventSources and dump to file."""
import yaml
import argparse
import os
import glob
import re


# gamma_20deg_0deg_run26640___cta-prod3-lapalma-2147m-LaPalma-SCT_cone10.simtel.gz

def parse_filename(filename):
    """Extract run parameters from a simtel.gz filename.

    Parameters
    ----------
    filename : list
        A filename formatted as [particle_type]_[ze]deg_[az]deg_run[run_number]
        ___[production info].simtel.gz

    Returns
    -------
    int
        Run/observation number
    str
        Type of the event primary particle (i.e. gamma, proton)
    int
        Run zenith angle in degrees
    int
        Run azimuth angle in degrees
    str
        A string containing other production-related descriptors.
    str
        N/A.

    """
    basename = os.path.basename(filename).replace('.simtel.gz', '')
    foutputs = re.split(
        '___', basename)
    prod_info = foutputs[1]
    soutputs = re.split(
        '_', foutputs[0])
    particle_type, ze, az, run_number = soutputs
    identifiers = [particle_type, ze, az, prod_info]
    
    identifiers[1] = int(re.sub('deg$', '', identifiers[1]))
    identifiers[2] = int(re.sub('deg$', '', identifiers[2]))
    run_number = int(re.sub('^run', '', run_number))

    return run_number, identifiers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Generate a runlist automatically from simtel.gz \
                     files following the naming convention \
                     [particle_type]_[ze]deg_[az]deg_run[run_number]___ \
                     [production info].simtel.gz"))
    parser.add_argument('file_dir',
                        help=("path to directory containing all simtel files"))
    parser.add_argument('--num_inputs_per_run',
                        help='number of input files between each run',
                        default=10)
    parser.add_argument('--output_file',
                        help='filepath/name of runlist file',
                        default="./runlist.yml")
    args = parser.parse_args()

    runlist = []

    filenames = glob.glob(os.path.join(args.file_dir, "*.simtel.gz"))
    filename_groups = {}

    # Separate files by particle type, ze, az, processing info, cone
    for filename in filenames:
        identifiers = tuple(parse_filename(filename)[1])
        if identifiers not in filename_groups:
            filename_groups[identifiers] = []
        filename_groups[identifiers].append(filename)

    for key, list in filename_groups.items():
        # Within each group, sort files by run number
        list.sort(key=lambda x: parse_filename(x)[0])

        inputs = []
        start_run_number = 0
        for i, filename in enumerate(list):
            run_number = parse_filename(filename)[0]
            if len(inputs) == 0:
                start_run_number = run_number

            inputs.append(filename)
            if len(inputs) >= int(args.num_inputs_per_run) or i == (
                    len(list) - 1):
                particle_type, ze, az, prod_info = key

                target_filename = "{}_{}deg_{}deg_runs{}___{}.h5".format(
                    particle_type,
                    ze,
                    az,
                    '{}-{}'.format(start_run_number, run_number),
                    prod_info)

                runlist.append({
                    'inputs': inputs,
                    'target': target_filename
                })

                inputs = []

    # dump to file
    stream = open(args.output_file, 'w')
    yaml.dump(runlist, stream)
