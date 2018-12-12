from tables import *
import argparse
from PIL import Image, ImageOps
import random
import numpy as np


def normalize(image_array):

    max_value = np.amax(image_array)

    if max_value == 0:
        return image_array

    else:
        image_array_normalized = np.multiply((255.0/max_value), image_array)
        return image_array_normalized

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Visualize a particular telescope image from a '
                     'given event number (and run number). Seperately shows '
                     'visualization for the charge image '
                     'and the timing image.'))
    parser.add_argument('data_file', help='path to input hdf5 file')
    parser.add_argument('bin_number')
    parser.add_argument('--event_number')
    parser.add_argument('--run_number')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--save_dir', default='./visualized_images')
    # parser.add_argument('tel_id')
    args = parser.parse_args()

    f = open_file(args.data_file, mode="r", title="Data file")

    row_str = 'f.root.E{}.Events'.format(int(args.bin_number))
    table = eval(row_str)

    tel_ids = ["T" + str(i) for i in range(5, 20)]

    if args.random:
        # select random run_number, event_number combination
        unique_events = [(x['run_number'], x['event_number'])
                         for x in table.iterrows()]
        run_num_selected, event_num_selected = random.choice(unique_events)

    else:
        run_num_selected = int(args.run_number)
        event_num_selected = int(args.event_number)

    result = [row[j]
              for j in tel_ids
              for row in table.where('(event_number == event_num_selected) &
                                      (run_number == run_num_selected)')]

    for i in [result[0]]:
        full_image_array = np.array(i)
        # print(full_image_array.shape)
        # full_image = Image.fromarray(full_image_array,'RGB')
        charge_image_array = full_image_array[:, :, 0].astype(np.uint32)
        # charge_image_array = normalize(charge_image_array)
        # print(charge_image_array.shape)
        charge_image = Image.fromarray(charge_image_array, 'I')
        timing_image_array = full_image_array[:, :, 1].astype(np.uint32)
        # timing_image_array = normalize(timing_image_array)
        # print(timing_image_array.shape)
        timing_image = Image.fromarray(timing_image_array, 'I')

        # full_image.save("test_full.png")
        charge_image.save("test_charge.png")
        timing_image.save("test_timing.png")
