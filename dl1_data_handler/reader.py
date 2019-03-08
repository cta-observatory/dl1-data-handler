from collections import OrderedDict
import random
import threading

import numpy as np
import tables

from dl1_data_handler.image_mapper import ImageMapper
from dl1_data_handler.processor import DL1DataProcessor

class DL1DataReader:

    @staticmethod
    def __synchronized_open_file(*args, **kwargs):
        with threading.Lock() as lock:
            return tables.open_file(*args, **kwargs)

    @staticmethod
    def __synchronized_close_file(*args, **kwargs):
        with threading.Lock() as lock:
            return self.close(*args, **kwargs)

    def __init__(self,
                 file_list,
                 mode='mono',
                 selected_telescope_type=None,
                 selected_telescope_ids=None,
                 selection_string=None,
                 intensity_selection=None,
                 shuffle=False,
                 seed=None,
                 image_channels=None,
                 mapping_method=None,
                 mapping_settings=None,
                 array_info=None,
                 event_info=None,
                 transforms=None
                ):

        # Construct dict of filename:file_handle pairs
        self.files = OrderedDict()
        for filename in file_list:
            self.files[filename] = \
                self.__synchronized_open_file(filename, mode='r')

        # Set data loading mode
        # Mono: single images of one telescope type
        # Stereo: events of one telescope type
        # Multi-stereo: events including multiple telescope types
        if mode in ['mono', 'stereo', 'multi-stereo']:
            self.mode = mode
        else:
            raise ValueError("Invalid mode selection '{}'. Valid options: "
                             "'mono', 'stereo', 'multi-stereo'".format(mode))

        self.example_identifiers = None
        self.telescopes = {}
        if selected_telescope_ids is None:
            selected_telescope_ids = {}

        if intensity_selection is None:
            intensity_selection = {}

        if mapping_settings is None:
            mapping_settings = {}

        # Loop over the files to assemble the selected event identifiers
        for filename, f in self.files.items():
            example_identifiers = []

            # Get dict of all the tel_types in the file mapped to their tel_ids
            telescopes = {}
            for row in f.root.Array_Information:
                tel_type = row['type'].decode()
                if tel_type not in telescopes:
                    telescopes[tel_type] = []
                telescopes[tel_type].append(row['id'])

            # Enforce an automatic minimal telescope selection cut:
            # there must be at least one triggered telescope of a
            # selected type in the event
            # Users can include stricter cuts in the selection string
            if self.mode in ['mono', 'stereo']:
                if selected_telescope_type is None:
                    # Default: use the first tel type in the file
                    default = f.root.Array_Information[0]['type'].decode()
                    selected_telescope_type = default
                self.tel_type = selected_telescope_type
                selected_tel_types = [selected_telescope_type]
            elif self.mode == 'multi-stereo':
                if selected_telescope_type is None:
                    # Default: use all tel types
                    selected_telescope_type = list(telescopes)
                self.tel_type = None
                selected_tel_types = selected_telescope_type
            multiplicity_conditions = ['(' + tel_type + '_multiplicity > 0)'
                                       for tel_type in selected_tel_types]
            tel_cut_string = '(' + ' | '.join(multiplicity_conditions) + ')'
            # Combine minimal telescope cut with explicit selection cuts
            if selection_string:
                cut_condition = selection_string + ' & ' + tel_cut_string
            else:
                cut_condition = tel_cut_string

            # Select which telescopes from the full dataset to include in each
            # event by a telescope type and an optional list of telescope ids.
            selected_telescopes = {}
            for tel_type in selected_tel_types:
                available_tel_ids = telescopes[tel_type]
                # Keep only the selected tel ids for the tel type
                if tel_type in selected_telescope_ids:
                    # Check all requested telescopes are available to select
                    requested_tel_ids = selected_telescope_ids[tel_type]
                    invalid_tel_ids = list(set(requested_tel_ids) -
                                           set(available_tel_ids))
                    if invalid_tel_ids:
                        raise ValueError("Tel ids {} are not a valid selection"
                                         "for tel type '{}'".format(
                                             invalid_tel_ids, tel_type))
                    selected_telescopes[tel_type] = requested_tel_ids
                else:
                    selected_telescopes[tel_type] = available_tel_ids

            selected_nrows = [row.nrow for row
                              in f.root.Events.where(cut_condition)
                              if self._select_event_intensity(
                                  row, intensity_selection)]

            # Make list of identifiers of all examples passing event selection
            if mode in ['stereo', 'multi-stereo']:
                example_identifiers = [(filename, nrow) for nrow
                                       in selected_nrows]
            elif mode == 'mono':
                example_identifiers = []
                field = '{}_indices'.format(self.tel_type)
                for indices in f.root.Events.read_coordinates(
                        selected_nrows, field=field):
                    for tel_id, index in zip(telescopes[self.tel_type],
                                             indices):
                        if (tel_id in selected_telescopes[self.tel_type]
                                and index != 0
                                and self._select_image_intensity(
                                    f.root._f_get_child(
                                        self.tel_type)[index]['charge'],
                                    intensity_selection)):
                            example_identifiers.append(
                                (filename, index, tel_id))

            # Confirm that the files are consistent and merge them
            if not self.telescopes:
                self.telescopes = telescopes
            if self.telescopes != telescopes:
                raise ValueError("Inconsistent telescope definition in "
                                 "{}".format(filename))
            self.selected_telescopes = selected_telescopes

            if self.example_identifiers is None:
                self.example_identifiers = example_identifiers
            else:
                self.example_identifiers.extend(example_identifiers)

        # Shuffle the examples
        if shuffle:
            random.seed(seed)
            random.shuffle(self.example_identifiers)

        if image_channels is None:
            image_channels = ['charge']
        self.image_channels = image_channels
        self.image_mapper = ImageMapper(mapping_method, image_channels,
                                        **mapping_settings)

        if array_info is None:
            array_info = []
        self.array_info = array_info

        if event_info is None:
            event_info = []
        self.event_info = event_info

        # Construct example description (before preprocessing)
        if self.mode == 'mono':
            self.unprocessed_example_description = [
                {
                    'name': 'image',
                    'tel_type': self.tel_type,
                    'base_name': None,
                    'shape': self.image_mapper.image_shape[self.tel_type],
                    'dtype': np.dtype(np.float32)
                    }
                ]
            for col_name in array_info:
                col = f.root.Array_Information._f_col(col_name)
                self.unprocessed_example_description.append(
                    {
                        'name': col_name,
                        'tel_type': self.tel_type,
                        'base_name': None,
                        'shape': col.shape,
                        'dtype': col.dtype
                        }
                    )
        elif self.mode == 'stereo':
            num_tels = len(self.selected_telescopes[self.tel_type])
            self.unprocessed_example_description = [
                {
                    'name': 'image',
                    'tel_type': self.tel_type,
                    'base_name': None,
                    'shape': ((num_tels,)
                              + self.image_mapper.image_shape[self.tel_type]),
                    'dtype': np.dtype(np.float32)
                    },
                {
                    'name': 'trigger',
                    'tel_type': self.tel_type,
                    'base_name': None,
                    'shape': (num_tels, 1),
                    'dtype': np.dtype(np.int8)
                    }
                ]
            for col_name in array_info:
                col = f.root.Array_Information._f_col(col_name)
                self.unprocessed_example_description.append(
                    {
                        'name': col_name,
                        'tel_type': self.tel_type,
                        'base_name': None,
                        'shape': (num_tels,) + col.shape,
                        'dtype': col.dtype
                        }
                    )
        elif self.mode == 'multi-stereo':
            self.unprocessed_example_description = []
            for tel_type in self.selected_telescopes:
                num_tels = len(self.selected_telescopes[tel_type])
                self.unprocessed_example_description.extend([
                    {
                        'name': tel_type + '_image',
                        'tel_type': tel_type,
                        'base_name': 'image',
                        'shape': ((num_tels,)
                                  + self.image_mapper.image_shape[tel_type]),
                        'dtype': np.dtype(np.float32)
                        },
                    {
                        'name': tel_type + '_trigger',
                        'tel_type': tel_type,
                        'base_name': 'trigger',
                        'shape': (num_tels, 1),
                        'dtype': np.dtype(np.int8)
                        }
                    ])
                for col_name in array_info:
                    col = f.root.Array_Information._f_col(col_name)
                    self.unprocessed_example_description.append(
                        {
                            'name': tel_type + '_' + col_name,
                            'tel_type': tel_type,
                            'base_name': col_name,
                            'shape': (num_tels,) + col.shape,
                            'dtype': col.dtype
                            }
                        )
        # Add event info to description
        for col in self.event_info:
            self.unprocessed_example_description.append(
                {
                    'name': col,
                    'tel_type': None,
                    'base_name': None,
                    'shape': f.root.Events._f_col(col).shape,
                    'dtype': f.root.Events._f_col(col).dtype
                    }
                )

        self.processor = DL1DataProcessor(
            self.mode,
            self.unprocessed_example_description,
            transforms
            )

        # Definition of preprocessed example
        self.example_description = self.processor.output_description

    def _select_event_intensity(self, row, intensity_selection):
        if intensity_selection is None or self.mode == 'mono':
            return True
        elif self.mode in ['stereo', 'multi-stereo']:
            # TODO: define event-wise intensity
            return True

    @staticmethod
    def _select_image_intensity(image_charge, intensity_selection):
        intensity = np.sum(image_charge)
        lower = intensity_selection.get('lower', -np.inf)
        upper = intensity_selection.get('upper', np.inf)
        return (lower < intensity < upper)

    # Get a single telescope image from a particular event, uniquely
    # identified by the filename, tel_type, and image table index.
    # First extract a raw 1D vector and transform it into a 2D image using a
    # mapping table.
    def _get_image(self, filename, tel_type, image_index):

        f = self.files[filename]
        record = f.root._f_get_child(tel_type)[image_index]
        query = "type == '{}'".format(tel_type)
        length = [x['num_pixels'] for x
                  in f.root.Telescope_Type_Information.where(query)][0]
        num_channels = len(self.image_channels)
        vector = np.empty(shape=(length + 1, num_channels), dtype=np.float32)
        # An "empty" pixel at index 0 is used to fill blank areas in image
        vector[0, :] = 0.0
        # If the telescope didn't trigger, the image index is 0 and a blank
        # image of all zeros with be loaded
        for i, channel in enumerate(self.image_channels):
            vector[1:, i] = record[channel]
        image = self.image_mapper.map_image(vector, tel_type)

        return image

    def __len__(self):
        return len(self.example_identifiers)

    def __getitem__(self, idx):

        identifiers = self.example_identifiers[idx]

        # Get record for the event
        filename = identifiers[0]
        f = self.files[filename]

        def append_array_info(array_info, tel_id):
            query = "id == {}".format(tel_id)
            row = [row for row in f.root.Array_Information.where(query)][0]
            for info, column in zip(array_info, self.array_info):
                info.append(row[column])

        def load_tel_type_data(nrow, tel_type):
            images = []
            triggers = []
            array_info = [[] for column in self.array_info]
            for tel_id in self.selected_telescopes[tel_type]:
                tel_index = self.telescopes[tel_type].index(tel_id)
                image_index = f.root.Events[nrow][
                    tel_type + '_indices'][tel_index]
                image = self._get_image(filename, tel_type, image_index)
                trigger = 0 if image_index == 0 else 1
                images.append(image)
                triggers.append(trigger)
                append_array_info(array_info, tel_id)
            example = [np.stack(images), np.array(triggers, dtype=np.int8)]
            example.extend([np.stack(info) for info in array_info])
            return example

        # Load the data and any selected array info
        if self.mode == "mono":
            # Get a single image
            image_index, tel_id = identifiers[1:3]
            nrow = f.root._f_get_child(self.tel_type)[image_index]['event_index']

            image = self._get_image(filename, self.tel_type, image_index)
            example = [image]

            array_info = [[] for column in self.array_info]
            append_array_info(array_info, tel_id)
            example.extend([info for info in array_info])
        elif self.mode == "stereo":
            # Get a list of images and an array of binary trigger values
            nrow = identifiers[1]
            example = load_tel_type_data(nrow, self.tel_type)
        elif self.mode == "multi-stereo":
            # Get a list of images and an array of binary trigger values
            # for each selected telescope type
            nrow = identifiers[1]
            example = []
            for tel_type in self.selected_telescopes:
                tel_type_example = load_tel_type_data(nrow, tel_type)
                example.extend(tel_type_example)

        # Load event info
        record = f.root.Events[nrow]
        for column in self.event_info:
            example.append(record[column])

        # Preprocess the example
        example = self.processor.process(example)

        return example

    # Return a dictionary of number of examples in the dataset, grouped by
    # the array names listed in the iterable group_by.
    def num_examples(self, group_by=None):
        example_indices = []
        if group_by:
            for name in group_by:
                index = [i for i, des in enumerate(self.example_description)
                         if des['name'] == name][0]
                example_indices.append(index)
        num_examples = {}
        for example in self:
            group = tuple([example[index] for index in example_indices])
            if group in num_examples:
                num_examples[group] += 1
            else:
                num_examples[group] = 1
        return num_examples
