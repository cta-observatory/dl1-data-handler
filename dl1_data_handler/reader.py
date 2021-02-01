from collections import Counter, OrderedDict
import random
import threading
import numpy as np
import tables

from dl1_data_handler.image_mapper import ImageMapper
from dl1_data_handler.processor import DL1DataProcessor

from pyirf.simulations import SimulatedEventsInfo
import astropy.units as u
from astropy import table

__all__ = [
    'DL1DataReader',
    'DL1DataReaderSTAGE1',
    'DL1DataReaderDL1DH'
]


lock = threading.Lock()


class DL1DataReader:

    def __init__(self,
                 file_list,
                 mode='mono',
                 subarray_info=None,
                 event_info=None
                ):
        
        # Construct dict of filename:file_handle pairs
        self.files = OrderedDict()
        for filename in file_list:
            with lock:
                self.files[filename] = tables.open_file(filename, mode='r')
                
        # Set data loading mode
        # Mono: single images of one telescope type
        # Stereo: events including multiple telescope types
        if mode in ['mono', 'stereo']:
            self.mode = mode
        else:
            raise ValueError("Invalid mode selection '{}'. Valid options: "
                             "'mono', 'stereo'".format(mode))
      
        if subarray_info is None:
            subarray_info = []
        self.subarray_info = subarray_info

        if event_info is None:
            event_info = []
        self.event_info = event_info
        
    def _get_camera_type(self, tel_type):
        return tel_type.split('_')[-1]
    
    def __len__(self):
        return len(self.example_identifiers)
    
    # Return a dictionary of number of examples in the dataset, grouped by
    # the array names listed in the iterable group_by.
    # If example_indices is a list of indices, consider only those examples,
    # otherwise all examples in the reader are considered.
    def num_examples(self, group_by=None, example_indices=None):
        grouping_indices = []
        if group_by is not None:
            for name in group_by:
                for idx, des in enumerate(self.example_description):
                    if des['name'] == name:
                        grouping_indices.append(idx)
        group_nums = {}
        if example_indices is None:
            example_indices = list(range(len(self)))
        for idx in example_indices:
            example = self[idx]
            # Use tuple() and tolist() to convert list and NumPy array
            # to hashable keys
            group = tuple([example[idx].tolist() for idx in grouping_indices])
            if group in group_nums:
                group_nums[group] += 1
            else:
                group_nums[group] = 1
        return group_nums
        
        
    def _construct_unprocessed_example_description(self, subarray_table, events_table):
        """
        Construct example description (before preprocessing).
        
        Parameters
        ----------
            subarray_table (tables.Table): the table containing the subarray information
            events_table (tables.Table): the table containing the simulated events information
        """
        if self.mode == 'mono':
            self.unprocessed_example_description = []
            if self.image_channels is not None:
                self.unprocessed_example_description.append(
                    {
                        'name': 'image',
                        'tel_type': self.tel_type,
                        'base_name': 'image',
                        'shape': self.image_mapper.image_shapes[self._get_camera_type(self.tel_type)],
                        'dtype': np.dtype(np.float32)
                    }
                )
            if self.parameter_list is not None:
                self.unprocessed_example_description.append(
                    {
                        'name': 'parameters',
                        'tel_type': self.tel_type,
                        'base_name': 'parameters',
                        'shape': len(self.parameter_list),
                        'dtype': np.dtype(np.float32)
                    }
                )
            for col_name in self.subarray_info:
                col = subarray_table.cols._f_col(col_name)
                self.unprocessed_example_description.append(
                    {
                        'name': col_name,
                        'tel_type': self.tel_type,
                        'base_name': col_name,
                        'shape': (1,) + col.shape[1:],
                        'dtype': col.dtype
                    }
                )

        elif self.mode == 'stereo':
            self.unprocessed_example_description = []
            for tel_type in self.selected_telescopes:
                num_tels = len(self.selected_telescopes[tel_type])
                self.unprocessed_example_description.extend([
                    {
                        'name': tel_type + '_trigger',
                        'tel_type': tel_type,
                        'base_name': 'trigger',
                        'shape': (num_tels,),
                        'dtype': np.dtype(np.int8)
                    }
                ])

                if self.image_channels is not None:
                    self.unprocessed_example_description.extend([
                        {
                            'name': tel_type + '_image',
                            'tel_type': tel_type,
                            'base_name': 'image',
                            'shape': ((num_tels,)
                                    + self.image_mapper.image_shapes[self._get_camera_type(tel_type)]),
                            'dtype': np.dtype(np.float32)
                        }
                    ])
                if self.parameter_list is not None:
                    self.unprocessed_example_description.extend([
                        {
                            'name': tel_type + '_parameters',
                            'tel_type': tel_type,
                            'base_name': 'parameters',
                            'shape': ((num_tels,)
                                    + (len(self.parameter_list),)),
                            'dtype': np.dtype(np.float32)
                        }
                    ])
                
                for col_name in self.subarray_info:
                    col = subarray_table.cols._f_col(col_name)
                    self.unprocessed_example_description.append(
                        {
                            'name': tel_type + '_' + col_name,
                            'tel_type': tel_type,
                            'base_name': col_name,
                            'shape': (num_tels,) + col.shape[1:],
                            'dtype': col.dtype
                        }
                    )

        # Add parameters info to description
        # working only with mono at the moment
        if self.mode == 'mono':
            for col_name in self.training_parameters:
                col = parameters_table.cols._f_col(col_name)
                self.unprocessed_example_description.append(
                    {
                        'name': 'parameter_' + str(col_name),
                        'tel_type': self.tel_type,
                        'base_name': col_name,
                        'shape': col.shape[1:],
                        'dtype': col.dtype
                    }
                )

        # Add event info to description
        for col_name in self.event_info:
            col = events_table.cols._f_col(col_name)
            self.unprocessed_example_description.append(
                {
                    'name': col_name,
                    'tel_type': None,
                    'base_name': col_name,
                    'shape': col.shape[1:],
                    'dtype': col.dtype
                    }
                )
        return

    # Get a single telescope image from a particular event, uniquely
    # identified by the filename, tel_type, and image table index.
    # First extract a raw 1D vector and transform it into a 2D image using a
    # mapping table. When 'indexed_conv' is selected this function should
    # return the unmapped vector.
    def _get_image(self, child, tel_type, image_index):

        num_pixels = self.num_pixels[self._get_camera_type(tel_type)]
        num_channels = len(self.image_channels)
        vector = np.zeros(shape=(num_pixels, num_channels), dtype=np.float32)
        # If the telescope didn't trigger, the image index is -1 and a blank
        # image of all zeros with be loaded
        if image_index != -1:
            with lock:
                record = child[image_index]
                for i, channel in enumerate(self.image_channels):
                    if channel == 'image_mask':
                        image_mask = record[channel]
                        vector[:, i] = record['image'] * image_mask
                    else:
                        vector[:, i] = record[channel]
        
        # If 'indexed_conv' is selected, we only need the unmapped vector.
        if self.image_mapper.mapping_method[self._get_camera_type(tel_type)] == 'indexed_conv':
           return vector
        image = self.image_mapper.map_image(vector, self._get_camera_type(tel_type))
        return image
    
    def _append_subarray_info(self, subarray_table, subarray_info, query):
        with lock:
            for row in subarray_table.where(query):
                for info, column in zip(subarray_info, self.subarray_info):
                    dtype = subarray_table.cols._f_col(column).dtype
                    info.append(np.array(row[column], dtype=dtype))
        return
    
    
# CTA DL1 data model v1.0.2
# ctapipe stage1 v0.10.1 (standard settings writing images and parameters)
class DL1DataReaderSTAGE1(DL1DataReader):

    def __init__(self,
                 file_list,
                 mode='mono',
                 selected_telescope_types=None,
                 selected_telescope_ids=None,
                 multiplicity_selection=None,
                 event_selection=None,
                 parameter_selection=None,
                 shuffle=False,
                 seed=None,
                 image_channels=None,
                 parameter_list=None,
                 mapping_settings=None,
                 subarray_info=None,
                 event_info=None,
                 transforms=None,
                 validate_processor=False
                ):
        
        super().__init__(file_list=file_list, mode=mode, subarray_info=subarray_info, event_info=event_info)

        self.example_identifiers = None
        self.telescopes = {}
        if selected_telescope_ids is None:
            selected_telescope_ids = {}

        if multiplicity_selection is None:
            multiplicity_selection = {}

        if mapping_settings is None:
            mapping_settings = {}

        simulation_info = None
        for filename, f in self.files.items():

            # Read simulation information from each observation needed for pyIRF
            if 'simulation' in f.root.configuration:
                simulation_info = self._construct_simulated_info(f.root.configuration.simulation, simulation_info)
            # Teslecope selection
            telescopes, selected_telescopes = self._construct_telescopes_selection(self.files[filename].root.configuration.instrument.subarray.layout, selected_telescope_types, selected_telescope_ids)
            
            # Multiplicity selection
            subarray_multiplicity = 0
            for tel_type in selected_telescopes:
                if tel_type not in multiplicity_selection:
                    multiplicity_selection[tel_type] = 1
                else:
                    subarray_multiplicity += multiplicity_selection[tel_type]
            if subarray_multiplicity == 0:
                subarray_multiplicity = 1
            if 'Subarray' not in multiplicity_selection:
                multiplicity_selection['Subarray'] = subarray_multiplicity
                
            # MC event selection
            event_table = f.root.simulation.event.subarray.shower
            if event_selection is None:
                selected_nrows = list(range(len(event_table)))
            else:
                selected_nrows = [row.nrow for row in event_table.where(event_selection)]
            selected_nrows_tuple = np.stack((event_table[selected_nrows]["obs_id"], event_table[selected_nrows]["event_id"]), axis=-1)
            
            # Construct the example identifiers for 'mono' or 'stereo' events that passed the MC event selection.
            # TODO: Add commennts
            example_identifiers = []
            if self.mode == 'mono':
                for tel_id in selected_telescopes[self.tel_type]:
                    
                    tel_table = "tel_{:03d}".format(tel_id)
                    if tel_table in f.root.dl1.event.telescope.parameters:
                        # Image and parameter selection based on the parameter table
                        parameter_child = f.root.dl1.event.telescope.parameters._f_get_child(tel_table)
                        if parameter_selection is None:
                            image_indices = list(range(len(parameter_child)))
                        else:
                            image_indices = [row.nrow for row in parameter_child.where(parameter_selection)]
                    
                        # TODO: Timing with a larger dataset (or tables.Table) if for loop and searching for each image by obs_id and event_id via pytables.where() is faster than this concat and unique hack. Discussion more elegant and effiecient solution for this. Suggestions?
                        image_indices_tuple = np.stack((parameter_child[image_indices]["obs_id"], parameter_child[image_indices]["event_id"]), axis=-1)
                        unique_tuples, unique_counts = np.unique(np.concatenate((image_indices_tuple, selected_nrows_tuple)), axis=0, return_counts=True)
                    
                        if event_selection is None:
                            nrows = np.where(unique_counts > 1)[0]
                            for image_index, nrow in zip(image_indices, nrows):
                                example_identifiers.append((filename, nrow, image_index, tel_id))
                        else:
                            _, unique_tuples_images_count = np.unique(np.concatenate((image_indices_tuple, unique_tuples[np.where(unique_counts > 1)[0]])), axis=0, return_counts=True)
                            _, unique_tuples_nrows_count = np.unique(np.concatenate((selected_nrows_tuple, unique_tuples[np.where(unique_counts > 1)[0]])), axis=0, return_counts=True)
                            image_rows = np.where(unique_tuples_images_count > 1)[0]
                            nrows = np.where(unique_tuples_nrows_count > 1)[0]
                            for image_index, nrow in zip(image_rows, nrows):
                                example_identifiers.append((filename, selected_nrows[nrow], image_indices[image_index], tel_id))
            elif self.mode == 'stereo':
                for selected_nrow in selected_nrows:
                    tels_with_trigger = f.root.dl1.event.subarray.trigger.cols._f_col("tels_with_trigger")[selected_nrow]
                    mapped_array_triggers = []
                    for tel_type in selected_telescopes:
                        tel_ids = np.array(selected_telescopes[tel_type])
                        # match position of trigger array with subtract -1 from the tel_id
                        triggers = np.array(tels_with_trigger[tel_ids-1])
                        mapped_triggers = -np.ones(len(tel_ids), np.int8)
                        if sum(triggers) >= multiplicity_selection[tel_type]:
                            for tel_id in tel_ids[triggers]:
                                tel_table = "tel_{:03d}".format(tel_id)
                                if tel_table in f.root.dl1.event.telescope.images:
                                    images = self.files[filename].root.dl1.event.telescope.images._f_get_child(tel_table)
                                    idx = [row.nrow for row in images.where("(obs_id == {}) & (event_id == {})".format(selected_nrows_tuple[selected_nrow][0], selected_nrows_tuple[selected_nrow][1]))]
                                    mapped_triggers[np.where(tel_ids==tel_id)[0]] = idx[0]
                            if len(selected_telescopes) == 1:
                                mapped_array_triggers.append(mapped_triggers)
                        if len(selected_telescopes) > 1:
                            mapped_array_triggers.append(mapped_triggers)
                    subarray_trigger = 0
                    for trigger in mapped_array_triggers:
                        subarray_trigger += sum(trigger>=0)
                    if mapped_array_triggers and subarray_trigger >= multiplicity_selection['Subarray'] :
                        example_identifiers.append((filename, selected_nrow, mapped_array_triggers))

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

        
        if simulation_info:
            # Created pyIRF SimulatedEventsInfo
            self.pyIRFSimulatedEventsInfo = SimulatedEventsInfo(n_showers=int(simulation_info['num_showers'] * simulation_info['shower_reuse']),
                                                                energy_min=u.Quantity(simulation_info['energy_range_min'], u.TeV),
                                                                energy_max=u.Quantity(simulation_info['energy_range_max'], u.TeV),
                                                                max_impact=u.Quantity(simulation_info['max_scatter_range'], u.m),
                                                                spectral_index=simulation_info['spectral_index'],
                                                                viewcone=u.Quantity(simulation_info['max_viewcone_radius'], u.deg))

        # Shuffle the examples
        if shuffle:
            random.seed(seed)
            random.shuffle(self.example_identifiers)

        self.parameter_list = parameter_list
        self.image_channels = image_channels

        # ImageMapper (1D charges -> 2D images)
        if self.image_channels is not None:
            self.pixel_positions, self.num_pixels = self._construct_pixel_positions(f.root.configuration.instrument.telescope)
            if 'camera_types' not in mapping_settings:
                mapping_settings['camera_types'] = self.pixel_positions.keys()
            self.image_mapper = ImageMapper(pixel_positions=self.pixel_positions,
                                            **mapping_settings)
            
            for camera_type in mapping_settings['camera_types']:
                self.image_mapper.image_shapes[camera_type] = (
                    self.image_mapper.image_shapes[camera_type][0],
                    self.image_mapper.image_shapes[camera_type][1],
                    len(self.image_channels)  # number of channels
                )
                
        super()._construct_unprocessed_example_description(f.root.configuration.instrument.subarray.layout, event_table)
        
        self.processor = DL1DataProcessor(
            self.mode,
            self.unprocessed_example_description,
            transforms,
            validate_processor
            )

        # Definition of preprocessed example
        self.example_description = self.processor.output_description

    def _construct_telescopes_selection(self, subarray_table, selected_telescope_types, selected_telescope_ids):
        """
        Construct the selection of the telescopes from the args (`selected_telescope_types`, `selected_telescope_ids`).
        Parameters
        ----------
            subarray_table (tables.table):
            selected_telescope_type (array of str):
            selected_telescope_ids (array of int):
            #selection_string (str):
             
        Returns
        -------
        telescopes (dict): dictionary of `{: }`
        selected_telescopes (dict): dictionary of `{: }`
        #cut_condition (str): cut condition for pytables where function

        """
         
        # Get dict of all the tel_types in the file mapped to their tel_ids
        telescopes = {}
        for row in subarray_table:
            tel_type = row['tel_description'].decode()
            if tel_type not in telescopes:
                telescopes[tel_type] = []
            telescopes[tel_type].append(row['tel_id'])

        # Enforce an automatic minimal telescope selection cut:
        # there must be at least one triggered telescope of a
        # selected type in the event
        # Users can include stricter cuts in the selection string
        if selected_telescope_types is None:
            # Default: use the first tel type in the file
            default = subarray_table[0]['tel_description'].decode()
            selected_telescope_types = [default]
        if self.mode == 'mono':
            self.tel_type = selected_telescope_types[0]
        
        # Select which telescopes from the full dataset to include in each
        # event by a telescope type and an optional list of telescope ids.
        selected_telescopes = {}
        for tel_type in selected_telescope_types:
            available_tel_ids = telescopes[tel_type]
            # Keep only the selected tel ids for the tel type
            if tel_type in selected_telescope_ids:
                # Check all requested telescopes are available to select
                requested_tel_ids = selected_telescope_ids[tel_type]
                invalid_tel_ids = (set(requested_tel_ids) - set(available_tel_ids))
                if invalid_tel_ids:
                    raise ValueError("Tel ids {} are not a valid selection"
                                    "for tel type '{}'".format(invalid_tel_ids, tel_type))
                selected_telescopes[tel_type] = requested_tel_ids
            else:
                selected_telescopes[tel_type] = available_tel_ids
        
        return telescopes, selected_telescopes


    def _construct_simulated_info(self, simulation_table, simulation_info):
        """
        Construct the simulated_info from the DL1 hdf5 file for the pyIRF SimulatedEventsInfo table.
        Parameters
        ----------
            simulation_table (tables.Tables): table containing the simulation information
            simulation_info (dict): dictionary of pyIRF simulation info

        Returns
        -------
        simulation_info (dict): updated dictionary of pyIRF simulation info
        
        """
        
        runs = simulation_table._f_get_child('run')
        num_showers = sum(np.array(runs.cols._f_col("num_showers")))
        shower_reuse = max(np.array(runs.cols._f_col("shower_reuse")))
        energy_range_min = min(np.array(runs.cols._f_col("energy_range_min")))
        energy_range_max = max(np.array(runs.cols._f_col("energy_range_max")))
        max_scatter_range = max(np.array(runs.cols._f_col("max_scatter_range")))
        spectral_index = np.array(runs.cols._f_col("spectral_index"))[0]
        max_viewcone_radius = max(np.array(runs.cols._f_col("max_viewcone_radius")))
        if simulation_info:
            simulation_info["num_showers"] += num_showers
            if simulation_info["shower_reuse"] > shower_reuse:
                simulation_info["shower_reuse"] = shower_reuse
            if simulation_info["energy_range_min"] > energy_range_min:
                simulation_info["energy_range_min"] = energy_range_min
            if simulation_info["energy_range_max"] < energy_range_max:
                simulation_info["energy_range_max"] = energy_range_max
            if simulation_info["max_scatter_range"] < max_scatter_range:
                simulation_info["max_scatter_range"] = max_scatter_range
            if simulation_info["max_viewcone_radius"] < max_viewcone_radius:
                simulation_info["max_viewcone_radius"] = max_viewcone_radius
        else:
            simulation_info = {}
            simulation_info["num_showers"] = num_showers
            simulation_info["shower_reuse"] = shower_reuse
            simulation_info["energy_range_min"] = energy_range_min
            simulation_info["energy_range_max"] = energy_range_max
            simulation_info["max_scatter_range"] = max_scatter_range
            simulation_info["spectral_index"] = spectral_index
            simulation_info["max_viewcone_radius"] = max_viewcone_radius

        return simulation_info

    def _construct_pixel_positions(self, telescope_type_information):
        """
        Construct the pixel position of the cameras from the DL1 hdf5 file.
        Parameters
        ----------
            telescope_type_information (tables.Table):

        Returns
        -------
        pixel_positions (dict): dictionary of `{cameras: pixel_positions}`
        num_pixels (dict): dictionary of `{cameras: num_pixels}`

        """
        
        cameras = [description.decode('UTF-8').split("_")[-1] for description in telescope_type_information.optics.cols._f_col("description")]
        pixel_positions = {}
        num_pixels = {}
        for camera in cameras:
            cam_geom = telescope_type_information.camera._f_get_child('geometry_{}'.format(camera))
            pix_x = np.array(cam_geom.cols._f_col("pix_x"))
            pix_y = np.array(cam_geom.cols._f_col("pix_y"))
            num_pixels[camera] = len(pix_x)
            pixel_positions[camera] = np.stack((pix_x, pix_y))
            # For now hardcoded, since this information is not in the h5 files.
            # The official CTA DL1 format will contain this information.
            if camera in ['LSTCam', 'NectarCam', 'MAGICCam']:
                rotation_angle = -70.9 * np.pi/180.0 if camera == 'MAGICCam' else -100.893 * np.pi/180.0
                rotation_matrix = np.matrix([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                            [np.sin(rotation_angle), np.cos(rotation_angle)]], dtype=float)
                pixel_positions[camera] = np.squeeze(np.asarray(np.dot(rotation_matrix, pixel_positions[camera])))

        return pixel_positions, num_pixels
        
    
    def _load_tel_type_data(self, filename, nrow, tel_type, trigger_info):
        triggers = []
        images = []
        parameters_lists = []
        subarray_info = [[] for column in self.subarray_info]
        for i, tel_id in enumerate(self.selected_telescopes[tel_type]):
            if self.image_channels is not None:
                child = None
                with lock:
                    tel_table = "tel_{:03d}".format(tel_id)
                    if tel_table in self.files[filename].root.dl1.event.telescope.images:
                        child = self.files[filename].root.dl1.event.telescope.images._f_get_child(tel_table)
                if child:
                    image = super()._get_image(child, tel_type, trigger_info[i])
                    images.append(image)

            if self.parameter_list is not None:
                child = None
                with lock:
                    tel_table = "tel_{:03d}".format(tel_id)
                    if tel_table in self.files[filename].root.dl1.event.telescope.parameters:
                        child = self.files[filename].root.dl1.event.telescope.parameters._f_get_child(tel_table)
                parameter_list = []
                for parameter in self.parameter_list:
                    if trigger_info[i] != -1 and child:
                        parameter_list.append(child[trigger_info[i]][parameter])
                    else:
                        parameter_list.append(np.nan)
                parameters_lists.append(np.array(parameter_list, dtype=np.float32))

            tel_query = "tel_id == {}".format(tel_id)
            super()._append_subarray_info(self.files[filename].root.configuration.instrument.subarray.layout, subarray_info, tel_query)
        
        example = [np.array(trigger_info >= 0, np.int8)]
        if self.image_channels is not None:
            example.extend([np.stack(images)])
        if self.parameter_list is not None:
            example.extend([np.stack(parameters_lists)])
        example.extend([np.stack(info) for info in subarray_info])
        return example
    

    def __getitem__(self, idx):

        identifiers = self.example_identifiers[idx]

        # Get record for the event
        filename = identifiers[0]

        # Load the data and any selected array info
        if self.mode == "mono":
            # Get a single image
            nrow, index, tel_id = identifiers[1:4]
            example = []
            if self.image_channels is not None:
                with lock:
                    tel_table = "tel_{:03d}".format(tel_id)
                    child = self.files[filename].root.dl1.event.telescope.images._f_get_child(tel_table)
                image = super()._get_image(child, self.tel_type, index)
                example.append(image)

            if self.parameter_list is not None:
                with lock:
                    tel_table = "tel_{:03d}".format(tel_id)
                    child = self.files[filename].root.dl1.event.telescope.parameters._f_get_child(tel_table)
                    parameter_list = child[index][self.parameter_list]
                    example.append(np.array(list(parameter_list), dtype=np.float32))

            subarray_info = [[] for column in self.subarray_info]
            tel_query = "tel_id == {}".format(tel_id)
            super()._append_subarray_info(self.files[filename].root.configuration.instrument.subarray.layout, subarray_info, tel_query)
            example.extend([np.stack(info) for info in subarray_info])
        elif self.mode == "stereo":
            # Get a list of images and an array of binary trigger values
            # for each selected telescope type
            nrow = identifiers[1]
            trigger_info = identifiers[2]

            example = []
            for ind, tel_type in enumerate(self.selected_telescopes):
                tel_type_example = self._load_tel_type_data(filename, nrow, tel_type, trigger_info[ind])
                example.extend(tel_type_example)

        # Load event info
        with lock:
            events = self.files[filename].root.simulation.event.subarray.shower
            for column in self.event_info:
                dtype = events.cols._f_col(column).dtype
                example.append(np.array(events[nrow][column], dtype=dtype))

        # Preprocess the example
        example = self.processor.process(example)

        return example


# Not updated yet. Don't use. Supported until? At least ICRC2021 for a backup?
class DL1DataReaderDL1DH(DL1DataReader):

    def __init__(self,
                 file_list,
                 mode='mono',
                 selected_telescope_types=None,
                 selected_telescope_ids=None,
                 selection_string=None,
                 event_selection=None,
                 image_selection=None,
                 shuffle=False,
                 seed=None,
                 image_channels=None,
                 mapping_settings=None,
                 subarray_info=None,
                 event_info=None,
                 transforms=None,
                 validate_processor=False
                ):
                
        super().__init__(file_list=file_list, mode=mode, subarray_info=subarray_info, event_info=event_info)

        self.example_identifiers = None
        self.telescopes = {}
        if selected_telescope_ids is None:
            selected_telescope_ids = {}

        if event_selection is None:
            event_selection = {}

        if image_selection is None:
            image_selection = {}

        if mapping_settings is None:
            mapping_settings = {}
        
        for filename, f in self.files.items():

            # Tesleecope cutting
            telescopes, selected_telescopes, cut_condition = self._construct_selection_cuts(f.root.Array_Information, selected_telescope_types, selected_telescope_ids, selection_string)
            
            # Event cutting
            selected_nrows = set([row.nrow for row
                              in f.root.Events.where(cut_condition)])
            selected_nrows &= self._select_event(f, event_selection)
            selected_nrows = list(selected_nrows)

            # Image cutting
            # Make list of identifiers of all examples passing event selection
            if self.mode in ['stereo', 'multi-stereo']:
                example_identifiers = [(filename, nrow) for nrow
                                       in selected_nrows]
            elif self.mode == 'mono':
                example_identifiers = []
                field = '{}_indices'.format(self.tel_type)
                selected_indices = f.root.Events.read_coordinates(selected_nrows, field=field)
                for tel_id in selected_telescopes[self.tel_type]:
                    tel_index = telescopes[self.tel_type].index(tel_id)
                    img_ids = np.array(selected_indices[:, tel_index])
                    mask = (img_ids != 0)
                    # TODO handle all selected channels
                    mask[mask] &= self._select_image(
                        f.root['Images'][self.tel_type][img_ids[mask]]['charge'],
                        image_selection)
                    for image_index, nrow in zip(img_ids[mask],
                                               np.array(selected_nrows)[mask]):
                        example_identifiers.append((filename, nrow,
                                                    image_index, tel_id))

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

        # ImageMapper (1D charges -> 2D images)
        if image_channels is None:
            image_channels = ['charge']
        self.image_channels = image_channels
        self.pixel_positions, self.num_pixels = self._construct_pixel_positions(f.root.Telescope_Type_Information)
        if 'camera_types' not in mapping_settings:
            mapping_settings['camera_types'] = self.pixel_positions.keys()
        self.image_mapper = ImageMapper(pixel_positions=self.pixel_positions,
                                        **mapping_settings)
        camera_type = super()._get_camera_type(self.tel_type)
        self.image_mapper.image_shapes[camera_type] = (
                self.image_mapper.image_shapes[camera_type][0],
                self.image_mapper.image_shapes[camera_type][1],
                len(self.image_channels)  # number of channels
                )

        super()._construct_unprocessed_example_description(f.root.Array_Information, f.root.Events)
        
        self.processor = DL1DataProcessor(
            self.mode,
            self.unprocessed_example_description,
            transforms,
            validate_processor
            )

        # Definition of preprocessed example
        self.example_description = self.processor.output_description

    def _construct_selection_cuts(self, subarray_table, selected_telescope_type, selected_telescope_ids, selection_string):
        """
        Construct the pixel position of the cameras from the DL1 hdf5 file.
        Parameters
        ----------
            subarray_table (tables.table):
            selected_telescope_type (array of str):
            selected_telescope_ids (array of int):
            selection_string (str):
            
        Returns
        -------
        telescopes (dict): dictionary of `{: }`
        selected_telescopes (dict): dictionary of `{: }`
        cut_condition (str): cut condition for pytables where function

        """
        
        # Get dict of all the tel_types in the file mapped to their tel_ids
        telescopes = {}
        for row in subarray_table:
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
                default = subarray_table[0]['type'].decode()
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
                invalid_tel_ids = (set(requested_tel_ids)
                                   - set(available_tel_ids))
                if invalid_tel_ids:
                    raise ValueError("Tel ids {} are not a valid selection"
                                     "for tel type '{}'".format(
                                         invalid_tel_ids, tel_type))
                selected_telescopes[tel_type] = requested_tel_ids
            else:
                selected_telescopes[tel_type] = available_tel_ids
        
        return telescopes, selected_telescopes, cut_condition
    
    def _construct_pixel_positions(self, telescope_type_information):
        """
        Construct the pixel position of the cameras from the DL1 hdf5 file.
        Parameters
        ----------
            file (tables.): the file containing the data

        Returns
        -------
        pixel_positions (dict): dictionary of `{cameras: pixel_positions}`
        num_pixels (dict): dictionary of `{cameras: num_pixels}`

        """
        cameras = [x['camera'].decode() for x in telescope_type_information]
        num_pix = [x['num_pixels'] for x in telescope_type_information]
        pix_pos = [x['pixel_positions'] for x in telescope_type_information]
        pixel_positions = {}
        num_pixels = {}
        for i, camera in enumerate(cameras):
            pixel_positions[camera] = pix_pos[i][:num_pix[i]].T
            num_pixels[camera] = num_pix[i]
            # For now hardcoded, since this information is not in the h5 files.
            # The official CTA DL1 format will contain this information.
            if camera in ['LSTCam', 'NectarCam', 'MAGICCam']:
                rotation_angle = -70.9 * np.pi/180.0 if camera == 'MAGICCam' else -100.893 * np.pi/180.0
                rotation_matrix = np.matrix([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                            [np.sin(rotation_angle), np.cos(rotation_angle)]], dtype=float)
                pixel_positions[camera] = np.squeeze(np.asarray(np.dot(rotation_matrix, pixel_positions[camera])))
        
        return pixel_positions, num_pixels

    def _select_event(self, file, filters):
        """
        Filter the data event wise.
        Parameters
        ----------
            file (tables.File): the file containing the data
            filters (dict): dictionary of `{filter_function: filter_parameters}` to apply on the data

        Returns
        -------
        the filtered nrows

        """
        indices = set(np.arange(len(file.root.Events[:])))
        for filter_function, filter_parameters in filters.items():
            indices &= filter_function(self, file, **filter_parameters)
        return indices

    
    def _select_image(self, images, filters):
        """
        Filter the data image wise.
        Parameters
        ----------
            images (tables.File): the images to filter on
            filters (dict): dictionary of `{filter_function: filter_parameters}` to apply on the data

        Returns
        -------
        the mask of filtered images

        """
        mask = np.full(len(images), True)
        for filter_function, filter_parameters in filters.items():
            mask &= filter_function(self, images, **filter_parameters)
        return mask

    def _load_tel_type_data(self, filename, nrow, tel_type):
        images = []
        triggers = []
        subarray_info = [[] for column in self.subarray_info]
        with lock:
            child = self.files[filename].root['Images']._f_get_child(self.tel_type)
        for tel_id in self.selected_telescopes[tel_type]:
            tel_index = self.telescopes[tel_type].index(tel_id)
            with lock:
                image_index = self.files[filename].root.Events[nrow][
                    tel_type + '_indices'][tel_index]
            image = super()._get_image(child, tel_type, image_index)
            trigger = 0 if image_index == 0 else 1
            images.append(image)
            triggers.append(trigger)
            query = "id == {}".format(tel_id)
            super()._append_subarray_info(self.files[filename].root.Array_Information, subarray_info, query)

        example = [np.stack(images), np.array(triggers, dtype=np.int8)]
        example.extend([np.stack(info) for info in subarray_info])
        return example


    def __getitem__(self, idx):

        identifiers = self.example_identifiers[idx]

        # Get record for the event
        filename = identifiers[0]

        # Load the data and any selected array info
        if self.mode == "mono":
            # Get a single image
            nrow, image_index, tel_id = identifiers[1:4]
            with lock:
                child = self.files[filename].root['Images']._f_get_child(self.tel_type)
            image = super()._get_image(child, self.tel_type, image_index)
            example = [image]

            subarray_info = [[] for column in self.subarray_info]
            query = "id == {}".format(tel_id)
            super()._append_subarray_info(self.files[filename].root.Array_Information, subarray_info, query)
            example.extend([np.stack(info) for info in subarray_info])
        elif self.mode == "stereo":
            # Get a list of images and an array of binary trigger values
            nrow = identifiers[1]
            example = self._load_tel_type_data(filename, nrow, self.tel_type)
        elif self.mode == "multi-stereo":
            # Get a list of images and an array of binary trigger values
            # for each selected telescope type
            nrow = identifiers[1]
            example = []
            for tel_type in self.selected_telescopes:
                tel_type_example = self._load_tel_type_data(filename, nrow,
                                                            tel_type)
                example.extend(tel_type_example)

        # Load parameters, working only with mono mode at the moment
        if self.mode == "mono":
            with lock:
                parameters = self.files[filename].root['/Parameters' + str(self.algorithm)][self.tel_type]
                for column in self.training_parameters:
                    dtype = parameters.cols._f_col(column).dtype
                    example.append(np.array(parameters[image_index][column], dtype=dtype))

        # Load event info
        with lock:
            events = self.files[filename].root.Events
            for column in self.event_info:
                dtype = events.cols._f_col(column).dtype
                example.append(np.array(events[nrow][column], dtype=dtype))

        # Preprocess the example
        example = self.processor.process(example)

        return example

