from collections import Counter, OrderedDict
import random
import threading
import numpy as np
import pandas as pd
import tables

from dl1_data_handler.image_mapper import ImageMapper
from dl1_data_handler.processor import DL1DataProcessor

import astropy.units as u

# from astropy import table
from astropy.table import (
    Table,
    join,  # let us merge tables horizontally
    vstack,  # and vertically
)

__all__ = ["DL1DataReader", "DL1DataReaderSTAGE1", "DL1DataReaderDL1DH"]

split_datasets_by_tel_types = {
    "/simulation/event/telescope/images/LST_LST_LSTCam",
    "/simulation/event/telescope/images/MST_MST_FlashCam",
    "/simulation/event/telescope/images/MST_MST_NectarCam",
    "/simulation/event/telescope/images/SST_ASTRI_CHEC",
    "/dl1/event/telescope/images/LST_LST_LSTCam",
    "/dl1/event/telescope/images/MST_MST_FlashCam",
    "/dl1/event/telescope/images/MST_MST_NectarCam",
    "/dl1/event/telescope/images/SST_ASTRI_CHEC",
    "/simulation/event/telescope/parameters/LST_LST_LSTCam",
    "/simulation/event/telescope/parameters/MST_MST_FlashCam",
    "/simulation/event/telescope/parameters/MST_MST_NectarCam",
    "/simulation/event/telescope/parameters/SST_ASTRI_CHEC",
    "/dl1/event/telescope/parameters/LST_LST_LSTCam",
    "/dl1/event/telescope/parameters/MST_MST_FlashCam",
    "/dl1/event/telescope/parameters/MST_MST_NectarCam",
    "/dl1/event/telescope/parameters/SST_ASTRI_CHEC",
}

lock = threading.Lock()


class DL1DataReader:
    def __init__(self, file_list, mode="mono", subarray_info=None, event_info=None):

        # Construct dict of filename:file_handle pairs
        self.files = OrderedDict()
        # Order the file_list
        file_list = np.sort(file_list)
        for filename in file_list:
            with lock:
                self.files[filename] = tables.open_file(filename, mode="r")

        # Save the user attributes for the first file
        self._v_attrs = self.files[list(self.files)[0]].root._v_attrs

        # Set class weights to None
        self.class_weight = None

        # Translate from CORSIKA shower primary ID to the particle name
        self.shower_primary_id_to_name = {
            0: "gamma",
            101: "proton",
            1: "electron",
            255: "hadron",
        }

        # Set data loading mode
        # Mono: single images of one telescope type
        # Stereo: events including multiple telescope types
        if mode in ["mono", "stereo"]:
            self.mode = mode
        else:
            raise ValueError(
                "Invalid mode selection '{}'. Valid options: "
                "'mono', 'stereo'".format(mode)
            )

        if subarray_info is None:
            subarray_info = []
        self.subarray_info = subarray_info

        if event_info is None:
            event_info = []
        self.event_info = event_info

    def _get_camera_type(self, tel_type):
        return tel_type.split("_")[-1]

    def __len__(self):
        return len(self.example_identifiers)

    def _construct_unprocessed_example_description(
        self, subarray_table, events_table=None
    ):
        """
        Construct example description (before preprocessing).

        Parameters
        ----------
            subarray_table (tables.Table): the table containing the subarray information
            events_table (tables.Table): the table containing the simulated events information
        """
        if self.mode == "mono":
            self.unprocessed_example_description = []
            if self.image_channels is not None:
                self.unprocessed_example_description.append(
                    {
                        "name": "image",
                        "tel_type": self.tel_type,
                        "base_name": "image",
                        "shape": self.image_mapper.image_shapes[
                            self._get_camera_type(self.tel_type)
                        ],
                        "dtype": np.dtype(np.float32),
                    }
                )
            if self.parameter_list is not None:
                self.unprocessed_example_description.append(
                    {
                        "name": "parameters",
                        "tel_type": self.tel_type,
                        "base_name": "parameters",
                        "shape": ((1,) + (len(self.parameter_list),)),
                        "dtype": np.dtype(np.float32),
                    }
                )

            if self.pointing_mode == "divergent":
                self.unprocessed_example_description.append(
                    {
                        "name": "pointing",
                        "tel_type": self.tel_type,
                        "base_name": "pointing",
                        "shape": (2,),
                        "dtype": np.dtype(np.float32),
                    }
                )

            for col_name in self.subarray_info:
                col = subarray_table.cols._f_col(col_name)
                self.unprocessed_example_description.append(
                    {
                        "name": col_name,
                        "tel_type": self.tel_type,
                        "base_name": col_name,
                        "shape": (1,) + col.shape[1:],
                        "dtype": col.dtype,
                    }
                )

        elif self.mode == "stereo":
            self.unprocessed_example_description = []
            for tel_type in self.selected_telescopes:
                num_tels = len(self.selected_telescopes[tel_type])
                self.unprocessed_example_description.extend(
                    [
                        {
                            "name": tel_type + "_triggers",
                            "tel_type": tel_type,
                            "base_name": "triggers",
                            "shape": (num_tels,),
                            "dtype": np.dtype(np.int8),
                        }
                    ]
                )

                if self.image_channels is not None:
                    self.unprocessed_example_description.extend(
                        [
                            {
                                "name": tel_type + "_images",
                                "tel_type": tel_type,
                                "base_name": "images",
                                "shape": (
                                    (num_tels,)
                                    + self.image_mapper.image_shapes[
                                        self._get_camera_type(tel_type)
                                    ]
                                ),
                                "dtype": np.dtype(np.float32),
                            }
                        ]
                    )
                if self.parameter_list is not None:
                    self.unprocessed_example_description.extend(
                        [
                            {
                                "name": tel_type + "_parameters",
                                "tel_type": tel_type,
                                "base_name": "parameters",
                                "shape": ((num_tels,) + (len(self.parameter_list),)),
                                "dtype": np.dtype(np.float32),
                            }
                        ]
                    )

                if self.pointing_mode in ["divergent"]:
                    self.unprocessed_example_description.append(
                        {
                            "name": tel_type + "_pointings",
                            "tel_type": tel_type,
                            "base_name": "pointings",
                            "shape": (num_tels,) + (2,),
                            "dtype": np.dtype(np.float32),
                        }
                    )

                for col_name in self.subarray_info:
                    col = subarray_table.cols._f_col(col_name)
                    self.unprocessed_example_description.append(
                        {
                            "name": tel_type + "_" + col_name,
                            "tel_type": tel_type,
                            "base_name": col_name,
                            "shape": (num_tels,) + col.shape[1:],
                            "dtype": col.dtype,
                        }
                    )

        if self.pointing_mode == "subarray":
            self.unprocessed_example_description.append(
                {
                    "name": "pointing",
                    "tel_type": None,
                    "base_name": "pointing",
                    "shape": (2,),
                    "dtype": np.dtype(np.float32),
                }
            )

        # Add event info to description
        if self.process_type == "Simulation":
            for col_name in self.event_info:
                col = events_table.cols._f_col(col_name)
                self.unprocessed_example_description.append(
                    {
                        "name": col_name,
                        "tel_type": None,
                        "base_name": col_name,
                        "shape": col.shape[1:],
                        "dtype": col.dtype,
                    }
                )
        return

    def _construct_simulated_info(self, file, simulation_info, file_type="stage1"):
        """
        Construct the simulated_info from the DL1 hdf5 file for the pyIRF SimulatedEventsInfo table & GammaBoard.
        Parameters
        ----------
            file (hdf5 file): file containing the simulation information
            file_type (string): type of file (Valid option: 'stage1' or 'dl1dh')
            simulation_info (dict): dictionary of pyIRF simulation info

        Returns
        -------
        simulation_info (dict): updated dictionary of pyIRF simulation info

        """

        if file_type == "stage1":
            simulation_table = file.root.configuration.simulation
            runs = simulation_table._f_get_child("run")
            shower_reuse = max(np.array(runs.cols._f_col("shower_reuse")))
            if self.data_model_mainversion >= 4:
                n_showers = sum(np.array(runs.cols._f_col("n_showers"))) * shower_reuse
            else:
                n_showers = sum(np.array(runs.cols._f_col("num_showers"))) * shower_reuse
            if "service" in file.root.simulation:
                service_table = file.root.simulation.service
                shower_distributions = service_table._f_get_child("shower_distribution")
                num_showers = np.sum(
                    np.array(shower_distributions.cols._f_col("histogram"))
                )
            energy_range_min = min(np.array(runs.cols._f_col("energy_range_min")))
            energy_range_max = max(np.array(runs.cols._f_col("energy_range_max")))
            max_scatter_range = max(np.array(runs.cols._f_col("max_scatter_range")))
            spectral_index = np.array(runs.cols._f_col("spectral_index"))[0]
            min_viewcone_radius = max(np.array(runs.cols._f_col("min_viewcone_radius")))
            max_viewcone_radius = max(np.array(runs.cols._f_col("max_viewcone_radius")))
            min_alt = min(np.array(runs.cols._f_col("min_alt")))
            max_alt = max(np.array(runs.cols._f_col("max_alt")))
        elif file_type == "dl1dh":
            n_showers = file.root._v_attrs["num_showers"]
            energy_range_min = file.root._v_attrs["energy_range_min"]
            energy_range_max = file.root._v_attrs["energy_range_max"]
            max_scatter_range = file.root._v_attrs["max_scatter_range"]
            spectral_index = file.root._v_attrs["spectral_index"]
            min_viewcone_radius = file.root._v_attrs["min_viewcone_radius"]
            max_viewcone_radius = file.root._v_attrs["max_viewcone_radius"]
            min_alt = file.root._v_attrs["min_alt"]
            max_alt = file.root._v_attrs["max_alt"]

        if simulation_info:
            simulation_info["n_showers"] += float(n_showers)
            if simulation_info["energy_range_min"] > energy_range_min:
                simulation_info["energy_range_min"] = energy_range_min
            if simulation_info["energy_range_max"] < energy_range_max:
                simulation_info["energy_range_max"] = energy_range_max
            if simulation_info["max_scatter_range"] < max_scatter_range:
                simulation_info["max_scatter_range"] = max_scatter_range
            if simulation_info["min_viewcone_radius"] > min_viewcone_radius:
                simulation_info["min_viewcone_radius"] = min_viewcone_radius
            if simulation_info["max_viewcone_radius"] < max_viewcone_radius:
                simulation_info["max_viewcone_radius"] = max_viewcone_radius
            if simulation_info["min_alt"] > min_alt:
                simulation_info["min_alt"] = min_alt
            if simulation_info["max_alt"] < max_alt:
                simulation_info["max_alt"] = max_alt
        else:
            simulation_info = {}
            simulation_info["n_showers"] = float(n_showers)
            simulation_info["energy_range_min"] = energy_range_min
            simulation_info["energy_range_max"] = energy_range_max
            simulation_info["max_scatter_range"] = max_scatter_range
            simulation_info["spectral_index"] = spectral_index
            simulation_info["min_viewcone_radius"] = min_viewcone_radius
            simulation_info["max_viewcone_radius"] = max_viewcone_radius
            simulation_info["min_alt"] = min_alt
            simulation_info["max_alt"] = max_alt

        return simulation_info

    # Get a single telescope image from a particular event, uniquely
    # identified by the filename, tel_type, and image table index.
    # First extract a raw 1D vector and transform it into a 2D image using a
    # mapping table. When 'indexed_conv' is selected this function should
    # return the unmapped vector.
    def _get_image(self, child, tel_type, image_index, parameter_table=-1):

        vector = np.zeros(
            shape=(
                self.num_pixels[self._get_camera_type(tel_type)],
                len(self.image_channels),
            ),
            dtype=np.float32,
        )
        # If the telescope didn't trigger, the image index is -1 and a blank
        # image of all zeros with be loaded
        if image_index != -1 and child:
            with lock:
                record = child[image_index]
                for i, channel in enumerate(self.image_channels):
                    if "clean" in channel or "mask" in channel:
                        cleaning_mask = "image_mask"
                        if parameter_table >= 0:
                            cleaning_mask += str(parameter_table)
                        if "image" in channel:
                            vector[:, i] = record["image"] * record[cleaning_mask]
                        if "time" in channel:
                            vector[:, i] = record["peak_time"] * record[cleaning_mask]
                    else:
                        vector[:, i] = record[channel]

                    # Apply the transform to recover orginal floating point values if the file were compressed
                    if "image" in channel and self.image_scale:
                        vector[:, i] /= self.image_scale
                    if "time" in channel and self.image_scale:
                        vector[:, i] /= self.peak_time_scale

        # If 'indexed_conv' is selected, we only need the unmapped vector.
        if (
            self.image_mapper.mapping_method[self._get_camera_type(tel_type)]
            == "indexed_conv"
        ):
            return vector

        return self.image_mapper.map_image(vector, self._get_camera_type(tel_type))

    def _append_subarray_info(self, subarray_table, subarray_info, query):
        with lock:
            for row in subarray_table.where(query):
                for info, column in zip(subarray_info, self.subarray_info):
                    dtype = subarray_table.cols._f_col(column).dtype
                    info.append(np.array(row[column], dtype=dtype))
        return


# CTA DL1 data model
class DL1DataReaderSTAGE1(DL1DataReader):
    def __init__(
        self,
        file_list,
        example_identifiers_file=None,
        mode="mono",
        pointing_mode="fix_subarray",
        selected_telescope_types=None,
        selected_telescope_ids=None,
        multiplicity_selection=None,
        event_selection=None,
        parameter_selection=None,
        shuffle=False,
        seed=None,
        image_channels=None,
        mapping_settings=None,
        parameter_list=None,
        subarray_info=None,
        event_info=None,
        transforms=None,
        validate_processor=False,
    ):

        # Import ctapipe DL1 reader
        from ctapipe.io import (
            read_table,
        )  # let us read full tables inside the DL1 output file

        super().__init__(
            file_list=file_list,
            mode=mode,
            subarray_info=subarray_info,
            event_info=event_info,
        )

        first_file = list(self.files)[0]
        self.data_model_version = self._v_attrs["CTA PRODUCT DATA MODEL VERSION"]
        self.data_model_mainversion = int(self.data_model_version.split(".")[0].replace("v",""))
        self.process_type = self._v_attrs["CTA PROCESS TYPE"]
        self.instrument_id = self._v_attrs["CTA INSTRUMENT ID"]

        # Set pointing mode
        # Fix_subarray: Fix subarray pointing (MC production)
        # Subarray: Subarray pointing with different pointing over time (Operation or MC production with different pointing)
        # Fix_divergent: Fix divergent pointing (MC production)
        # Divergent: Divergent pointing with different pointing over time (Operation or MC production with different pointing)
        if pointing_mode in ["fix_subarray", "subarray", "fix_divergent", "divergent"]:
            self.pointing_mode = pointing_mode
        else:
            raise ValueError(
                "Invalid pointing mode selection '{}'. Valid options: "
                "'fix_subarray', 'subarray', 'fix_divergent', 'divergent'".format(
                    pointing_mode
                )
            )

        if selected_telescope_ids is None:
            selected_telescope_ids = []
        (
            self.telescopes,
            self.selected_telescopes,
            self.camera2index,
        ) = self._construct_telescopes_selection(
            self.files[first_file].root.configuration.instrument.subarray.layout,
            selected_telescope_types,
            selected_telescope_ids,
        )

        if multiplicity_selection is None:
            multiplicity_selection = {}

        if mapping_settings is None:
            mapping_settings = {}

        self.parameter_list = parameter_list
        self.image_channels = image_channels
        self.image_scale = None
        self.peak_time_scale = None

        # Get stage1 split_datasets_by type
        self.split_datasets_by = "tel_id"
        for table in split_datasets_by_tel_types:
            if table in self.files[first_file].root:
                self.split_datasets_by = "tel_type"
                if table.split("/")[-2] == "images" and self.image_channels is not None:
                    # Check the transform value used for the file compression
                    if (
                        "image_TRANSFORM_SCALE"
                        in self.files[first_file].root[table]._v_attrs
                        and self.image_scale is None
                    ):
                        self.image_scale = (
                            self.files[first_file]
                            .root[table]
                            ._v_attrs["image_TRANSFORM_SCALE"]
                        )
                    if (
                        "peak_time_TRANSFORM_SCALE"
                        in self.files[first_file].root[table]._v_attrs
                        and self.peak_time_scale is None
                    ):
                        self.peak_time_scale = (
                            self.files[first_file]
                            .root[table]
                            ._v_attrs["peak_time_TRANSFORM_SCALE"]
                        )

        self.simulation_info = None
        self.simulated_particles = {}
        self.simulated_particles["total"] = 0
        self.example_identifiers = None
        if example_identifiers_file is None:
            example_identifiers_file = {}
        else:
            example_identifiers_file = pd.HDFStore(example_identifiers_file)

        if "/example_identifiers" in list(example_identifiers_file.keys()):
            self.example_identifiers = pd.read_hdf(
                example_identifiers_file, key="/example_identifiers"
            ).to_numpy()
            if "/simulation_info" in list(example_identifiers_file.keys()):
                self.simulation_info = pd.read_hdf(
                    example_identifiers_file, key="/simulation_info"
                ).to_dict("records")[0]
            if "/simulated_particles" in list(example_identifiers_file.keys()):
                self.simulated_particles = pd.read_hdf(
                    example_identifiers_file, key="/simulated_particles"
                ).to_dict("records")[0]
            if "/class_weight" in list(example_identifiers_file.keys()):
                self.class_weight = pd.read_hdf(
                    example_identifiers_file, key="/class_weight"
                ).to_dict("records")[0]
            if "/class_names" in list(example_identifiers_file.keys()):
                class_names = pd.read_hdf(
                    example_identifiers_file, key="/class_names"
                ).to_dict("records")
                self.class_names = [name[0] for name in class_names]
            if "/shower_primary_id_to_class" in list(example_identifiers_file.keys()):
                self.shower_primary_id_to_class = pd.read_hdf(
                    example_identifiers_file, key="/shower_primary_id_to_class"
                ).to_dict("records")[0]
            self.num_classes = len(self.simulated_particles) - 1
            example_identifiers_file.close()
        else:

            for file_idx, (filename, f) in enumerate(self.files.items()):

                # Read simulation information from each observation needed for pyIRF
                if self.process_type == "Simulation":
                    self.simulation_info = super()._construct_simulated_info(
                        f, self.simulation_info, file_type="stage1"
                    )
                # Telescope selection
                (
                    telescopes,
                    selected_telescopes,
                    camera2index,
                ) = self._construct_telescopes_selection(
                    f.root.configuration.instrument.subarray.layout,
                    selected_telescope_types,
                    selected_telescope_ids,
                )

                # Multiplicity selection
                if "Subarray" not in multiplicity_selection:
                    multiplicity_selection["Subarray"] = 1
                    for tel_type in selected_telescopes:
                        if tel_type in multiplicity_selection:
                            multiplicity_selection["Subarray"] = multiplicity_selection[
                                tel_type
                            ]
                if len(selected_telescopes) > 1:
                    for tel_type in selected_telescopes:
                        if tel_type not in multiplicity_selection:
                            multiplicity_selection[tel_type] = 0
                else:
                    multiplicity_selection[
                        list(selected_telescopes.keys())[0]
                    ] = multiplicity_selection["Subarray"]

                # Construct the shower simulation table
                if self.process_type == "Simulation":
                    simshower_table = read_table(f, "/simulation/event/subarray/shower")
                    simshower_table.add_column(
                        np.arange(len(simshower_table)), name="sim_index", index=0
                    )
                    true_shower_primary_id = simshower_table["true_shower_primary_id"][
                        0
                    ]

                # Construct the example identifiers for 'mono' or 'stereo' mode.
                example_identifiers = []
                if self.mode == "mono":

                    if self.split_datasets_by == "tel_id":
                        # Construct the table containing all events.
                        # First, the telescope tables are joined with the shower simulation
                        # table and then those joined/merged tables are vertically stacked.
                        tel_tables = []
                        for tel_id in selected_telescopes[self.tel_type]:
                            tel_table = read_table(
                                f, f"/dl1/event/telescope/parameters/tel_{tel_id:03d}"
                            )
                            tel_table.add_column(
                                np.arange(len(tel_table)), name="img_index", index=0
                            )

                            if self.process_type == "Simulation":
                                tel_table = join(
                                    left=tel_table,
                                    right=simshower_table,
                                    keys=["obs_id", "event_id"],
                                )
                            tel_tables.append(tel_table)
                        allevents = vstack(tel_tables)
                    elif self.split_datasets_by == "tel_type":
                        # Construct the table containing all events.
                        # Join the telescope type table with the shower simulation table.
                        tel_type_table = read_table(
                            f, f"/dl1/event/telescope/parameters/{self.tel_type}"
                        )
                        tel_type_table.add_column(
                            np.arange(len(tel_type_table)), name="img_index", index=0
                        )
                        tel_tables = []
                        for tel_id in selected_telescopes[self.tel_type]:
                            tel_table = tel_type_table[
                                tel_type_table["tel_id"] == tel_id
                            ]

                            if self.process_type == "Simulation":
                                tel_table = join(
                                    left=tel_table,
                                    right=simshower_table,
                                    keys=["obs_id", "event_id"],
                                )
                            tel_tables.append(tel_table)
                        allevents = vstack(tel_tables)

                    # MC event selection based on the shower simulation table
                    if event_selection:
                        for filter in event_selection:
                            if "min_value" in filter:
                                allevents = allevents[
                                    allevents[filter["col_name"]] >= filter["min_value"]
                                ]
                            if "max_value" in filter:
                                allevents = allevents[
                                    allevents[filter["col_name"]] < filter["max_value"]
                                ]

                    # Image and parameter selection based on the parameter tables
                    if parameter_selection:
                        for filter in parameter_selection:
                            if "min_value" in filter:
                                if filter["col_name"] in allevents.colnames:
                                    allevents = allevents[
                                        allevents[filter["col_name"]] >= filter["min_value"]
                                    ]
                                else:
                                    allevents = allevents[
                                        allevents["camera_frame_"+filter["col_name"]] >= filter["min_value"]
                                    ]
                            if "max_value" in filter:
                                if filter["col_name"] in allevents.colnames:
                                    allevents = allevents[
                                        allevents[filter["col_name"]] < filter["max_value"]
                                    ]
                                else:
                                    allevents = allevents[
                                        allevents["camera_frame_"+filter["col_name"]] < filter["max_value"]
                                    ]

                    # TODO: Fix pointing over time (see ctapipe issue 1484 & 1562)
                    if self.pointing_mode in ["subarray", "divergent"]:
                        array_pointing = 0

                    # Track number of events for each particle type
                    if self.process_type == "Simulation":
                        self.simulated_particles["total"] += len(allevents)
                        if true_shower_primary_id in self.simulated_particles:
                            self.simulated_particles[true_shower_primary_id] += len(
                                allevents
                            )
                        else:
                            self.simulated_particles[true_shower_primary_id] = len(
                                allevents
                            )

                        # Construct the example identifiers
                        for sim_idx, img_idx, tel_id in zip(
                            allevents["sim_index"],
                            allevents["img_index"],
                            allevents["tel_id"],
                        ):
                            if self.pointing_mode in ["subarray", "divergent"]:
                                example_identifiers.append(
                                    (file_idx, sim_idx, img_idx, tel_id, array_pointing)
                                )
                            else:
                                example_identifiers.append(
                                    (file_idx, sim_idx, img_idx, tel_id)
                                )
                    else:
                        # Construct the example identifiers
                        for img_idx, tel_id in zip(
                            allevents["img_index"],
                            allevents["tel_id"],
                        ):
                            if self.pointing_mode in ["subarray", "divergent"]:
                                example_identifiers.append(
                                    (file_idx, img_idx, tel_id, array_pointing)
                                )
                            else:
                                example_identifiers.append((file_idx, img_idx, tel_id))

                elif self.mode == "stereo":

                    # Read the trigger table.
                    allevents = read_table(f, "/dl1/event/subarray/trigger")
                    if self.process_type == "Simulation":
                        # The shower simulation table is joined with the subarray trigger table.
                        allevents = join(
                            left=allevents,
                            right=simshower_table,
                            keys=["obs_id", "event_id"],
                        )

                        # MC event selection based on the shower simulation table.
                        if event_selection:
                            for filter in event_selection:
                                if "min_value" in filter:
                                    allevents = allevents[
                                        allevents[filter["col_name"]]
                                        >= filter["min_value"]
                                    ]
                                if "max_value" in filter:
                                    allevents = allevents[
                                        allevents[filter["col_name"]]
                                        < filter["max_value"]
                                    ]

                    # Apply the multiplicity cut on the subarray.
                    # Therefore, two telescope types have to be selected at least.
                    event_id = allevents["event_id"]
                    tels_with_trigger = np.array(allevents["tels_with_trigger"])
                    if self.process_type == "Simulation":
                        sim_indices = np.array(allevents["sim_index"], np.int32)
                    if len(selected_telescopes) > 1:
                        # Get all tel ids from the subarray
                        tel_ids = np.hstack(list(selected_telescopes.values()))
                        # Construct a boolean array of allowed telescopes
                        allowed_tels = np.array(
                            [
                                1 if tel_id in tel_ids - 1 else 0
                                for tel_id in range(tels_with_trigger.shape[1])
                            ],
                            bool,
                        )
                        # Construct the telescope trigger information restricted to allowed telescopes
                        allowed_tels_with_trigger = tels_with_trigger * allowed_tels
                        # Get the multiplicity and apply the subarray multiplicity cut
                        subarray_multiplicity, _ = allowed_tels_with_trigger.nonzero()
                        events, multiplicity = np.unique(
                            subarray_multiplicity, axis=0, return_counts=True
                        )
                        selected_events = events[
                            np.where(multiplicity >= multiplicity_selection["Subarray"])
                        ]
                        event_id = event_id[selected_events]
                        if self.process_type == "Simulation":
                            sim_indices = sim_indices[selected_events]

                    image_indices = {}
                    for tel_type in selected_telescopes:
                        # Get all selected tel ids of this telescope type
                        tel_ids = np.array(selected_telescopes[tel_type])
                        # Construct a boolean array of allowed telescopes of this telescope type
                        allowed_tels = np.array(
                            [
                                1 if tel_id in tel_ids - 1 else 0
                                for tel_id in range(tels_with_trigger.shape[1])
                            ],
                            bool,
                        )
                        # Construct the telescope trigger information restricted to allowed telescopes of this telescope type
                        allowed_tels_with_trigger = tels_with_trigger * allowed_tels
                        # Apply the multiplicity cut on the telescope type only.
                        if len(selected_telescopes) == 1:
                            # Get the multiplicity of this telescope type and apply the multiplicity cut
                            (
                                tel_type_multiplicity,
                                _,
                            ) = allowed_tels_with_trigger.nonzero()
                            events, multiplicity = np.unique(
                                tel_type_multiplicity, axis=0, return_counts=True
                            )
                            selected_events = events[
                                np.where(
                                    multiplicity >= multiplicity_selection[tel_type]
                                )
                            ]
                            event_id = event_id[selected_events]
                            if self.process_type == "Simulation":
                                sim_indices = sim_indices[selected_events]
                        selected_events_trigger = allowed_tels_with_trigger[
                            selected_events
                        ]

                        # Get the position of each images of telescopes of this telescope type that triggered
                        img_idx = -np.ones(
                            (len(selected_events), len(tel_ids)), np.int32
                        )

                        if self.split_datasets_by == "tel_id":
                            for tel_id in tel_ids:
                                # Get the trigger information of this telescope
                                tel_trigger_info = selected_events_trigger[
                                    :, tel_id - 1
                                ]
                                tel_trigger_info = np.where(tel_trigger_info)[0]
                                # The telescope table is joined with the selected and merged table.
                                tel_table = read_table(
                                    f,
                                    f"/dl1/event/telescope/parameters/tel_{tel_id:03d}",
                                )
                                tel_table.add_column(
                                    np.arange(len(tel_table)), name="img_index", index=0
                                )
                                # MC event selection based on the parameter tables.
                                if parameter_selection:
                                    for filter in parameter_selection:
                                        if "min_value" in filter:
                                            if filter["col_name"] in tel_table.colnames:
                                                tel_table = tel_table[
                                                    tel_table[filter["col_name"]]
                                                    >= filter["min_value"]
                                                ]
                                            else:
                                                tel_table = tel_table[
                                                    tel_table["camera_frame_"+filter["col_name"]]
                                                    >= filter["min_value"]
                                                ]
                                        if "max_value" in filter:
                                            if filter["col_name"] in tel_table.colnames:
                                                tel_table = tel_table[
                                                    tel_table[filter["col_name"]]
                                                    < filter["max_value"]
                                                ]
                                            else:
                                                tel_table = tel_table[
                                                    tel_table["camera_frame_"+filter["col_name"]]
                                                    < filter["max_value"]
                                                ]
                                merged_table = join(
                                    left=tel_table,
                                    right=allevents[selected_events],
                                    keys=["obs_id", "event_id"],
                                )
                                # Get the original position of image in the telescope table.
                                tel_img_index = np.array(
                                    merged_table["img_index"], np.int32
                                )
                                for trig, img in zip(tel_trigger_info, tel_img_index):
                                    img_idx[trig][np.where(tel_ids == tel_id)] = img

                            # Apply the multiplicity cut after the parameter cuts for a particular telescope type
                            if (
                                parameter_selection
                                and multiplicity_selection[tel_type] > 0
                            ):
                                aftercuts_multiplicty_mask = (
                                    np.count_nonzero(img_idx + 1, axis=1)
                                    >= multiplicity_selection[tel_type]
                                )
                                img_idx = img_idx[aftercuts_multiplicty_mask]
                                event_id = event_id[aftercuts_multiplicty_mask]
                                if self.process_type == "Simulation":
                                    sim_indices = sim_indices[
                                        aftercuts_multiplicty_mask
                                    ]
                            image_indices[tel_type] = img_idx

                        elif self.split_datasets_by == "tel_type":
                            # The telescope table is joined with the selected and merged table.
                            tel_type_table = read_table(
                                f, f"/dl1/event/telescope/parameters/{tel_type}"
                            )
                            tel_type_table.add_column(
                                np.arange(len(tel_type_table)),
                                name="img_index",
                                index=0,
                            )
                            # MC event selection based on the parameter tables.
                            if parameter_selection:
                                for filter in parameter_selection:
                                    if "min_value" in filter:
                                        tel_type_table = tel_type_table[
                                            tel_type_table[filter["col_name"]]
                                            >= filter["min_value"]
                                        ]
                                    if "max_value" in filter:
                                        tel_type_table = tel_type_table[
                                            tel_type_table[filter["col_name"]]
                                            < filter["max_value"]
                                        ]

                            merged_table = join(
                                left=tel_type_table,
                                right=allevents[selected_events],
                                keys=["obs_id", "event_id"],
                            )

                            for tel_id in tel_ids:
                                merged_table_per_tel_id = merged_table[
                                    merged_table["tel_id"] == int(tel_id)
                                ]
                                # Get the trigger information of this telescope
                                tel_trigger_info = selected_events_trigger[
                                    :, tel_id - 1
                                ]
                                tel_trigger_info = np.where(tel_trigger_info)[0]
                                # Get the original position of image in the telescope table.
                                tel_img_index = np.array(
                                    merged_table_per_tel_id["img_index"], np.int32
                                )
                                for trig, img in zip(tel_trigger_info, tel_img_index):
                                    img_idx[trig][np.where(tel_ids == tel_id)] = img

                            # Apply the multiplicity cut after the parameter cuts for a particular telescope type
                            if (
                                parameter_selection
                                and multiplicity_selection[tel_type] > 0
                            ):
                                aftercuts_multiplicty_mask = (
                                    np.count_nonzero(img_idx + 1, axis=1)
                                    >= multiplicity_selection[tel_type]
                                )
                                img_idx = img_idx[aftercuts_multiplicty_mask]
                                event_id = event_id[aftercuts_multiplicty_mask]
                                if self.process_type == "Simulation":
                                    sim_indices = sim_indices[
                                        aftercuts_multiplicty_mask
                                    ]
                            image_indices[tel_type] = img_idx

                    # Apply the multiplicity cut after the parameter cuts for the subarray
                    if parameter_selection and multiplicity_selection["Subarray"] > 1:
                        subarray_triggers = np.zeros(len(event_id))
                        for tel_type in selected_telescopes:
                            subarray_triggers += np.count_nonzero(
                                image_indices[tel_type] + 1, axis=1
                            )
                        aftercuts_multiplicty_mask = (
                            subarray_triggers >= multiplicity_selection["Subarray"]
                        )
                        if self.process_type == "Simulation":
                            sim_indices = sim_indices[aftercuts_multiplicty_mask]
                        for tel_type in selected_telescopes:
                            image_indices[tel_type] = image_indices[tel_type][
                                aftercuts_multiplicty_mask
                            ]

                    # TODO: Fix pointing over time (see ctapipe issue 1484 & 1562)
                    if self.pointing_mode == "subarray":
                        array_pointings = 0
                    elif self.pointing_mode == "divergent":
                        tel_ids = np.hstack(list(selected_telescopes.values()))
                        array_pointings = np.zeros(len(tel_ids), np.int8)

                    if self.process_type == "Simulation":
                        # Track number of events for each particle type
                        self.simulated_particles["total"] += len(sim_indices)
                        if true_shower_primary_id in self.simulated_particles:
                            self.simulated_particles[true_shower_primary_id] += len(
                                sim_indices
                            )
                        else:
                            self.simulated_particles[true_shower_primary_id] = len(
                                sim_indices
                            )

                        # Construct the example identifiers
                        # TODO: Find a better way!?
                        for idx, sim_idx in enumerate(sim_indices):
                            img_idx = []
                            for tel_type in selected_telescopes:
                                img_idx.append(image_indices[tel_type][idx])
                            if self.pointing_mode in ["subarray", "divergent"]:
                                example_identifiers.append(
                                    (file_idx, sim_idx, img_idx, array_pointings)
                                )
                            else:
                                example_identifiers.append((file_idx, sim_idx, img_idx))

                # Confirm that the files are consistent and merge them
                if not self.telescopes:
                    self.telescopes = telescopes
                if self.telescopes != telescopes:
                    raise ValueError(
                        "Inconsistent telescope definition in " "{}".format(filename)
                    )
                self.selected_telescopes = selected_telescopes

                if self.example_identifiers is None:
                    self.example_identifiers = example_identifiers
                else:
                    self.example_identifiers.extend(example_identifiers)

            # Handling the particle ids automatically and class weights calculation
            # Scaling by total/2 helps keep the loss to a similar magnitude.
            # The sum of the weights of all examples stays the same.
            self.num_classes = len(self.simulated_particles) - 1
            if self.process_type == "Simulation":
                if len(self.simulated_particles) > 2:
                    self.shower_primary_id_to_class = {}
                    self.class_names = []
                    for p, particle_id in enumerate(
                        list(self.simulated_particles.keys())[1:]
                    ):
                        self.shower_primary_id_to_class[particle_id] = p
                        self.class_names.append(
                            (self.shower_primary_id_to_name[particle_id])
                        )

                    self.class_weight = {}
                    for particle_id, num_particles in self.simulated_particles.items():
                        if particle_id != "total":
                            self.class_weight[
                                self.shower_primary_id_to_class[particle_id]
                            ] = (1 / num_particles) * (
                                self.simulated_particles["total"] / 2.0
                            )

            # Shuffle the examples
            if shuffle:
                random.seed(seed)
                random.shuffle(self.example_identifiers)

            # Dump example_identifiers and simulation_info to a pandas hdf5 file
            if not isinstance(example_identifiers_file, dict):
                pd.DataFrame(data=self.example_identifiers).to_hdf(
                    example_identifiers_file, key="example_identifiers", mode="a"
                )
                if self.simulation_info:
                    pd.DataFrame(
                        data=pd.DataFrame(self.simulation_info, index=[0])
                    ).to_hdf(example_identifiers_file, key="simulation_info", mode="a")
                if self.simulated_particles:
                    pd.DataFrame(
                        data=pd.DataFrame(self.simulated_particles, index=[0])
                    ).to_hdf(
                        example_identifiers_file, key="simulated_particles", mode="a"
                    )
                    if self.class_weight:
                        pd.DataFrame(
                            data=pd.DataFrame(self.class_weight, index=[0])
                        ).to_hdf(example_identifiers_file, key="class_weight", mode="a")
                        pd.DataFrame(data=pd.DataFrame(self.class_names)).to_hdf(
                            example_identifiers_file, key="class_names", mode="a"
                        )
                        pd.DataFrame(
                            data=pd.DataFrame(
                                self.shower_primary_id_to_class, index=[0]
                            )
                        ).to_hdf(
                            example_identifiers_file,
                            key="shower_primary_id_to_class",
                            mode="a",
                        )
                example_identifiers_file.close()

        # ImageMapper (1D charges -> 2D images)
        if self.image_channels is not None:

            # Check the transform value used for the file compression
            for tel_id in np.arange(1, 180):
                tel_table = "tel_{:03d}".format(tel_id)
                if tel_table in self.files[first_file].root.dl1.event.telescope.images:
                    if (
                        "image_TRANSFORM_SCALE"
                        in self.files[first_file]
                        .root.dl1.event.telescope.images[tel_table]
                        ._v_attrs
                        and self.image_scale is None
                    ):
                        self.image_scale = (
                            self.files[first_file]
                            .root.dl1.event.telescope.images[tel_table]
                            ._v_attrs["image_TRANSFORM_SCALE"]
                        )
                    if (
                        "peak_time_TRANSFORM_SCALE"
                        in self.files[first_file]
                        .root.dl1.event.telescope.images[tel_table]
                        ._v_attrs
                        and self.peak_time_scale is None
                    ):
                        self.peak_time_scale = (
                            self.files[first_file]
                            .root.dl1.event.telescope.images[tel_table]
                            ._v_attrs["peak_time_TRANSFORM_SCALE"]
                        )

            self.pixel_positions, self.num_pixels = self._construct_pixel_positions(
                self.files[first_file].root.configuration.instrument.telescope
            )

            if "camera_types" not in mapping_settings:
                mapping_settings["camera_types"] = self.pixel_positions.keys()
            self.image_mapper = ImageMapper(
                pixel_positions=self.pixel_positions, **mapping_settings
            )

            for camera_type in mapping_settings["camera_types"]:
                self.image_mapper.image_shapes[camera_type] = (
                    self.image_mapper.image_shapes[camera_type][0],
                    self.image_mapper.image_shapes[camera_type][1],
                    len(self.image_channels),  # number of channels
                )

        if self.pointing_mode == "fix_subarray":
            subarray_pointing = self.files[
                first_file
            ].root.dl1.monitoring.subarray.pointing
            self.pointing = np.array(
                [
                    subarray_pointing[0]["array_altitude"],
                    subarray_pointing[0]["array_azimuth"],
                ],
                np.float32,
            )
            # Set the telescope pointing to the delta Alt/Az tranform
            if transforms is not None:
                for transform in transforms:
                    if transform.name == "deltaAltAz_fix_subarray":
                        transform.set_tel_pointing(self.pointing)

        elif self.pointing_mode == "fix_divergent":
            self.pointing = {}
            for tel_type in selected_telescopes:
                tel_ids = np.array(selected_telescopes[tel_type])
                for tel_id in tel_ids:
                    tel_table = "tel_{:03d}".format(tel_id)
                    telescope_pointing = self.files[
                        first_file
                    ].root.dl1.monitoring.telescope.pointing._f_get_child(tel_table)
                    self.pointing[tel_id] = np.array(
                        [
                            telescope_pointing[0]["altitude"],
                            telescope_pointing[0]["azimuth"],
                        ],
                        np.float32,
                    )

        if self.process_type == "Simulation":
            super()._construct_unprocessed_example_description(
                self.files[first_file].root.configuration.instrument.subarray.layout,
                self.files[first_file].root.simulation.event.subarray.shower,
            )
        else:
            super()._construct_unprocessed_example_description(
                self.files[first_file].root.configuration.instrument.subarray.layout
            )

        self.processor = DL1DataProcessor(
            self.mode,
            self.unprocessed_example_description,
            transforms,
            validate_processor,
        )

        # Definition of preprocessed example
        self.example_description = self.processor.output_description

    def _construct_telescopes_selection(
        self, subarray_table, selected_telescope_types, selected_telescope_ids
    ):
        """
        Construct the selection of the telescopes from the args (`selected_telescope_types`, `selected_telescope_ids`).
        Parameters
        ----------
            subarray_table (tables.table):
            selected_telescope_type (array of str):
            selected_telescope_ids (array of int):

        Returns
        -------
        telescopes (dict): dictionary of `{: }`
        selected_telescopes (dict): dictionary of `{: }`
        camera2index (dict): dictionary of `{: }`

        """

        # Get dict of all the tel_types in the file mapped to their tel_ids
        telescopes = {}
        camera2index = {}
        for row in subarray_table:
            tel_type = row["tel_description"].decode()
            if tel_type not in telescopes:
                telescopes[tel_type] = []
            if self.data_model_mainversion > 1:
                camera_index = row["camera_index"]
                if self._get_camera_type(tel_type) not in camera2index:
                    camera2index[self._get_camera_type(tel_type)] = camera_index
            telescopes[tel_type].append(row["tel_id"])

        # Enforce an automatic minimal telescope selection cut:
        # there must be at least one triggered telescope of a
        # selected type in the event
        # Users can include stricter cuts in the selection string
        if selected_telescope_types is None:
            # Default: use the first tel type in the file
            default = subarray_table[0]["tel_description"].decode()
            selected_telescope_types = [default]
        if self.mode == "mono":
            self.tel_type = selected_telescope_types[0]

        # Select which telescopes from the full dataset to include in each
        # event by a telescope type and an optional list of telescope ids.
        selected_telescopes = {}
        for tel_type in selected_telescope_types:
            available_tel_ids = telescopes[tel_type]
            # Keep only the selected tel ids for the tel type
            if selected_telescope_ids:
                selected_telescopes[tel_type] = np.intersect1d(
                    available_tel_ids, selected_telescope_ids
                )
            else:
                selected_telescopes[tel_type] = available_tel_ids

        return telescopes, selected_telescopes, camera2index

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
        if self.data_model_mainversion < 4:
            cameras = [
                description.decode("UTF-8").split("_")[-1]
                for description in telescope_type_information.optics.cols._f_col(
                    "description"
                )
            ]
        else:
            cameras = self.camera2index.keys()

        pixel_positions = {}
        num_pixels = {}
        for camera in cameras:
            if not self.data_model_version.startswith("v1"):
                cam_geom = telescope_type_information.camera._f_get_child(
                    "geometry_{}".format(self.camera2index[camera])
                )
            else:
                cam_geom = telescope_type_information.camera._f_get_child(
                    "geometry_{}".format(camera)
                )
            pix_x = np.array(cam_geom.cols._f_col("pix_x"))
            pix_y = np.array(cam_geom.cols._f_col("pix_y"))
            num_pixels[camera] = len(pix_x)
            pixel_positions[camera] = np.stack((pix_x, pix_y))
            # For now hardcoded, since this information is not in the h5 files.
            # The official CTA DL1 format will contain this information.
            if camera in ["LSTCam", "LSTSiPMCam", "NectarCam", "MAGICCam"]:
                rotation_angle = -70.9 * np.pi / 180.0
                if camera == "MAGICCam":
                    rotation_angle = -100.893 * np.pi / 180.0
                if camera == "LSTSiPMCam":
                    rotation_angle = -cam_geom._v_attrs["PIX_ROT"] * np.pi / 180.0
                if self.process_type == "Observation" and camera == "LSTCam":
                    rotation_angle = -70.8935 * np.pi / 180.0
                rotation_matrix = np.matrix(
                    [
                        [np.cos(rotation_angle), -np.sin(rotation_angle)],
                        [np.sin(rotation_angle), np.cos(rotation_angle)],
                    ],
                    dtype=float,
                )
                pixel_positions[camera] = np.squeeze(
                    np.asarray(np.dot(rotation_matrix, pixel_positions[camera]))
                )

        return pixel_positions, num_pixels

    def _load_tel_type_data(self, filename, tel_type, trigger_info, pointing_info=None):
        triggers = []
        images = []
        parameters_lists = []
        pointings = []
        subarray_info = [[] for column in self.subarray_info]
        if self.split_datasets_by == "tel_id":
            for i, tel_id in enumerate(self.selected_telescopes[tel_type]):
                if self.image_channels is not None:
                    child = None
                    with lock:
                        tel_table = "tel_{:03d}".format(tel_id)
                        if (
                            tel_table
                            in self.files[filename].root.dl1.event.telescope.images
                        ):
                            child = self.files[
                                filename
                            ].root.dl1.event.telescope.images._f_get_child(tel_table)
                    images.append(super()._get_image(child, tel_type, trigger_info[i]))

                if self.parameter_list is not None:
                    child = None
                    with lock:
                        tel_table = "tel_{:03d}".format(tel_id)
                        if (
                            tel_table
                            in self.files[filename].root.dl1.event.telescope.parameters
                        ):
                            child = self.files[
                                filename
                            ].root.dl1.event.telescope.parameters._f_get_child(
                                tel_table
                            )
                    parameter_list = []
                    for parameter in self.parameter_list:
                        if trigger_info[i] != -1 and child:
                            parameter_list.append(child[trigger_info[i]][parameter])
                        else:
                            parameter_list.append(np.nan)
                    parameters_lists.append(np.array(parameter_list, dtype=np.float32))

                if self.pointing_mode == "divergent":
                    child = None
                    with lock:
                        tel_table = "tel_{:03d}".format(tel_id)
                        if (
                            tel_table
                            in self.files[
                                filename
                            ].root.dl1.monitoring.telescope.pointing
                        ):
                            child = self.files[
                                filename
                            ].root.dl1.monitoring.telescope.pointing._f_get_child(
                                tel_table
                            )
                    if child:
                        pointings.append(
                            np.array(
                                [
                                    child[pointing_info[i]]["altitude"],
                                    child[pointing_info[i]]["azimuth"],
                                ],
                                np.float32,
                            )
                        )
                    else:
                        pointings.append(np.array([np.nan, np.nan], np.float32))

        elif self.split_datasets_by == "tel_type":
            if self.image_channels is not None:
                with lock:
                    img_child = self.files[
                        filename
                    ].root.dl1.event.telescope.images._f_get_child(tel_type)
            if self.parameter_list is not None:
                with lock:
                    prmtr_child = self.files[
                        filename
                    ].root.dl1.event.telescope.parameters._f_get_child(tel_type)

            for i, tel_id in enumerate(self.selected_telescopes[tel_type]):
                if self.image_channels is not None:
                    images.append(
                        super()._get_image(img_child, tel_type, trigger_info[i])
                    )

                if self.parameter_list is not None:
                    parameter_list = []
                    for parameter in self.parameter_list:
                        parameter_list.append(
                            prmtr_child[trigger_info[i]][parameter]
                            if trigger_info[i] != 0
                            else np.nan
                        )
                    parameters_lists.append(np.array(parameter_list, dtype=np.float32))

            tel_query = "tel_id == {}".format(tel_id)
            super()._append_subarray_info(
                self.files[filename].root.configuration.instrument.subarray.layout,
                subarray_info,
                tel_query,
            )

        example = [np.array(trigger_info >= 0, np.int8)]
        if self.image_channels is not None:
            example.extend([np.stack(images)])
        if self.parameter_list is not None:
            example.extend([np.stack(parameters_lists)])
        if self.pointing_mode == "divergent":
            example.extend([np.stack(pointings)])
        example.extend([np.stack(info) for info in subarray_info])
        return example

    def __getitem__(self, idx):

        identifiers = self.example_identifiers[idx]

        # Get record for the event
        filename = list(self.files)[identifiers[0]]

        # Load the data and any selected array info
        if self.mode == "mono":
            # Get a single image
            if self.process_type == "Simulation":
                nrow, index, tel_id = identifiers[1:4]
            else:
                index, tel_id = identifiers[1:3]

            example = []
            if self.split_datasets_by == "tel_id":
                if self.image_channels is not None:
                    with lock:
                        tel_table = "tel_{:03d}".format(tel_id)
                        child = self.files[
                            filename
                        ].root.dl1.event.telescope.images._f_get_child(tel_table)
                    example.append(super()._get_image(child, self.tel_type, index))

                if self.parameter_list is not None:
                    with lock:
                        tel_table = "tel_{:03d}".format(tel_id)
                        child = self.files[
                            filename
                        ].root.dl1.event.telescope.parameters._f_get_child(tel_table)
                    parameter_list = list(child[index][self.parameter_list])
                    example.extend([np.stack(parameter_list)])
            elif self.split_datasets_by == "tel_type":
                if self.image_channels is not None:
                    with lock:
                        child = self.files[
                            filename
                        ].root.dl1.event.telescope.images._f_get_child(self.tel_type)
                    example.append(super()._get_image(child, self.tel_type, index))

                if self.parameter_list is not None:
                    with lock:
                        child = self.files[
                            filename
                        ].root.dl1.event.telescope.parameters._f_get_child(
                            self.tel_type
                        )
                    parameter_list = list(child[index][self.parameter_list])
                    example.extend([np.stack(parameter_list)])

            subarray_info = [[] for column in self.subarray_info]
            tel_query = "tel_id == {}".format(tel_id)
            super()._append_subarray_info(
                self.files[filename].root.configuration.instrument.subarray.layout,
                subarray_info,
                tel_query,
            )
            example.extend([np.stack(info) for info in subarray_info])

            if self.pointing_mode == "subarray":
                pointing_info = identifiers[4]
                with lock:
                    subarray_pointing = self.files[
                        filename
                    ].root.dl1.monitoring.subarray.pointing
                example.append(
                    np.array(
                        [
                            subarray_pointing[pointing_info]["array_altitude"],
                            subarray_pointing[pointing_info]["array_azimuth"],
                        ],
                        np.float32,
                    )
                )
            elif self.pointing_mode == "divergent":
                pointing_info = identifiers[4]
                with lock:
                    tel_table = "tel_{:03d}".format(tel_id)
                    if (
                        tel_table
                        in self.files[filename].root.dl1.monitoring.telescope.pointing
                    ):
                        child = self.files[
                            filename
                        ].root.dl1.monitoring.telescope.pointing._f_get_child(tel_table)
                example.append(
                    np.array(
                        [
                            child[pointing_info]["altitude"],
                            child[pointing_info]["azimuth"],
                        ],
                        np.float32,
                    )
                )

        elif self.mode == "stereo":
            # Get a list of images and/or image parameters, an array of binary trigger values and telescope pointings
            # for each selected telescope type
            pointing_info = None
            if self.process_type == "Simulation":
                nrow = identifiers[1]
                trigger_info = identifiers[2]
                if self.pointing_mode == "divergent":
                    pointing_info = identifiers[3]
            else:
                trigger_info = identifiers[1]
                if self.pointing_mode == "divergent":
                    pointing_info = identifiers[2]

            example = []
            for ind, tel_type in enumerate(self.selected_telescopes):
                tel_type_example = self._load_tel_type_data(
                    filename, tel_type, trigger_info[ind], pointing_info
                )
                example.extend(tel_type_example)

            if self.pointing_mode == "subarray":
                pointing_info = identifiers[3]
                with lock:
                    subarray_pointing = self.files[
                        filename
                    ].root.dl1.monitoring.subarray.pointing
                example.append(
                    np.array(
                        [
                            subarray_pointing[pointing_info]["array_altitude"],
                            subarray_pointing[pointing_info]["array_azimuth"],
                        ],
                        np.float32,
                    )
                )

        # Load event info
        if self.process_type == "Simulation":
            with lock:
                events = self.files[filename].root.simulation.event.subarray.shower
                for column in self.event_info:
                    dtype = events.cols._f_col(column).dtype
                    example.append(np.array(events[nrow][column], dtype=dtype))

        # Preprocess the example
        example = self.processor.process(example)

        return example


class DL1DataReaderDL1DH(DL1DataReader):
    def __init__(
        self,
        file_list,
        example_identifiers_file=None,
        mode="stereo",
        pointing_mode="subarray",
        selected_telescope_types=None,
        selected_telescope_ids=None,
        parameter_table=0,
        selection_string=None,
        multiplicity_selection=None,
        event_selection=None,
        parameter_selection=None,
        image_selection=None,
        shuffle=False,
        seed=None,
        image_channels=None,
        mapping_settings=None,
        parameter_list=None,
        subarray_info=None,
        event_info=None,
        transforms=None,
        validate_processor=False,
    ):

        super().__init__(
            file_list=file_list,
            mode=mode,
            subarray_info=subarray_info,
            event_info=event_info,
        )

        first_file = list(self.files)[0]
        self.data_model_version = "dl1dh_v" + self._v_attrs["dl1_data_handler_version"]
        self.process_type = (
            "Simulation" if "corsika_version" in self._v_attrs else "Observation"
        )
        self.instrument_id = (
            "MAGIC"
            if self.files[first_file]
            .root.Array_Information[0]["type"]
            .decode()
            .split("_")[1]
            == "MAGIC"
            else "CTA"
        )

        # Set pointing mode
        # Fix_subarray: Fix subarray pointing (MC production)
        # Subarray: Subarray pointing with different pointing over time (Operation or MC production with different pointing)
        if pointing_mode in ["fix_subarray", "subarray"]:
            self.pointing_mode = pointing_mode
        else:
            raise ValueError(
                "Invalid pointing mode selection '{}'. Valid options: "
                "'fix_subarray', 'subarray'".format(pointing_mode)
            )

        # Set the number of the parameter table
        self.parameter_table = parameter_table

        self.example_identifiers = None
        self.telescopes = {}
        if selected_telescope_ids is None:
            selected_telescope_ids = []

        if multiplicity_selection is None:
            multiplicity_selection = {}

        if event_selection is None:
            event_selection = {}

        if image_selection is None:
            image_selection = {}

        if mapping_settings is None:
            mapping_settings = {}

        self.image_scale = None
        self.peak_time_scale = None
        self.simulation_info = None
        self.simulated_particles = {}
        self.simulated_particles["total"] = 0
        self.example_identifiers = None
        if example_identifiers_file is None:
            example_identifiers_file = {}
        else:
            example_identifiers_file = pd.HDFStore(example_identifiers_file)

        if "/example_identifiers" in list(example_identifiers_file.keys()):
            self.example_identifiers = pd.read_hdf(
                example_identifiers_file, key="/example_identifiers"
            ).to_numpy()
            if "/simulation_info" in list(example_identifiers_file.keys()):
                self.simulation_info = pd.read_hdf(
                    example_identifiers_file, key="/simulation_info"
                ).to_dict("records")[0]
            if "/simulated_particles" in list(example_identifiers_file.keys()):
                self.simulated_particles = pd.read_hdf(
                    example_identifiers_file, key="/simulated_particles"
                ).to_dict("records")[0]
            if "/class_weight" in list(example_identifiers_file.keys()):
                self.class_weight = pd.read_hdf(
                    example_identifiers_file, key="/class_weight"
                ).to_dict("records")[0]
            if "/class_names" in list(example_identifiers_file.keys()):
                class_names = pd.read_hdf(
                    example_identifiers_file, key="/class_names"
                ).to_dict("records")
                self.class_names = [name[0] for name in class_names]
            if "/shower_primary_id_to_class" in list(example_identifiers_file.keys()):
                self.shower_primary_id_to_class = pd.read_hdf(
                    example_identifiers_file, key="/shower_primary_id_to_class"
                ).to_dict("records")[0]
            self.num_classes = len(self.simulated_particles) - 1
            (
                self.telescopes,
                self.selected_telescopes,
                cut_condition,
            ) = self._construct_telescopes_selection(
                self.files[first_file].root.Array_Information,
                selected_telescope_types,
                selected_telescope_ids,
                selection_string,
            )
        else:

            for file_idx, (filename, f) in enumerate(self.files.items()):

                if self.process_type == "Simulation":
                    self.simulation_info = super()._construct_simulated_info(
                        f, self.simulation_info, file_type="dl1dh"
                    )

                # Teslecope selection
                (
                    telescopes,
                    selected_telescopes,
                    cut_condition,
                ) = self._construct_telescopes_selection(
                    f.root.Array_Information,
                    selected_telescope_types,
                    selected_telescope_ids,
                    selection_string,
                )

                # Multiplicity selection
                if "Subarray" not in multiplicity_selection:
                    multiplicity_selection["Subarray"] = 1
                    for tel_type in selected_telescopes:
                        if tel_type in multiplicity_selection:
                            multiplicity_selection["Subarray"] = multiplicity_selection[
                                tel_type
                            ]
                if len(selected_telescopes) > 1:
                    for tel_type in selected_telescopes:
                        if tel_type not in multiplicity_selection:
                            multiplicity_selection[tel_type] = 0
                else:
                    multiplicity_selection[
                        list(selected_telescopes.keys())[0]
                    ] = multiplicity_selection["Subarray"]

                # Event selection
                selected_nrows = set(
                    [row.nrow for row in f.root.Events.where(cut_condition)]
                )
                selected_nrows &= self._select_event(f, event_selection)
                selected_nrows = list(selected_nrows)

                # Image & parameter selection
                # Make list of identifiers of all examples passing event selection
                if self.mode == "mono":
                    example_identifiers = []
                    field = "{}_indices".format(self.tel_type)
                    selected_indices = f.root.Events.read_coordinates(
                        selected_nrows, field=field
                    )
                    for tel_id in selected_telescopes[self.tel_type]:
                        tel_index = telescopes[self.tel_type].index(tel_id)
                        img_ids = np.array(selected_indices[:, tel_index])
                        mask = img_ids != 0
                        if parameter_selection:
                            parameters = f.root[
                                "Parameters" + str(self.parameter_table)
                            ][self.tel_type][img_ids[mask]]
                            parameter_mask = np.full(len(parameters), True)
                            for filter in parameter_selection:
                                selected_parameter = parameters[filter["col_name"]]
                                if "min_value" in filter:
                                    parameter_mask &= (
                                        selected_parameter >= filter["min_value"]
                                    )
                                if "max_value" in filter:
                                    parameter_mask &= (
                                        selected_parameter < filter["max_value"]
                                    )
                            mask[mask] &= parameter_mask

                        # TODO handle all selected channels
                        mask[mask] &= self._select_image(
                            f.root["Images"][self.tel_type][img_ids[mask]]["image"],
                            image_selection,
                        )
                        for image_index, nrow in zip(
                            img_ids[mask], np.array(selected_nrows)[mask]
                        ):
                            example_identifiers.append(
                                (file_idx, nrow, image_index, tel_id)
                            )

                elif self.mode == "stereo":
                    example_identifiers = []
                    image_indices = {}
                    subarray_triggers = np.zeros(len(selected_nrows))
                    for tel_type in selected_telescopes:
                        field = "{}_indices".format(tel_type)
                        selected_indices = f.root.Events.read_coordinates(
                            selected_nrows, field=field
                        )
                        triggers = np.zeros(len(selected_indices))
                        img_indices = []
                        for tel_id in selected_telescopes[tel_type]:
                            tel_index = telescopes[tel_type].index(tel_id)
                            img_ids = np.array(selected_indices[:, tel_index])
                            mask = img_ids != 0
                            if parameter_selection:
                                parameters = f.root[
                                    "Parameters" + str(self.parameter_table)
                                ][tel_type][img_ids[mask]]
                                parameter_mask = np.full(len(parameters), True)
                                for filter in parameter_selection:
                                    selected_parameter = parameters[filter["col_name"]]
                                    if "min_value" in filter:
                                        parameter_mask &= (
                                            selected_parameter >= filter["min_value"]
                                        )
                                    if "max_value" in filter:
                                        parameter_mask &= (
                                            selected_parameter < filter["max_value"]
                                        )
                                mask[mask] &= parameter_mask

                            # TODO handle all selected channels
                            mask[mask] &= self._select_image(
                                f.root["Images"][tel_type][img_ids[mask]]["image"],
                                image_selection,
                            )

                            mask = mask.astype(int)
                            triggers += mask
                            img_indices.append(list(img_ids * mask))

                        if multiplicity_selection[tel_type] > 0:
                            trigger_mask = triggers >= multiplicity_selection[tel_type]
                            triggers = triggers[trigger_mask]
                            selected_nrows = np.array(selected_nrows)[trigger_mask]
                            subarray_triggers = subarray_triggers[trigger_mask]
                            for i, img_ind in enumerate(img_indices):
                                img_indices[i] = np.array(img_ind)[trigger_mask]
                        image_indices[tel_type] = np.array(img_indices).T
                        subarray_triggers += triggers

                    if (
                        len(selected_telescopes) > 1
                        and multiplicity_selection["Subarray"] > 1
                    ):
                        subarray_trigger_mask = (
                            subarray_triggers >= multiplicity_selection["Subarray"]
                        )
                        selected_nrows = np.array(selected_nrows)[subarray_trigger_mask]
                        for tel_type in selected_telescopes:
                            image_indices[tel_type] = np.array(image_indices[tel_type])[
                                subarray_trigger_mask
                            ]

                    for idx, nrow in enumerate(selected_nrows):
                        image_index = []
                        for tel_type in selected_telescopes:
                            image_index.append(image_indices[tel_type][idx])
                        example_identifiers.append((file_idx, nrow, image_index))

                # Track number of events for each particle type
                true_shower_primary_id = f.root.Events.cols._f_col(
                    "true_shower_primary_id"
                )[0]
                self.simulated_particles["total"] += len(example_identifiers)
                if true_shower_primary_id in self.simulated_particles:
                    self.simulated_particles[true_shower_primary_id] += len(
                        example_identifiers
                    )
                else:
                    self.simulated_particles[true_shower_primary_id] = len(
                        example_identifiers
                    )

                # Confirm that the files are consistent and merge them
                if not self.telescopes:
                    self.telescopes = telescopes
                if self.telescopes != telescopes:
                    raise ValueError(
                        "Inconsistent telescope definition in " "{}".format(filename)
                    )
                self.selected_telescopes = selected_telescopes

                if self.example_identifiers is None:
                    self.example_identifiers = example_identifiers
                else:
                    self.example_identifiers.extend(example_identifiers)

            # Handling the particle ids automatically and class weights calculation
            # Scaling by total/2 helps keep the loss to a similar magnitude.
            # The sum of the weights of all examples stays the same.
            self.num_classes = len(self.simulated_particles) - 1
            if self.process_type == "Simulation":
                if len(self.simulated_particles) > 2:
                    self.shower_primary_id_to_class = {}
                    self.class_names = []
                    for p, particle_id in enumerate(
                        list(self.simulated_particles.keys())[1:]
                    ):
                        self.shower_primary_id_to_class[particle_id] = p
                        self.class_names.append(
                            (self.shower_primary_id_to_name[particle_id])
                        )

                    self.class_weight = {}
                    for particle_id, num_particles in self.simulated_particles.items():
                        if particle_id != "total":
                            self.class_weight[
                                self.shower_primary_id_to_class[particle_id]
                            ] = (1 / num_particles) * (
                                self.simulated_particles["total"] / 2.0
                            )

            # Shuffle the examples
            if shuffle:
                random.seed(seed)
                random.shuffle(self.example_identifiers)

            # Dump example_identifiers and simulation_info to a pandas hdf5 file
            if not isinstance(example_identifiers_file, dict):
                pd.DataFrame(data=self.example_identifiers).to_hdf(
                    example_identifiers_file, key="example_identifiers", mode="a"
                )
                if self.simulation_info:
                    pd.DataFrame(
                        data=pd.DataFrame(self.simulation_info, index=[0])
                    ).to_hdf(example_identifiers_file, key="simulation_info", mode="a")
                if self.simulated_particles:
                    pd.DataFrame(
                        data=pd.DataFrame(self.simulated_particles, index=[0])
                    ).to_hdf(
                        example_identifiers_file, key="simulated_particles", mode="a"
                    )
                    if self.class_weight:
                        pd.DataFrame(
                            data=pd.DataFrame(self.class_weight, index=[0])
                        ).to_hdf(example_identifiers_file, key="class_weight", mode="a")
                        pd.DataFrame(data=pd.DataFrame(self.class_names)).to_hdf(
                            example_identifiers_file, key="class_names", mode="a"
                        )
                        pd.DataFrame(
                            data=pd.DataFrame(
                                self.shower_primary_id_to_class, index=[0]
                            )
                        ).to_hdf(
                            example_identifiers_file,
                            key="shower_primary_id_to_class",
                            mode="a",
                        )
                example_identifiers_file.close()

        if self.pointing_mode == "fix_subarray":
            run_array_direction = self.files[first_file].root._v_attrs[
                "run_array_direction"
            ]
            self.pointing = np.array(
                [run_array_direction[1], run_array_direction[0]], np.float32
            )

        self.parameter_list = parameter_list
        self.image_channels = image_channels

        # ImageMapper (1D charges -> 2D images)
        if self.image_channels is not None:
            self.pixel_positions, self.num_pixels = self._construct_pixel_positions(
                self.files[first_file].root.Telescope_Type_Information
            )
            if "camera_types" not in mapping_settings:
                mapping_settings["camera_types"] = self.pixel_positions.keys()
            self.image_mapper = ImageMapper(
                pixel_positions=self.pixel_positions, **mapping_settings
            )

            for camera_type in mapping_settings["camera_types"]:
                self.image_mapper.image_shapes[camera_type] = (
                    self.image_mapper.image_shapes[camera_type][0],
                    self.image_mapper.image_shapes[camera_type][1],
                    len(self.image_channels),  # number of channels
                )

        super()._construct_unprocessed_example_description(
            self.files[first_file].root.Array_Information,
            self.files[first_file].root.Events,
        )

        self.processor = DL1DataProcessor(
            self.mode,
            self.unprocessed_example_description,
            transforms,
            validate_processor,
        )

        # Definition of preprocessed example
        self.example_description = self.processor.output_description

    def _construct_telescopes_selection(
        self,
        subarray_table,
        selected_telescope_types,
        selected_telescope_ids,
        selection_string,
    ):
        """
        Construct the selection of the telescopes
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
            tel_type = row["type"].decode()
            if tel_type not in telescopes:
                telescopes[tel_type] = []
            telescopes[tel_type].append(row["id"])

        # Enforce an automatic minimal telescope selection cut:
        # there must be at least one triggered telescope of a
        # selected type in the event
        # Users can include stricter cuts in the selection string
        if self.mode == "mono":
            if selected_telescope_types is None:
                # Default: use the first tel type in the file
                default = subarray_table[0]["type"].decode()
                selected_telescope_types = default
            self.tel_type = selected_telescope_types[0]
        elif self.mode == "stereo":
            if selected_telescope_types is None:
                # Default: use all tel types
                selected_telescope_types = list(telescopes)
            self.tel_type = None
        selected_tel_types = selected_telescope_types

        multiplicity_conditions = [
            "(" + tel_type + "_multiplicity > 0)" for tel_type in selected_tel_types
        ]
        tel_cut_string = "(" + " | ".join(multiplicity_conditions) + ")"
        # Combine minimal telescope cut with explicit selection cuts
        if selection_string:
            cut_condition = selection_string + " & " + tel_cut_string
        else:
            cut_condition = tel_cut_string

        # Select which telescopes from the full dataset to include in each
        # event by a telescope type and an optional list of telescope ids.
        selected_telescopes = {}
        for tel_type in selected_tel_types:
            available_tel_ids = telescopes[tel_type]
            # Keep only the selected tel ids for the tel type
            if selected_telescope_ids:
                selected_telescopes[tel_type] = np.intersect1d(
                    available_tel_ids, selected_telescope_ids
                )
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
        cameras = [x["camera"].decode() for x in telescope_type_information]
        num_pix = [x["num_pixels"] for x in telescope_type_information]
        pix_pos = [x["pixel_positions"] for x in telescope_type_information]
        pixel_positions = {}
        num_pixels = {}
        for i, camera in enumerate(cameras):
            pixel_positions[camera] = pix_pos[i][: num_pix[i]].T
            num_pixels[camera] = num_pix[i]
            # For now hardcoded, since this information is not in the h5 files.
            # The official CTA DL1 format will contain this information.
            if camera in ["LSTCam", "NectarCam", "MAGICCam"]:
                rotation_angle = (
                    -70.9 * np.pi / 180.0
                    if camera == "MAGICCam"
                    else -100.893 * np.pi / 180.0
                )
                rotation_matrix = np.matrix(
                    [
                        [np.cos(rotation_angle), -np.sin(rotation_angle)],
                        [np.sin(rotation_angle), np.cos(rotation_angle)],
                    ],
                    dtype=float,
                )
                pixel_positions[camera] = np.squeeze(
                    np.asarray(np.dot(rotation_matrix, pixel_positions[camera]))
                )

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

    def _load_tel_type_data(self, filename, nrow, tel_type, trigger_info):
        triggers = []
        images = []
        parameters_lists = []
        subarray_info = [[] for column in self.subarray_info]
        if self.image_channels is not None:
            with lock:
                img_child = self.files[filename].root["Images"]._f_get_child(tel_type)
        if self.parameter_list is not None:
            with lock:
                prmtr_child = self.files[filename].root[
                    "Parameters" + str(self.parameter_table)
                ][tel_type]

        for i, tel_id in enumerate(self.selected_telescopes[tel_type]):
            trigger = 0 if trigger_info[i] == 0 else 1
            triggers.append(trigger)

            if self.image_channels is not None:
                images.append(
                    super()._get_image(
                        img_child, tel_type, trigger_info[i], self.parameter_table
                    )
                )

            if self.parameter_list is not None:
                parameter_list = []
                for parameter in self.parameter_list:
                    parameter_val = (
                        prmtr_child[trigger_info[i]][parameter]
                        if trigger_info[i] != 0
                        else np.nan
                    )
                    parameter_list.append(parameter_val)
                parameters_lists.append(np.array(parameter_list, dtype=np.float32))

            query = "id == {}".format(tel_id)
            super()._append_subarray_info(
                self.files[filename].root.Array_Information, subarray_info, query
            )

        example = [np.array(triggers, np.int8)]
        if self.image_channels is not None:
            example.extend([np.stack(images)])
        if self.parameter_list is not None:
            example.extend([np.stack(parameters_lists)])
        example.extend([np.stack(info) for info in subarray_info])

        return example

    def __getitem__(self, idx):

        identifiers = self.example_identifiers[idx]

        # Get record for the event
        filename = list(self.files)[identifiers[0]]

        # Load the data and any selected array info
        if self.mode == "mono":
            # Get a single image
            nrow, index, tel_id = identifiers[1:4]
            example = []
            if self.image_channels is not None:
                with lock:
                    child = (
                        self.files[filename].root["Images"]._f_get_child(self.tel_type)
                    )
                example.append(
                    super()._get_image(
                        child, self.tel_type, index, self.parameter_table
                    )
                )

            if self.parameter_list is not None:
                with lock:
                    parameters = self.files[filename].root[
                        "Parameters" + str(self.parameter_table)
                    ][self.tel_type]
                parameter_list = list(parameters[index][self.parameter_list])
                example.extend([np.stack(parameter_list)])

            subarray_info = [[] for column in self.subarray_info]
            query = "id == {}".format(tel_id)
            super()._append_subarray_info(
                self.files[filename].root.Array_Information, subarray_info, query
            )
            example.extend([np.stack(info) for info in subarray_info])
        elif self.mode == "stereo":

            # Get a list of images and/or image parameters and array of binary trigger values
            # for each selected telescope type
            nrow = identifiers[1]
            trigger_info = identifiers[2]
            example = []
            for ind, tel_type in enumerate(self.selected_telescopes):
                tel_type_example = self._load_tel_type_data(
                    filename, nrow, tel_type, trigger_info[ind]
                )
                example.extend(tel_type_example)

        if self.pointing_mode == "subarray":
            with lock:
                events = self.files[filename].root.Events
                example.append(
                    np.array(
                        [
                            events[nrow]["array_pointing_alt"],
                            events[nrow]["array_pointing_az"],
                        ],
                        np.float32,
                    )
                )

        # Load event info
        with lock:
            events = self.files[filename].root.Events
            for column in self.event_info:
                dtype = events.cols._f_col(column).dtype
                example.append(np.array(events[nrow][column], dtype=dtype))

        # Preprocess the example
        example = self.processor.process(example)

        return example
