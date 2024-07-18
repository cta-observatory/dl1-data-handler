from collections import Counter, OrderedDict
import random
import threading
import numpy as np
import pandas as pd
import tables

from dl1_data_handler.image_mapper import ImageMapper
from dl1_data_handler.processor import DL1DataProcessor

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import (
    Table,
    join,  # let us merge tables horizontally
    vstack,  # and vertically
)

from ctapipe.io import (
    read_table,
)  # let us read full tables inside the DL1 output file


__all__ = ["DLDataReader"]


lock = threading.Lock()


class DLDataReader:
    def __init__(
        self,
        file_list,
        example_identifiers_file=None,
        mode="mono",
        selected_telescope_types=None,
        selected_telescope_ids=None,
        multiplicity_selection=None,
        event_selection=None,
        parameter_selection=None,
        shuffle=False,
        seed=None,
        trigger_settings=None,
        waveform_settings=None,
        image_settings=None,
        mapping_settings=None,
        parameter_settings=None,
        subarray_info=None,
        event_info=None,
        transforms=None,
        validate_processor=False,
    ):

        # Construct dict of filename:file_handle pairs
        self.files = OrderedDict()
        # Order the file_list
        file_list = np.sort(file_list)
        for filename in file_list:
            with lock:
                self.files[filename] = tables.open_file(filename, mode="r")
        first_file = list(self.files)[0]

        # Save the user attributes and useful information retrieved from the first file
        self._v_attrs = self.files[first_file].root._v_attrs
        self.subarray_layout = self.files[
            first_file
        ].root.configuration.instrument.subarray.layout
        self.tel_ids = self.subarray_layout.cols._f_col("tel_id")
        self.process_type = self._v_attrs["CTA PROCESS TYPE"]
        self.data_format_version = self._v_attrs["CTA PRODUCT DATA MODEL VERSION"]

        # Temp fix until ctapipe can process LST-1 data writing into data format v6.0.0.
        # For dl1 images we can process real data with version v5.0.0 without any problems.
        # TODO: Remove v5.0.0 once v6.0.0 is available
        if self.process_type == "Observation" and image_settings is not None:
            if int(self.data_format_version.split(".")[0].replace("v", "")) < 5:
                raise IOError(
                    f"Provided ctapipe data format version is '{self.data_format_version}' (must be >= v.5.0.0 for LST-1 data)."
                )
        else:
            if int(self.data_format_version.split(".")[0].replace("v", "")) < 6:
                raise IOError(
                    f"Provided ctapipe data format version is '{self.data_format_version}' (must be >= v.6.0.0)."
                )
        # Add several checks for real data processing, i.e. no quality cut applied and a single file is provided.
        if self.process_type == "Observation" and parameter_selection is not None:
            raise ValueError(
                f"When processing real observational data, please do not select any quality cut (currently: '{parameter_selection}')."
            )
        if self.process_type == "Observation" and len(self.files) != 1:
            raise ValueError(
                f"When processing real observational data, please provide a single file (currently: '{len(self.files)}')."
            )
        self.subarray_shower = None
        if self.process_type == "Simulation":
            self.subarray_shower = self.files[
                first_file
            ].root.simulation.event.subarray.shower
        self.instrument_id = self._v_attrs["CTA INSTRUMENT ID"]
        # Set class weights to None
        self.class_weight = None

        # Translate from CORSIKA shower primary ID to the particle name
        self.shower_primary_id_to_name = {
            0: "gamma",
            101: "proton",
            1: "electron",
            255: "hadron",
            404: "nsb",
        }

        # Set data loading mode
        # Mono: single images of one telescope type
        # Stereo: events including multiple telescope types
        if mode in ["mono", "stereo"]:
            self.mode = mode
        else:
            raise ValueError(
                f"Invalid mode selection '{mode}'. Valid options: 'mono', 'stereo'"
            )

        if subarray_info is None:
            subarray_info = []
        self.subarray_info = subarray_info

        if event_info is None:
            event_info = []
        self.event_info = event_info

        if selected_telescope_ids is None:
            selected_telescope_ids = []
        (
            self.telescopes,
            self.selected_telescopes,
            self.camera2index,
        ) = self._construct_telescopes_selection(
            self.subarray_layout,
            selected_telescope_types,
            selected_telescope_ids,
        )

        if multiplicity_selection is None:
            multiplicity_selection = {}

        if mapping_settings is None:
            mapping_settings = {}

        # Telescope pointings
        self.telescope_pointings = {}
        self.fix_pointing = None
        tel_id = None
        self.tel_trigger_table = None
        if self.process_type == "Observation":
            for tel_id in self.tel_ids:
                with lock:
                    self.telescope_pointings[f"tel_{tel_id:03d}"] = read_table(
                        self.files[first_file],
                        f"/dl1/monitoring/telescope/pointing/tel_{tel_id:03d}",
                    )
            with lock:
                self.tel_trigger_table = read_table(
                    self.files[first_file],
                    "/dl1/event/telescope/trigger",
                )
        elif self.process_type == "Simulation":
            for tel_id in self.tel_ids:
                with lock:
                    self.telescope_pointings[f"tel_{tel_id:03d}"] = read_table(
                        self.files[first_file],
                        f"/configuration/telescope/pointing/tel_{tel_id:03d}",
                    )

            # Only fix telescope pointings valid for MCs!
            # No divergent pointing implemented!
            fix_pointing_alt = self.telescope_pointings[f"tel_{tel_id:03d}"][
                "telescope_pointing_altitude"
            ]
            fix_pointing_az = self.telescope_pointings[f"tel_{tel_id:03d}"][
                "telescope_pointing_azimuth"
            ]
            # Reading the pointing for the first obs_id assuming fix tel pointing
            fix_pointing_az = fix_pointing_az[0] * fix_pointing_az.unit
            fix_pointing_alt = fix_pointing_alt[0] * fix_pointing_alt.unit
            self.fix_pointing = SkyCoord(
                fix_pointing_az.to_value(u.deg),
                fix_pointing_alt.to_value(u.deg),
                frame="altaz",
                unit="deg",
            )
            # Set the telescope pointing of the SkyOffsetSeparation tranform to the fix pointing
            if transforms is not None:
                for transform in transforms:
                    if transform.name == "SkyOffsetSeparation":
                        transform.set_pointing(self.fix_pointing)

        # AI-based trigger system
        self.trigger_settings = trigger_settings
        self.reco_cherenkov_photons, self.include_nsb_patches = False, None
        if self.trigger_settings is not None:
            self.reco_cherenkov_photons = self.trigger_settings[
                "reco_cherenkov_photons"
            ]
            self.include_nsb_patches = self.trigger_settings["include_nsb_patches"]
            self.get_trigger_patch = self.trigger_settings["get_trigger_patch"]

        # Raw (R0) or calibrated (R1) waveform
        self.waveform_type = None
        self.waveform_scale, self.waveform_offset = 0.0, 0
        if waveform_settings is not None:
            self.waveform_type = waveform_settings["waveform_type"]
            self.waveform_max_from_simulation = waveform_settings[
                "waveform_max_from_simulation"
            ]
            if "raw" in self.waveform_type:
                first_tel_table = f"tel_{self.tel_ids[0]:03d}"
                self.waveform_sequence_max_length = (
                    self.files[first_file]
                    .root.r0.event.telescope._f_get_child(first_tel_table)
                    .coldescrs["waveform"]
                    .shape[-1]
                )
                self.waveform_r0pedsub = waveform_settings["waveform_r0pedsub"]
                self.waveform_FADC_offset = None
                if "waveform_FADC_offset" in waveform_settings:
                    self.waveform_FADC_offset = waveform_settings[
                        "waveform_FADC_offset"
                    ]
            if "calibrate" in self.waveform_type:
                first_tel_table = f"tel_{self.tel_ids[0]:03d}"
                with lock:
                    wvf_table_v_attrs = (
                        self.files[first_file]
                        .root.r1.event.telescope._f_get_child(first_tel_table)
                        ._v_attrs
                    )

                self.waveform_sequence_max_length = (
                    self.files[first_file]
                    .root.r1.event.telescope._f_get_child(first_tel_table)
                    .coldescrs["waveform"]
                    .shape[-1]
                )
                self.waveform_r0pedsub = False
                self.waveform_FADC_offset = None
                # Check the transform value used for the file compression
                if "CTAFIELD_5_TRANSFORM_SCALE" in wvf_table_v_attrs:
                    self.waveform_scale = wvf_table_v_attrs[
                        "CTAFIELD_5_TRANSFORM_SCALE"
                    ]
                    self.waveform_offset = wvf_table_v_attrs[
                        "CTAFIELD_5_TRANSFORM_OFFSET"
                    ]
            self.waveform_sequence_length = waveform_settings[
                "waveform_sequence_length"
            ]
            if self.waveform_sequence_length is None:
                self.waveform_sequence_length = self.waveform_sequence_max_length
            # Set returning format for waveforms
            self.waveform_format = waveform_settings["waveform_format"]
            if not ("first" in self.waveform_format or "last" in self.waveform_format):
                raise ValueError(
                    f"Invalid returning format for waveforms '{self.waveform_format}'. Valid options: "
                    "'timechannel_first', 'timechannel_last'"
                )

        # Integrated charges and peak arrival times (DL1a)
        self.image_channels = None
        if image_settings is not None:
            self.image_channels = image_settings["image_channels"]
        self.image_scale, self.image_offset = 0.0, 0
        self.peak_time_scale, self.peak_time_offset = 0.0, 0
        # Image parameters (DL1b)
        self.parameter_list = None
        if parameter_settings is not None:
            self.parameter_list = parameter_settings["parameter_list"]

        # Get offset and scaling of images
        if self.image_channels is not None:
            first_tel_table = f"tel_{self.tel_ids[0]:03d}"
            with lock:
                img_table_v_attrs = (
                    self.files[first_file]
                    .root.dl1.event.telescope.images._f_get_child(first_tel_table)
                    ._v_attrs
                )
            # Check the transform value used for the file compression
            if (
                "CTAFIELD_3_TRANSFORM_SCALE" in img_table_v_attrs
            ):
                self.image_scale = img_table_v_attrs["CTAFIELD_3_TRANSFORM_SCALE"]
                self.image_offset = img_table_v_attrs["CTAFIELD_3_TRANSFORM_OFFSET"]
            if "CTAFIELD_4_TRANSFORM_SCALE" in img_table_v_attrs:
                self.peak_time_scale = img_table_v_attrs["CTAFIELD_4_TRANSFORM_SCALE"]
                self.peak_time_offset = img_table_v_attrs["CTAFIELD_4_TRANSFORM_OFFSET"]

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
            if self.include_nsb_patches == "auto":
                self._nsb_prob = np.around(1 / self.num_classes, decimals=2)
                self._shower_prob = np.around(1 - self._nsb_prob, decimals=2)
            example_identifiers_file.close()
        else:
            for file_idx, (filename, f) in enumerate(self.files.items()):
                # Read simulation information from each observation needed for pyIRF
                if self.process_type == "Simulation":
                    self.simulation_info = self._construct_simulated_info(
                        f, self.simulation_info
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
                    multiplicity_selection[list(selected_telescopes.keys())[0]] = (
                        multiplicity_selection["Subarray"]
                    )

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

                    # AI-based trigger system
                    # Obtain trigger patch info from an external algorithm (i.e. DBScan)
                    if (
                        self.trigger_settings is not None
                        and "raw" in self.waveform_type
                    ):
                        if self.get_trigger_patch == "file":
                            try:
                                # Read csv containing the trigger patch info
                                trigger_patch_info_csv_file = (
                                    pd.read_csv(
                                        filename.replace("r0.dl1.h5", "npe.csv")
                                    )[["obs_id", "event_id", "tel_id", "trg_pixel_id", "trg_waveform_sample_id"]]
                                    .astype(int)
                                )
                                trigger_patch_info = Table.from_pandas(
                                    trigger_patch_info_csv_file
                                )
                                # Join the events table ith the trigger patch info
                                allevents = join(
                                    left=trigger_patch_info,
                                    right=allevents,
                                    keys=["obs_id", "event_id", "tel_id"],
                                )
                                # Remove non-trigger events with negative pixel ids
                                allevents = allevents[allevents["trg_pixel_id"] >= 0]
                            except:
                                raise IOError(
                                    f"There is a problem with '{filename.replace('r0.dl1.h5','npe.csv')}'!"
                                )

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
                                        allevents[filter["col_name"]]
                                        >= filter["min_value"]
                                    ]
                                else:
                                    allevents = allevents[
                                        allevents["camera_frame_" + filter["col_name"]]
                                        >= filter["min_value"]
                                    ]
                            if "max_value" in filter:
                                if filter["col_name"] in allevents.colnames:
                                    allevents = allevents[
                                        allevents[filter["col_name"]]
                                        < filter["max_value"]
                                    ]
                                else:
                                    allevents = allevents[
                                        allevents["camera_frame_" + filter["col_name"]]
                                        < filter["max_value"]
                                    ]

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
                        if self.trigger_settings is not None and self.get_trigger_patch == "file":
                            for (
                                sim_idx,
                                img_idx,
                                tel_id,
                                trg_pix_id,
                                trg_wvf_id,
                            ) in zip(
                                allevents["sim_index"],
                                allevents["img_index"],
                                allevents["tel_id"],
                                allevents["trg_pixel_id"],
                                allevents["trg_waveform_sample_id"],
                            ):
                                example_identifiers.append(
                                    (
                                        file_idx,
                                        sim_idx,
                                        img_idx,
                                        tel_id,
                                        trg_pix_id,
                                        trg_wvf_id,
                                    )
                                )
                        else:
                            for sim_idx, img_idx, tel_id in zip(
                                allevents["sim_index"],
                                allevents["img_index"],
                                allevents["tel_id"],
                            ):
                                example_identifiers.append(
                                    (file_idx, sim_idx, img_idx, tel_id)
                                )
                    else:
                        # Construct the example identifiers
                        if self.trigger_settings is not None and self.get_trigger_patch == "file":
                            for img_idx, tel_id, trg_pix_id, trg_wvf_id in zip(
                                allevents["img_index"],
                                allevents["tel_id"],
                                allevents["trg_pixel_id"],
                                allevents["trg_waveform_sample_id"],
                            ):
                                example_identifiers.append(
                                    (
                                        file_idx,
                                        img_idx,
                                        tel_id,
                                        trg_pix_id,
                                        trg_wvf_id,
                                    )
                                )
                        else:
                            for img_idx, tel_id in zip(
                                allevents["img_index"],
                                allevents["tel_id"],
                            ):
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
                    tel_id_to_trigger_idx = {
                        tel_id: idx for idx, tel_id in enumerate(self.tel_ids)
                    }
                    if self.process_type == "Simulation":
                        sim_indices = np.array(allevents["sim_index"], np.int32)
                    if len(selected_telescopes) > 1:
                        # Get all tel ids from the subarray
                        selection_mask = np.zeros_like(tels_with_trigger)
                        tel_ids = np.array(selected_telescopes.values())
                        for tel_id in tel_ids:
                            selection_mask[:, tel_id_to_trigger_idx[tel_id]] = 1
                        # Construct the telescope trigger information restricted to allowed telescopes
                        allowed_tels_with_trigger = tels_with_trigger * selection_mask
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
                        selection_mask = np.zeros_like(tels_with_trigger)
                        tel_ids = np.array(selected_telescopes[tel_type])
                        for tel_id in tel_ids:
                            selection_mask[:, tel_id_to_trigger_idx[tel_id]] = 1
                        # Construct the telescope trigger information restricted to allowed telescopes of this telescope type
                        allowed_tels_with_trigger = tels_with_trigger * selection_mask
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

                        for tel_id in tel_ids:
                            # Get the trigger information of this telescope
                            tel_trigger_info = selected_events_trigger[
                                :, tel_id_to_trigger_idx[tel_id]
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
                                                tel_table[
                                                    "camera_frame_" + filter["col_name"]
                                                ]
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
                                                tel_table[
                                                    "camera_frame_" + filter["col_name"]
                                                ]
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
                        if parameter_selection and multiplicity_selection[tel_type] > 0:
                            aftercuts_multiplicty_mask = (
                                np.count_nonzero(img_idx + 1, axis=1)
                                >= multiplicity_selection[tel_type]
                            )
                            img_idx = img_idx[aftercuts_multiplicty_mask]
                            event_id = event_id[aftercuts_multiplicty_mask]
                            if self.process_type == "Simulation":
                                sim_indices = sim_indices[aftercuts_multiplicty_mask]
                        image_indices[tel_type] = img_idx

                    # Apply the multiplicity cut after the parameter cuts for the subarray
                    if multiplicity_selection["Subarray"] > 1:
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
                            example_identifiers.append((file_idx, sim_idx, img_idx))
                    else:
                        # Construct the example identifiers
                        for idx in range(len(allevents)):
                            img_idx = []
                            for tel_type in selected_telescopes:
                                img_idx.append(image_indices[tel_type][idx])
                            example_identifiers.append((file_idx, img_idx))

                # Confirm that the files are consistent and merge them
                if not self.telescopes:
                    self.telescopes = telescopes
                if self.telescopes != telescopes:
                    raise ValueError(
                        f"Inconsistent telescope definition in " "{filename}"
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
                # Include NSB patches is selected
                if self.include_nsb_patches == "auto":
                    for particle_id in list(self.simulated_particles.keys())[1:]:
                        self.simulated_particles[particle_id] = int(
                            self.simulated_particles[particle_id]
                            * self.num_classes
                            / (self.num_classes + 1)
                        )
                    self.simulated_particles[404] = int(
                        self.simulated_particles["total"] / (self.num_classes + 1)
                    )
                    self.num_classes += 1
                    self._nsb_prob = np.around(1 / self.num_classes, decimals=2)
                    self._shower_prob = np.around(1 - self._nsb_prob, decimals=2)

                if (
                    len(self.simulated_particles) > 2
                    and not self.reco_cherenkov_photons
                ):
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

        # ImageMapper (1D charges -> 2D images or 3D waveforms)
        if self.image_channels is not None or self.waveform_type is not None:

            # Retrieve the camera geometry from the file
            self.pixel_positions, self.num_pixels = self._construct_pixel_positions(
                self.files[first_file].root.configuration.instrument.telescope
            )

            if "camera_types" not in mapping_settings:
                mapping_settings["camera_types"] = self.pixel_positions.keys()
            self.image_mapper = ImageMapper(
                pixel_positions=self.pixel_positions, **mapping_settings
            )

            if self.waveform_type is not None:
                self.waveform_shapes = {}
                self.trigger_patches_xpos, self.trigger_patches_ypos = {}, {}
                for camera_type in mapping_settings["camera_types"]:
                    if "first" in self.waveform_format:
                        self.image_mapper.image_shapes[camera_type] = (
                            self.image_mapper.image_shapes[camera_type][0],
                            self.image_mapper.image_shapes[camera_type][1],
                            1,
                        )
                    if "last" in self.waveform_format:
                        self.image_mapper.image_shapes[camera_type] = (
                            self.image_mapper.image_shapes[camera_type][0],
                            self.image_mapper.image_shapes[camera_type][1],
                            self.waveform_sequence_length,
                        )

                    self.waveform_shapes[camera_type] = self.image_mapper.image_shapes[
                        camera_type
                    ]

                    # AI-based trigger system
                    if (
                        self.trigger_settings is not None
                        and "raw" in self.waveform_type
                    ):
                        # Autoset the trigger patches
                        if (
                            "trigger_patch_size" not in self.trigger_settings
                            or "trigger_patches" not in self.trigger_settings
                        ):
                            trigger_patches_xpos = np.linspace(
                                0,
                                self.image_mapper.image_shapes[camera_type][0],
                                num=self.trigger_settings["number_of_trigger_patches"][
                                    0
                                ]
                                + 1,
                                endpoint=False,
                                dtype=int,
                            )[1:]
                            trigger_patches_ypos = np.linspace(
                                0,
                                self.image_mapper.image_shapes[camera_type][1],
                                num=self.trigger_settings["number_of_trigger_patches"][
                                    0
                                ]
                                + 1,
                                endpoint=False,
                                dtype=int,
                            )[1:]
                            self.trigger_settings["trigger_patch_size"] = {
                                camera_type: [
                                    trigger_patches_xpos[0] * 2,
                                    trigger_patches_ypos[0] * 2,
                                ]
                            }
                            self.trigger_settings["trigger_patches"] = {camera_type: []}
                            for patches in np.array(
                                np.meshgrid(trigger_patches_xpos, trigger_patches_ypos)
                            ).T:
                                for patch in patches:
                                    self.trigger_settings["trigger_patches"][
                                        camera_type
                                    ].append({"x": patch[0], "y": patch[1]})

                        self.waveform_shapes[camera_type] = self.trigger_settings[
                            "trigger_patch_size"
                        ][camera_type]
                        self.trigger_patches_xpos[camera_type] = np.unique(
                            [
                                patch["x"]
                                for patch in trigger_settings["trigger_patches"][
                                    camera_type
                                ]
                            ]
                        )
                        self.trigger_patches_ypos[camera_type] = np.unique(
                            [
                                patch["y"]
                                for patch in trigger_settings["trigger_patches"][
                                    camera_type
                                ]
                            ]
                        )
            if self.image_channels is not None:
                for camera_type in mapping_settings["camera_types"]:
                    self.image_mapper.image_shapes[camera_type] = (
                        self.image_mapper.image_shapes[camera_type][0],
                        self.image_mapper.image_shapes[camera_type][1],
                        len(self.image_channels),  # number of channels
                    )

        if self.process_type == "Simulation":
            self._construct_unprocessed_example_description(
                self.subarray_layout,
                self.subarray_shower,
            )
        else:
            self._construct_unprocessed_example_description(self.subarray_layout)

        self.processor = DL1DataProcessor(
            self.mode,
            self.unprocessed_example_description,
            transforms,
            validate_processor,
        )

        # Definition of preprocessed example
        self.example_description = self.processor.output_description

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
            if self.waveform_type is not None:
                if "first" in self.waveform_format:
                    self.unprocessed_example_description.append(
                        {
                            "name": "waveform",
                            "tel_type": self.tel_type,
                            "base_name": "waveform",
                            "shape": (
                                self.waveform_sequence_length,
                                self.waveform_shapes[
                                    self._get_camera_type(self.tel_type)
                                ][0],
                                self.waveform_shapes[
                                    self._get_camera_type(self.tel_type)
                                ][1],
                                1,
                            ),
                            "dtype": np.dtype(np.float16),
                        }
                    )
                if "last" in self.waveform_format:
                    self.unprocessed_example_description.append(
                        {
                            "name": "waveform",
                            "tel_type": self.tel_type,
                            "base_name": "waveform",
                            "shape": (
                                self.waveform_shapes[
                                    self._get_camera_type(self.tel_type)
                                ][0],
                                self.waveform_shapes[
                                    self._get_camera_type(self.tel_type)
                                ][1],
                                self.waveform_sequence_length,
                            ),
                            "dtype": np.dtype(np.float16),
                        }
                    )
                if self.trigger_settings is not None:
                    self.unprocessed_example_description.append(
                        {
                            "name": "trigger_patch_true_image_sum",
                            "tel_type": self.tel_type,
                            "base_name": "true_image_sum",
                            "shape": (1,),
                            "dtype": np.dtype(np.int),
                        }
                    )

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
                            "name": tel_type + "_HWtriggers",
                            "tel_type": tel_type,
                            "base_name": "HWtriggers",
                            "shape": (num_tels,),
                            "dtype": np.dtype(np.int8),
                        }
                    ]
                )
                if self.waveform_type is not None:
                    if "first" in self.waveform_format:
                        self.unprocessed_example_description.append(
                            {
                                "name": tel_type + "_waveforms",
                                "tel_type": tel_type,
                                "base_name": "waveforms",
                                "shape": (
                                    num_tels,
                                    self.waveform_sequence_length,
                                    self.waveform_shapes[
                                        self._get_camera_type(tel_type)
                                    ][0],
                                    self.waveform_shapes[
                                        self._get_camera_type(tel_type)
                                    ][1],
                                    1,
                                ),
                                "dtype": np.dtype(np.float16),
                            }
                        )
                    if "last" in self.waveform_format:
                        self.unprocessed_example_description.append(
                            {
                                "name": tel_type + "_waveforms",
                                "tel_type": tel_type,
                                "base_name": "waveforms",
                                "shape": (
                                    num_tels,
                                    self.waveform_shapes[
                                        self._get_camera_type(tel_type)
                                    ][0],
                                    self.waveform_shapes[
                                        self._get_camera_type(tel_type)
                                    ][1],
                                    self.waveform_sequence_length,
                                ),
                                "dtype": np.dtype(np.float16),
                            }
                        )
                    if self.trigger_settings is not None:
                        self.unprocessed_example_description.append(
                            {
                                "name": tel_type + "_trigger_patch_true_image_sum",
                                "tel_type": tel_type,
                                "base_name": "true_image_sum",
                                "shape": (num_tels,) + (1,),
                                "dtype": np.dtype(np.int),
                            }
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

    def _construct_simulated_info(self, file, simulation_info):
        """
        Construct the simulated_info from the DL1 hdf5 file for the pyIRF SimulatedEventsInfo table & GammaBoard.
        Parameters
        ----------
            file (hdf5 file): file containing the simulation information
            simulation_info (dict): dictionary of pyIRF simulation info

        Returns
        -------
        simulation_info (dict): updated dictionary of pyIRF simulation info

        """

        simulation_table = file.root.configuration.simulation
        runs = simulation_table._f_get_child("run")
        shower_reuse = max(np.array(runs.cols._f_col("shower_reuse")))
        n_showers = sum(np.array(runs.cols._f_col("n_showers"))) * shower_reuse
        energy_range_min = min(np.array(runs.cols._f_col("energy_range_min")))
        energy_range_max = max(np.array(runs.cols._f_col("energy_range_max")))
        max_scatter_range = max(np.array(runs.cols._f_col("max_scatter_range")))
        spectral_index = np.array(runs.cols._f_col("spectral_index"))[0]
        min_viewcone_radius = max(np.array(runs.cols._f_col("min_viewcone_radius")))
        max_viewcone_radius = max(np.array(runs.cols._f_col("max_viewcone_radius")))
        min_alt = min(np.array(runs.cols._f_col("min_alt")))
        max_alt = max(np.array(runs.cols._f_col("max_alt")))

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
    def _get_image(self, child, tel_type, image_index):
        vector = np.zeros(
            shape=(
                self.num_pixels[self._get_camera_type(tel_type)],
                len(self.image_channels),
            ),
            dtype=np.float32,
        )
        # If the telescope didn't trigger, the image index is -1 and a blank
        # image of all zeros with be loaded
        if image_index != -1 and child is not None:
            with lock:
                record = child[image_index]
                for i, channel in enumerate(self.image_channels):
                    cleaning_mask = "image_mask"
                    mask = record[cleaning_mask]
                    if "image" in channel:
                        vector[:, i] = record["image"]
                    if "time" in channel:
                        cleaned_peak_times = record["peak_time"] * mask
                        vector[:, i] = (
                            record["peak_time"]
                            - cleaned_peak_times[np.nonzero(cleaned_peak_times)].mean()
                        )
                    if "clean" in channel or "mask" in channel:
                        vector[:, i] *= mask
                    # Apply the transform to recover orginal floating point values if the file were compressed
                    if "image" in channel:
                        if self.image_scale > 0.0:
                            vector[:, i] /= self.image_scale
                        if self.image_offset > 0:
                            vector[:, i] -= self.image_offset
                    if "time" in channel:
                        if self.peak_time_scale > 0.0:
                            vector[:, i] /= self.peak_time_scale
                        if self.peak_time_offset > 0:
                            vector[:, i] -= self.peak_time_offset

        # If 'indexed_conv' is selected, we only need the unmapped vector.
        if (
            self.image_mapper.mapping_method[self._get_camera_type(tel_type)]
            == "indexed_conv"
        ):
            return vector

        image = self.image_mapper.map_image(vector, self._get_camera_type(tel_type))
        if (
            self.process_type == "Observation"
            and self._get_camera_type(tel_type) == "LSTCam"
        ):
            image = np.transpose(
                np.flip(image, axis=(0, 1)), (1, 0, 2)
            )  # x = -y & y = -x
        return image

    def _append_subarray_info(self, subarray_table, subarray_info, query):
        with lock:
            for row in subarray_table.where(query):
                for info, column in zip(subarray_info, self.subarray_info):
                    dtype = subarray_table.cols._f_col(column).dtype
                    info.append(np.array(row[column], dtype=dtype))
        return

    # Get a single telescope waveform from a particular event, uniquely
    # identified by the filename, tel_type, and waveform table index.
    # First extract a raw 2D vector and transform it into a 3D waveform using a
    # mapping table. When 'indexed_conv' is selected this function should
    # return the unmapped vector.
    def _get_waveform(
        self,
        child,
        tel_type,
        waveform_index,
        img_child=None,
        sim_child=None,
        random_trigger_patch=False,
        trg_pixel_id=None,
        trg_waveform_sample_id=None,
    ):
        vector = np.zeros(
            shape=(
                self.num_pixels[self._get_camera_type(tel_type)],
                self.waveform_sequence_length,
            ),
            dtype=np.float16,
        )

        if "first" in self.waveform_format:
            waveform = np.zeros(
                shape=(
                    self.waveform_sequence_length,
                    self.waveform_shapes[self._get_camera_type(tel_type)][0],
                    self.waveform_shapes[self._get_camera_type(tel_type)][1],
                    1,
                ),
                dtype=np.float16,
            )
        if "last" in self.waveform_format:
            waveform = np.zeros(
                shape=(
                    self.waveform_shapes[self._get_camera_type(tel_type)][0],
                    self.waveform_shapes[self._get_camera_type(tel_type)][1],
                    self.waveform_sequence_length,
                ),
                dtype=np.float16,
            )

        # Retrieve the DL1 cleaning mask if the child of the DL1 images are provided
        dl1_cleaning_mask = None
        if waveform_index != -1 and img_child is not None:
            with lock:
                dl1_cleaning_mask = np.array(
                    img_child[waveform_index]["image_mask"], dtype=int
                )

        # Retrieve the true image if the child of the simulated images are provided
        true_image, trigger_patch_true_image_sum = None, None
        if waveform_index != -1 and sim_child is not None:
            with lock:
                true_image = np.expand_dims(
                    np.array(sim_child[waveform_index]["true_image"], dtype=int), axis=1
                )
            mapped_true_image = self.image_mapper.map_image(
                true_image, self._get_camera_type(tel_type)
            )

        # If the telescope didn't trigger, the waveform index is -1 and a blank
        # waveform of all zeros with be loaded
        if waveform_index != -1 and child is not None:
            with lock:
                vector = np.float32(child[waveform_index]["waveform"])
                if "calibrate" in self.waveform_type:
                    # Check if camera has one or two gain(s) and apply selection
                    if vector.shape[0] == 1:
                        vector = vector[0]
                    else:
                        selected_gain_channel = child[waveform_index][
                            "selected_gain_channel"
                        ][:, np.newaxis]
                        vector = np.where(
                            selected_gain_channel == 0, vector[0], vector[1]
                        )

            if self.waveform_type is not None:
                if "raw" in self.waveform_type:
                    vector = vector[0]
                if "calibrate" in self.waveform_type:
                    if self.waveform_scale > 0.0:
                        vector /= self.waveform_scale
                    if self.waveform_offset > 0:
                        vector -= self.waveform_offset
                waveform_max = np.argmax(np.sum(vector, axis=0))
            if dl1_cleaning_mask is not None:
                waveform_max = np.argmax(
                    np.sum(vector * dl1_cleaning_mask[:, None], axis=0)
                )
            if self.waveform_max_from_simulation:
                waveform_max = int((len(vector) / 2) - 1)
            if trg_waveform_sample_id is not None:
                waveform_max = trg_waveform_sample_id

            # Retrieve the sequence around the shower maximum and calculate the pedestal
            # level per pixel outside that sequence if R0-pedsub is selected and FADC
            # offset is not provided from the simulation.
            pixped_nsb, nsb_sequence_length = None, None
            if self.waveform_FADC_offset is not None:
                pixped_nsb = np.full(
                    (vector.shape[0],), self.waveform_FADC_offset, dtype=int
                )
            if (
                self.waveform_sequence_max_length - self.waveform_sequence_length
            ) < 0.001:
                waveform_start = 0
                waveform_stop = nsb_sequence_length = self.waveform_sequence_max_length
                if self.waveform_r0pedsub and pixped_nsb is None:
                    pixped_nsb = np.sum(vector, axis=1) / nsb_sequence_length
            else:
                waveform_start = 1 + waveform_max - self.waveform_sequence_length / 2
                waveform_stop = 1 + waveform_max + self.waveform_sequence_length / 2
                nsb_sequence_length = (
                    self.waveform_sequence_max_length - self.waveform_sequence_length
                )
                if waveform_stop > self.waveform_sequence_max_length:
                    waveform_start -= waveform_stop - self.waveform_sequence_max_length
                    waveform_stop = self.waveform_sequence_max_length
                    if self.waveform_r0pedsub and pixped_nsb is None:
                        pixped_nsb = (
                            np.sum(vector[:, : int(waveform_start)], axis=1)
                            / nsb_sequence_length
                        )
                if waveform_start < 0:
                    waveform_stop += np.abs(waveform_start)
                    waveform_start = 0
                    if self.waveform_r0pedsub and pixped_nsb is None:
                        pixped_nsb = (
                            np.sum(vector[:, int(waveform_stop) :], axis=1)
                            / nsb_sequence_length
                        )
            if self.waveform_r0pedsub and pixped_nsb is None:
                pixped_nsb = np.sum(vector[:, 0 : int(waveform_start)], axis=1)
                pixped_nsb += np.sum(
                    vector[:, int(waveform_stop) : self.waveform_sequence_max_length],
                    axis=1,
                )
                pixped_nsb = pixped_nsb / nsb_sequence_length

            # Subtract the pedestal per pixel if R0-pedsub selected
            if self.waveform_r0pedsub:
                vector = vector - pixped_nsb[:, None]

            # Apply the DL1 cleaning mask if selected
            if "clean" in self.waveform_type or "mask" in self.waveform_type:
                vector *= dl1_cleaning_mask[:, None]

            # Crop the waveform
            vector = vector[:, int(waveform_start) : int(waveform_stop)]

            # Map the waveform snapshots through the ImageMapper
            # and transform to selected returning format
            mapped_waveform = self.image_mapper.map_image(
                vector, self._get_camera_type(tel_type)
            )
            if (
                self.process_type == "Observation"
                and self._get_camera_type(tel_type) == "LSTCam"
            ):
                mapped_waveform = np.transpose(
                    np.flip(mapped_waveform, axis=(0, 1)), (1, 0, 2)
                )  # x = -y & y = -x

            if self.trigger_settings is not None:
                trigger_patch_center = {}
                waveform_shape_x = self.waveform_shapes[
                    self._get_camera_type(tel_type)
                ][0]
                waveform_shape_y = self.waveform_shapes[
                    self._get_camera_type(tel_type)
                ][1]

                # There are three different ways of retrieving the trigger patches.
                # In case an external algorithm (i.e. DBScan) is used, the trigger patch
                # is found by the pixel id provided in a csv file. Otherwise, we search
                # for a hot spot, which can either be the pixel with the highest intensity
                # of the true Cherenkov image or the integrated waveform.
                if self.get_trigger_patch == "file":
                    pixid_vector = np.zeros(vector.shape)
                    pixid_vector[trg_pixel_id, :] = 1
                    mapped_pixid = self.image_mapper.map_image(
                        pixid_vector, self._get_camera_type(tel_type)
                    )
                    hot_spot = np.unravel_index(
                        np.argmax(mapped_pixid, axis=None),
                        mapped_pixid.shape,
                    )
                elif self.get_trigger_patch == "simulation":
                    hot_spot = np.unravel_index(
                        np.argmax(mapped_true_image, axis=None),
                        mapped_true_image.shape,
                    )
                else:
                    integrated_waveform = np.sum(mapped_waveform, axis=2)
                    hot_spot = np.unravel_index(
                        np.argmax(integrated_waveform, axis=None),
                        integrated_waveform.shape,
                    )
                # Detect in which trigger patch the hot spot is located
                trigger_patch_center["x"] = self.trigger_patches_xpos[
                    self._get_camera_type(tel_type)
                ][
                    np.argmin(
                        np.abs(
                            self.trigger_patches_xpos[self._get_camera_type(tel_type)]
                            - hot_spot[0]
                        )
                    )
                ]
                trigger_patch_center["y"] = self.trigger_patches_ypos[
                    self._get_camera_type(tel_type)
                ][
                    np.argmin(
                        np.abs(
                            self.trigger_patches_ypos[self._get_camera_type(tel_type)]
                            - hot_spot[1]
                        )
                    )
                ]
                # Select randomly if a trigger patch with (guaranteed) cherenkov signal
                # or a random trigger patch are processed
                if random_trigger_patch:
                    counter = 0
                    while True:
                        counter += 1
                        n_trigger_patches = 0
                        if counter < 10:
                            n_trigger_patches = np.random.randint(
                                len(
                                    self.trigger_settings["trigger_patches"][
                                        self._get_camera_type(tel_type)
                                    ]
                                )
                            )
                        random_trigger_patch_center = self.trigger_settings[
                            "trigger_patches"
                        ][self._get_camera_type(tel_type)][n_trigger_patches]

                        # Get the number of cherenkov photons in the trigger patch
                        trigger_patch_true_image_sum = np.sum(
                            mapped_true_image[
                                int(
                                    random_trigger_patch_center["x"]
                                    - waveform_shape_x / 2
                                ) : int(
                                    random_trigger_patch_center["x"]
                                    + waveform_shape_x / 2
                                ),
                                int(
                                    random_trigger_patch_center["y"]
                                    - waveform_shape_y / 2
                                ) : int(
                                    random_trigger_patch_center["y"]
                                    + waveform_shape_y / 2
                                ),
                                :,
                            ],
                            dtype=int,
                        )
                        if trigger_patch_true_image_sum < 1.0 or counter >= 10:
                            break
                    trigger_patch_center = random_trigger_patch_center
                else:
                    # Get the number of cherenkov photons in the trigger patch
                    trigger_patch_true_image_sum = np.sum(
                        mapped_true_image[
                            int(trigger_patch_center["x"] - waveform_shape_x / 2) : int(
                                trigger_patch_center["x"] + waveform_shape_x / 2
                            ),
                            int(trigger_patch_center["y"] - waveform_shape_y / 2) : int(
                                trigger_patch_center["y"] + waveform_shape_y / 2
                            ),
                            :,
                        ],
                        dtype=int,
                    )
                # Crop the waveform according to the trigger patch
                mapped_waveform = mapped_waveform[
                    int(trigger_patch_center["x"] - waveform_shape_x / 2) : int(
                        trigger_patch_center["x"] + waveform_shape_x / 2
                    ),
                    int(trigger_patch_center["y"] - waveform_shape_y / 2) : int(
                        trigger_patch_center["y"] + waveform_shape_y / 2
                    ),
                    :,
                ]

            if "first" in self.waveform_format:
                for index in np.arange(0, self.waveform_sequence_length, dtype=int):
                    waveform[index] = np.expand_dims(
                        mapped_waveform[:, :, index], axis=2
                    )
            if "last" in self.waveform_format:
                waveform = mapped_waveform

        # If 'indexed_conv' is selected, we only need the unmapped vector.
        if (
            self.image_mapper.mapping_method[self._get_camera_type(tel_type)]
            == "indexed_conv"
        ):
            return vector, trigger_patch_true_image_sum
        return waveform, trigger_patch_true_image_sum

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

        pixel_positions = {}
        num_pixels = {}
        for camera in self.camera2index.keys():
            cam_geom = telescope_type_information.camera._f_get_child(
                f"geometry_{self.camera2index[camera]}"
            )
            pix_x = np.array(cam_geom.cols._f_col("pix_x"))
            pix_y = np.array(cam_geom.cols._f_col("pix_y"))
            num_pixels[camera] = len(pix_x)
            pixel_positions[camera] = np.stack((pix_x, pix_y))
            # For now hardcoded, since this information is not in the h5 files.
            # The official CTA DL1 format will contain this information.
            if camera in ["LSTCam", "LSTSiPMCam", "NectarCam", "MAGICCam"]:
                rotation_angle = -cam_geom._v_attrs["PIX_ROT"] * np.pi / 180.0
                if camera == "MAGICCam":
                    rotation_angle = -100.893 * np.pi / 180.0
                if self.process_type == "Observation" and camera == "LSTCam":
                    rotation_angle = -40.89299998552154 * np.pi / 180.0
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

    def _load_tel_type_data(
        self,
        filename,
        tel_type,
        trigger_info,
        random_trigger_patch=False,
    ):
        triggers = []
        waveforms = []
        trigger_patch_true_image_sums = []
        images = []
        parameters_lists = []
        subarray_info = [[] for column in self.subarray_info]
        for i, tel_id in enumerate(self.selected_telescopes[tel_type]):
            if self.waveform_type is not None:
                if "raw" in self.waveform_type:
                    child = None
                    with lock:
                        tel_table = f"tel_{tel_id:03d}"
                        if tel_table in self.files[filename].root.r0.event.telescope:
                            child = self.files[
                                filename
                            ].root.r0.event.telescope._f_get_child(tel_table)
                            img_child = None
                            if "dl1" in self.files[filename].root:
                                if (
                                    "images"
                                    in self.files[filename].root.dl1.event.telescope
                                ):
                                    img_child = self.files[
                                        filename
                                    ].root.dl1.event.telescope.images._f_get_child(
                                        tel_table
                                    )
                            sim_child = None
                            if (
                                self.trigger_settings is not None
                                and self.process_type == "Simulation"
                            ):
                                if (
                                    "images"
                                    in self.files[
                                        filename
                                    ].root.simulation.event.telescope
                                ):
                                    sim_child = self.files[
                                        filename
                                    ].root.simulation.event.telescope.images._f_get_child(
                                        tel_table
                                    )
                    waveform, trigger_patch_true_image_sum = self._get_waveform(
                        child,
                        tel_type,
                        trigger_info[i],
                        img_child,
                        sim_child,
                        random_trigger_patch,
                    )
                    waveforms.append(waveform)
                    if trigger_patch_true_image_sum:
                        trigger_patch_true_image_sums.append(
                            trigger_patch_true_image_sum
                        )

                if "calibrate" in self.waveform_type:
                    child = None
                    with lock:
                        tel_table = f"tel_{tel_id:03d}"
                        if tel_table in self.files[filename].root.r1.event.telescope:
                            child = self.files[
                                filename
                            ].root.r1.event.telescope._f_get_child(tel_table)
                            img_child, sim_child = None, None
                            if "dl1" in self.files[filename].root:
                                if (
                                    "images"
                                    in self.files[filename].root.dl1.event.telescope
                                ):
                                    img_child = self.files[
                                        filename
                                    ].root.dl1.event.telescope.images._f_get_child(
                                        tel_table
                                    )
                    waveform, _ = self._get_waveform(
                        child,
                        tel_type,
                        trigger_info[i],
                        img_child,
                        sim_child,
                        random_trigger_patch,
                    )
                    waveforms.append(waveform)

            if self.image_channels is not None:
                child = None
                with lock:
                    tel_table = f"tel_{tel_id:03d}"
                    if (
                        tel_table
                        in self.files[filename].root.dl1.event.telescope.images
                    ):
                        child = self.files[
                            filename
                        ].root.dl1.event.telescope.images._f_get_child(tel_table)
                images.append(self._get_image(child, tel_type, trigger_info[i]))

            if self.parameter_list is not None:
                child = None
                with lock:
                    tel_table = f"tel_{tel_id:03d}"
                    if (
                        tel_table
                        in self.files[filename].root.dl1.event.telescope.parameters
                    ):
                        child = self.files[
                            filename
                        ].root.dl1.event.telescope.parameters._f_get_child(tel_table)
                parameter_list = []
                for parameter in self.parameter_list:
                    if trigger_info[i] != -1 and child:
                        parameter_list.append(child[trigger_info[i]][parameter])
                    else:
                        parameter_list.append(np.nan)
                parameters_lists.append(np.array(parameter_list, dtype=np.float32))

        example = [np.array(trigger_info >= 0, np.int8)]
        if self.waveform_type is not None:
            example.extend([np.stack(waveforms)])
            if self.reco_cherenkov_photons and "raw" in self.waveform_type:
                example.extend([np.stack(trigger_patch_true_image_sums)])
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
            random_trigger_patch = False
            if self.process_type == "Simulation":
                nrow, index, tel_id = identifiers[1:4]
                if self.include_nsb_patches == "auto":
                    random_trigger_patch = np.random.choice(
                        [False, True], p=[self._shower_prob, self._nsb_prob]
                    )
                elif self.include_nsb_patches == "all":
                    random_trigger_patch = True
            else:
                index, tel_id = identifiers[1:3]

            trg_pixel_id, trg_waveform_sample_id = None, None
            if self.trigger_settings is not None and self.get_trigger_patch == "file":
                trg_pixel_id, trg_waveform_sample_id = identifiers[-2:]

            example = []
            if self.waveform_type is not None:
                if "raw" in self.waveform_type:
                    with lock:
                        tel_table = f"tel_{tel_id:03d}"
                        child = self.files[
                            filename
                        ].root.r0.event.telescope._f_get_child(tel_table)
                        img_child = None
                        if "dl1" in self.files[filename].root:
                            if (
                                "images"
                                in self.files[filename].root.dl1.event.telescope
                            ):
                                img_child = self.files[
                                    filename
                                ].root.dl1.event.telescope.images._f_get_child(
                                    tel_table
                                )
                        sim_child = None
                        if (
                            self.trigger_settings is not None
                            and self.process_type == "Simulation"
                        ):
                            if (
                                "images"
                                in self.files[filename].root.simulation.event.telescope
                            ):
                                sim_child = self.files[
                                    filename
                                ].root.simulation.event.telescope.images._f_get_child(
                                    tel_table
                                )
                    waveform, trigger_patch_true_image_sum = self._get_waveform(
                        child,
                        self.tel_type,
                        index,
                        img_child,
                        sim_child,
                        random_trigger_patch,
                        trg_pixel_id,
                        trg_waveform_sample_id,
                    )
                    example.append(waveform)
                    if trigger_patch_true_image_sum is not None:
                        example.append(trigger_patch_true_image_sum)

                if "calibrate" in self.waveform_type:
                    with lock:
                        tel_table = f"tel_{tel_id:03d}"
                        child = self.files[
                            filename
                        ].root.r1.event.telescope._f_get_child(tel_table)
                        img_child, sim_child = None, None
                        if "dl1" in self.files[filename].root:
                            if (
                                "images"
                                in self.files[filename].root.dl1.event.telescope
                            ):
                                img_child = self.files[
                                    filename
                                ].root.dl1.event.telescope.images._f_get_child(
                                    tel_table
                                )
                    waveform, _ = self._get_waveform(
                        child,
                        self.tel_type,
                        index,
                        img_child,
                        sim_child,
                        random_trigger_patch,
                    )
                    example.append(waveform)

            if self.image_channels is not None:
                with lock:
                    tel_table = f"tel_{tel_id:03d}"
                    child = self.files[
                        filename
                    ].root.dl1.event.telescope.images._f_get_child(tel_table)
                example.append(self._get_image(child, self.tel_type, index))

            if self.parameter_list is not None:
                with lock:
                    tel_table = f"tel_{tel_id:03d}"
                    child = self.files[
                        filename
                    ].root.dl1.event.telescope.parameters._f_get_child(tel_table)
                parameter_list = list(child[index][self.parameter_list])
                example.extend([np.stack(parameter_list)])

            subarray_info = [[] for column in self.subarray_info]
            tel_query = f"tel_id == {tel_id}"
            self._append_subarray_info(
                self.files[filename].root.configuration.instrument.subarray.layout,
                subarray_info,
                tel_query,
            )
            example.extend([np.stack(info) for info in subarray_info])

        elif self.mode == "stereo":
            # Get a list of images and/or image parameters, an array of binary trigger values
            # for each selected telescope type
            random_trigger_patch = False
            if self.process_type == "Simulation":
                nrow = identifiers[1]
                trigger_info = identifiers[2]
                if self.include_nsb_patches == "auto":
                    random_trigger_patch = np.random.choice(
                        [False, True], p=[self._shower_prob, self._nsb_prob]
                    )
                elif self.include_nsb_patches == "all":
                    random_trigger_patch = True
            else:
                trigger_info = identifiers[1]

            example = []
            for ind, tel_type in enumerate(self.selected_telescopes):
                tel_type_example = self._load_tel_type_data(
                    filename,
                    tel_type,
                    trigger_info[ind],
                    random_trigger_patch,
                )
                example.extend(tel_type_example)

        # Load event info
        if self.process_type == "Simulation":
            with lock:
                events = self.files[filename].root.simulation.event.subarray.shower
                for column in self.event_info:
                    dtype = events.cols._f_col(column).dtype
                    if random_trigger_patch and column == "true_shower_primary_id":
                        example.append(np.array(404, dtype=dtype))
                    else:
                        example.append(np.array(events[nrow][column], dtype=dtype))

        # Preprocess the example
        example = self.processor.process(example)

        return example
