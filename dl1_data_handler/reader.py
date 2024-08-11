from abc import abstractmethod
from collections import OrderedDict
import threading
import numpy as np
import tables

from dl1_data_handler.image_mapper import ImageMapper

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import (
    Table,
    unique,
    join,  # let us merge tables horizontally
    vstack,  # and vertically
)

from ctapipe.instrument import SubarrayDescription
from ctapipe.io import read_table  # let us read full tables inside the DL1 output file


__all__ = [
    "DLDataReader",
    "DLMonoReader",
    "DLStereoReader",
    "DLImageReader",
    "DLWaveformReader",
    "DLTriggerReader",
    "get_unmapped_image",
    "get_unmapped_waveform",
    "get_mapped_triggerpatch",
]


# Get a single telescope image from a particular event, uniquely
# identified by the filename, tel_type, and image table index.
# First extract a raw 1D vector and transform it into a 2D image using a
# mapping table. When 'indexed_conv' is selected this function should
# return the unmapped vector.
def get_unmapped_image(dl1_event, image_channels, image_transforms):
    unmapped_image = np.zeros(
        shape=(
            len(dl1_event["image"]),
            len(image_channels),
        ),
        dtype=np.float32,
    )
    for i, channel in enumerate(image_channels):
        mask = dl1_event["image_mask"]
        if "image" in channel:
            unmapped_image[:, i] = dl1_event["image"]
        if "time" in channel:
            cleaned_peak_times = dl1_event["peak_time"] * mask
            unmapped_image[:, i] = (
                dl1_event["peak_time"]
                - cleaned_peak_times[np.nonzero(cleaned_peak_times)].mean()
            )
        if "clean" in channel or "mask" in channel:
            unmapped_image[:, i] *= mask
        # Apply the transform to recover orginal floating point values if the file were compressed
        if "image" in channel:
            if image_transforms["image_scale"] > 0.0:
                unmapped_image[:, i] /= image_transforms["image_scale"]
            if image_transforms["image_offset"] > 0:
                unmapped_image[:, i] -= image_transforms["image_offset"]
        if "time" in channel:
            if image_transforms["peak_time_scale"] > 0.0:
                unmapped_image[:, i] /= image_transforms["peak_time_scale"]
            if image_transforms["peak_time_offset"] > 0:
                unmapped_image[:, i] -= image_transforms["peak_time_offset"]
    return unmapped_image


# Get a single telescope waveform from a particular event, uniquely
# identified by the filename, tel_type, and waveform table index.
# First extract a raw 2D vector and transform it into a 3D waveform using a
# mapping table. When 'indexed_conv' is selected this function should
# return the unmapped vector.
def get_unmapped_waveform(
    r1_event,
    waveform_settings,
    dl1_cleaning_mask=None,
):

    unmapped_waveform = np.float32(r1_event["waveform"])
    # Check if camera has one or two gain(s) and apply selection
    if unmapped_waveform.shape[0] == 1:
        unmapped_waveform = unmapped_waveform[0]
    else:
        selected_gain_channel = r1_event["selected_gain_channel"][:, np.newaxis]
        unmapped_waveform = np.where(
            selected_gain_channel == 0, unmapped_waveform[0], unmapped_waveform[1]
        )
    if waveform_settings["waveform_scale"] > 0.0:
        unmapped_waveform /= waveform_settings["waveform_scale"]
    if waveform_settings["waveform_offset"] > 0:
        unmapped_waveform -= waveform_settings["waveform_offset"]
    waveform_max = np.argmax(np.sum(unmapped_waveform, axis=0))
    if dl1_cleaning_mask is not None:
        waveform_max = np.argmax(
            np.sum(unmapped_waveform * dl1_cleaning_mask[:, None], axis=0)
        )
    if waveform_settings["max_from_simulation"]:
        waveform_max = int((len(unmapped_waveform) / 2) - 1)

    # Retrieve the sequence around the shower maximum
    if (
        waveform_settings["sequence_max_length"] - waveform_settings["sequence_length"]
    ) < 0.001:
        waveform_start = 0
        waveform_stop = waveform_settings["sequence_max_length"]
    else:
        waveform_start = 1 + waveform_max - waveform_settings["sequence_length"] / 2
        waveform_stop = 1 + waveform_max + waveform_settings["sequence_length"] / 2
        if waveform_stop > waveform_settings["sequence_max_length"]:
            waveform_start -= waveform_stop - waveform_settings["sequence_max_length"]
            waveform_stop = waveform_settings["sequence_max_length"]
        if waveform_start < 0:
            waveform_stop += np.abs(waveform_start)
            waveform_start = 0

    # Apply the DL1 cleaning mask if selected
    if "clean" in waveform_settings["type"] or "mask" in waveform_settings["type"]:
        unmapped_waveform *= dl1_cleaning_mask[:, None]

    # Crop the unmapped waveform in samples
    return unmapped_waveform[:, int(waveform_start) : int(waveform_stop)]


# Get a single telescope waveform from a particular event, uniquely
# identified by the filename, tel_type, and waveform table index.
# First extract a raw 2D vector and transform it into a 3D waveform using a
# mapping table. When 'indexed_conv' is selected this function should
# return the unmapped vector.
def get_mapped_triggerpatch(
    r0_event,
    waveform_settings,
    trigger_settings,
    image_mapper,
    camera_type,
    true_image=None,
    random_trigger_patch=False,
    trg_pixel_id=None,
    trg_waveform_sample_id=None,
):
    waveform = np.zeros(
        shape=(
            waveform_settings["shapes"][camera_type][0],
            waveform_settings["shapes"][camera_type][1],
            waveform_settings["sequence_length"],
        ),
        dtype=np.float16,
    )

    # Retrieve the true image if the child of the simulated images are provided
    mapped_true_image, trigger_patch_true_image_sum = None, None
    if true_image is not None:
        mapped_true_image = image_mapper.map_image(true_image, camera_type)

    vector = r0_event["waveform"][0]

    waveform_max = np.argmax(np.sum(vector, axis=0))

    if waveform_settings["max_from_simulation"]:
        waveform_max = int((len(vector) / 2) - 1)
    if trg_waveform_sample_id is not None:
        waveform_max = trg_waveform_sample_id

    # Retrieve the sequence around the shower maximum and calculate the pedestal
    # level per pixel outside that sequence if R0-pedsub is selected and FADC
    # offset is not provided from the simulation.
    pixped_nsb, nsb_sequence_length = None, None
    if "FADC_offset" in waveform_settings:
        pixped_nsb = np.full(
            (vector.shape[0],), waveform_settings["FADC_offset"], dtype=int
        )
    if (
        waveform_settings["sequence_max_length"] - waveform_settings["sequence_length"]
    ) < 0.001:
        waveform_start = 0
        waveform_stop = nsb_sequence_length = waveform_settings["sequence_max_length"]
        if trigger_settings["pedsub"] and pixped_nsb is None:
            pixped_nsb = np.sum(vector, axis=1) / nsb_sequence_length
    else:
        waveform_start = 1 + waveform_max - waveform_settings["sequence_length"] / 2
        waveform_stop = 1 + waveform_max + waveform_settings["sequence_length"] / 2
        nsb_sequence_length = (
            waveform_settings["sequence_max_length"]
            - waveform_settings["sequence_length"]
        )
        if waveform_stop > waveform_settings["sequence_max_length"]:
            waveform_start -= waveform_stop - waveform_settings["sequence_max_length"]
            waveform_stop = waveform_settings["sequence_max_length"]
            if trigger_settings["pedsub"] and pixped_nsb is None:
                pixped_nsb = (
                    np.sum(vector[:, : int(waveform_start)], axis=1)
                    / nsb_sequence_length
                )
        if waveform_start < 0:
            waveform_stop += np.abs(waveform_start)
            waveform_start = 0
            if trigger_settings["pedsub"] and pixped_nsb is None:
                pixped_nsb = (
                    np.sum(vector[:, int(waveform_stop) :], axis=1)
                    / nsb_sequence_length
                )
    if trigger_settings["pedsub"] and pixped_nsb is None:
        pixped_nsb = np.sum(vector[:, 0 : int(waveform_start)], axis=1)
        pixped_nsb += np.sum(
            vector[:, int(waveform_stop) : waveform_settings["sequence_max_length"]],
            axis=1,
        )
        pixped_nsb = pixped_nsb / nsb_sequence_length

    # Subtract the pedestal per pixel if R0-pedsub selected
    if trigger_settings["pedsub"]:
        vector = vector - pixped_nsb[:, None]

    # Crop the waveform
    vector = vector[:, int(waveform_start) : int(waveform_stop)]

    # Map the waveform snapshots through the ImageMapper
    # and transform to selected returning format
    mapped_waveform = image_mapper.map_image(vector, camera_type)

    trigger_patch_center = {}
    waveform_shape_x = waveform_settings["shapes"][camera_type][0]
    waveform_shape_y = waveform_settings["shapes"][camera_type][1]

    # There are three different ways of retrieving the trigger patches.
    # In case an external algorithm (i.e. DBScan) is used, the trigger patch
    # is found by the pixel id provided in a csv file. Otherwise, we search
    # for a hot spot, which can either be the pixel with the highest intensity
    # of the true Cherenkov image or the integrated waveform.
    if trigger_settings["get_patch_from"] == "file":
        pixid_vector = np.zeros(vector.shape)
        pixid_vector[trg_pixel_id, :] = 1
        mapped_pixid = image_mapper.map_image(pixid_vector, camera_type)
        hot_spot = np.unravel_index(
            np.argmax(mapped_pixid, axis=None),
            mapped_pixid.shape,
        )
    elif trigger_settings["get_patch_from"] == "simulation":
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
    trigger_patch_center["x"] = trigger_settings["patches_xpos"][camera_type][
        np.argmin(np.abs(trigger_settings["patches_xpos"][camera_type] - hot_spot[0]))
    ]
    trigger_patch_center["y"] = trigger_settings["patches_ypos"][camera_type][
        np.argmin(np.abs(trigger_settings["patches_ypos"][camera_type] - hot_spot[1]))
    ]
    # Select randomly if a trigger patch with (guaranteed) cherenkov signal
    # or a random trigger patch are processed
    if random_trigger_patch and mapped_true_image is not None:
        counter = 0
        while True:
            counter += 1
            n_trigger_patches = 0
            if counter < 10:
                n_trigger_patches = np.random.randint(
                    len(trigger_settings["patches"][camera_type])
                )
            random_trigger_patch_center = trigger_settings["patches"][camera_type][
                n_trigger_patches
            ]

            # Get the number of cherenkov photons in the trigger patch
            trigger_patch_true_image_sum = np.sum(
                mapped_true_image[
                    int(random_trigger_patch_center["x"] - waveform_shape_x / 2) : int(
                        random_trigger_patch_center["x"] + waveform_shape_x / 2
                    ),
                    int(random_trigger_patch_center["y"] - waveform_shape_y / 2) : int(
                        random_trigger_patch_center["y"] + waveform_shape_y / 2
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
    waveform = mapped_waveform

    # If 'indexed_conv' is selected, we only need the unmapped vector.
    if image_mapper.mapping_method[camera_type] == "indexed_conv":
        return vector, trigger_patch_true_image_sum
    return waveform, trigger_patch_true_image_sum


lock = threading.Lock()


class DLDataReader:
    def __init__(
        self,
        file_list,
        tel_types=None,
        tel_ids=None,
        mapping_settings=None,
    ):
        # Construct dict of filename:file_handle pairs
        self.files = OrderedDict()
        # Order the file_list
        file_list = np.sort(file_list)
        for filename in file_list:
            with lock:
                self.files[filename] = tables.open_file(filename, mode="r")
        self.first_file = list(self.files)[0]
        # Save the user attributes and useful information retrieved from the first file as a reference
        self._v_attrs = self.files[self.first_file].root._v_attrs
        self.process_type = self._v_attrs["CTA PROCESS TYPE"]
        self.data_format_version = self._v_attrs["CTA PRODUCT DATA MODEL VERSION"]
        self.instrument_id = self._v_attrs["CTA INSTRUMENT ID"]

        # Check for the minimum ctapipe data format version (v6.0.0)
        if int(self.data_format_version.split(".")[0].replace("v", "")) < 6:
            raise IOError(
                f"Provided ctapipe data format version is '{self.data_format_version}' (must be >= v.6.0.0)."
            )
        # Check for real data processing that only a single file is provided.
        if self.process_type == "Observation" and len(self.files) != 1:
            raise ValueError(
                f"When processing real observational data, please provide a single file (currently: '{len(self.files)}')."
            )

        self.subarray = SubarrayDescription.from_hdf(self.first_file)
        selected_tel_ids = None
        if tel_ids is not None:
            selected_tel_ids = np.array(tel_ids, dtype=np.int16)
        else:
            if tel_types is not None:
                selected_tel_ids = np.ravel(
                    [
                        np.array(self.subarray.get_tel_ids_for_type(str(tel_type)))
                        for tel_type in tel_types
                    ]
                )

        # Filter subarray by selected telescopes
        if selected_tel_ids is not None:
            self.subarray = self.subarray.select_subarray(selected_tel_ids)
        self.tel_ids = self.subarray.tel_ids
        self.selected_telescopes = {}
        for tel_type in self.subarray.telescope_types:
            # If is needed here for some sims where the same tel_type is stored twice
            if str(tel_type) not in self.selected_telescopes:
                self.selected_telescopes[str(tel_type)] = np.array(
                    self.subarray.get_tel_ids_for_type(str(tel_type))
                )

        # Check if only one telescope type is selected when reading in mono mode
        if self.mode == "mono" and len(self.selected_telescopes) > 1:
            raise ValueError(
                f"Mono mode selected but multiple telescope types are provided: '{self.selected_telescopes}'."
            )
        # Check that all files have the same SubarrayDescription
        for filename in self.files:
            # Read SubarrayDescription from the new file and
            subarray = SubarrayDescription.from_hdf(filename)
            # Filter subarray by selected telescopes
            subarray = subarray.select_subarray(self.tel_ids)
            # Check if it matches the reference
            if not subarray.__eq__(self.subarray):
                raise ValueError(
                    f"Subarray description of file '{filename}' does not match the reference subarray description."
                )

        # Set the telescope type as class attribute for mono mode for convenience
        self.tel_type = None
        if self.mode == "mono":
            self.tel_type = list(self.selected_telescopes)[0]
        # Get the camera index for the different telescope types
        camera2index = {}
        for t in self.subarray.tels.values():
            camera_index = self.subarray.camera_types.index(t.camera)
            if f"{t.camera.name}" not in camera2index:
                camera2index[f"{t.camera.name}"] = camera_index
        # Retrieve the camera geometry from the file
        pixel_positions = self._construct_pixel_positions(
            self.files[self.first_file].root.configuration.instrument.telescope,
            camera2index,
        )

        # Initialize the ImageMapper with the pixel positions and mapping settings
        if mapping_settings is None:
            mapping_settings = {}
        self.image_mapper = ImageMapper(
            pixel_positions=pixel_positions, **mapping_settings
        )

        # Translate from CORSIKA shower primary ID to the particle name
        self.shower_primary_id_to_name = {
            0: "gamma",
            101: "proton",
            1: "electron",
            404: "nsb",
        }

        # Telescope pointings
        self.telescope_pointings = {}
        self.fix_pointing = None
        tel_id = None
        self.tel_trigger_table = None
        if self.process_type == "Observation":
            for tel_id in self.tel_ids:
                with lock:
                    self.telescope_pointings[f"tel_{tel_id:03d}"] = read_table(
                        self.files[self.first_file],
                        f"/dl0/monitoring/telescope/pointing/tel_{tel_id:03d}",
                    )
            with lock:
                self.tel_trigger_table = read_table(
                    self.files[self.first_file],
                    "/dl1/event/telescope/trigger",
                )

        # Image parameters (DL1b)
        # Retrieve the column names for the DL1b parameter table
        with lock:
            self.dl1b_parameter_colnames = read_table(
                self.files[self.first_file],
                f"/dl1/event/telescope/parameters/tel_{self.tel_ids[0]:03d}",
            ).colnames

    def _get_camera_type(self, tel_type):
        return tel_type.split("_")[-1]

    def __len__(self):
        if self.mode == "mono":
            return len(self.example_identifiers)
        elif self.mode == "stereo":
            return len(self.unique_example_identifiers)

    def _construct_pixel_positions(self, telescope_type_information, camera2index):
        """
        Construct the pixel position of the cameras from the DL1 hdf5 file.

        # TODO: Converge further with ctapipe
        Parameters
        ----------
            telescope_type_information (tables.Table):

        Returns
        -------
        pixel_positions (dict): dictionary of `{cameras: pixel_positions}`

        """

        pixel_positions = {}
        for camera in camera2index.keys():
            cam_geom = telescope_type_information.camera._f_get_child(
                f"geometry_{camera2index[camera]}"
            )
            pix_x = np.array(cam_geom.cols._f_col("pix_x"))
            pix_y = np.array(cam_geom.cols._f_col("pix_y"))
            pixel_positions[camera] = np.stack((pix_x, pix_y))
            # For now hardcoded, since this information is not in the h5 files.
            # The official CTA DL1 format will contain this information.
            camera_prefix = camera.split("_")[0]
            if camera_prefix in ["LSTCam", "LSTSiPMCam", "NectarCam", "MAGICCam"]:
                rotation_angle = -cam_geom._v_attrs["PIX_ROT"] * np.pi / 180.0
                if camera_prefix == "MAGICCam":
                    rotation_angle = -100.893 * np.pi / 180.0
                if self.process_type == "Observation" and camera_prefix == "LSTCam":
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

        return pixel_positions

    def _get_tel_pointing(self, file, tel_ids):
        tel_pointing = []
        for tel_id in tel_ids:
            with lock:
                tel_pointing.append(
                    read_table(
                        file,
                        f"/configuration/telescope/pointing/tel_{tel_id:03d}",
                    )
                )
        return vstack(tel_pointing)

    def _transform_to_primary_class(self, table):
        # Handling the particle ids automatically and class weights calculation
        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        self.simulated_particles, self.class_weight = {}, {}
        if self.process_type == "Simulation":
            # Track number of events for each particle type
            self.simulated_particles["total"] = self.__len__()
            for primary_id in self.shower_primary_id_to_name:
                if self.mode == "mono":
                    n_particles = np.count_nonzero(
                        self.example_identifiers["true_shower_primary_id"] == primary_id
                    )
                elif self.mode == "stereo":
                    n_particles = np.count_nonzero(
                        self.unique_example_identifiers["true_shower_primary_id"]
                        == primary_id
                    )
                # Store the number of events for each particle type if there are any
                if n_particles > 0 and primary_id != 404:
                    self.simulated_particles[primary_id] = n_particles
            self.n_classes = len(self.simulated_particles) - 1
            # Include NSB patches is selected
            if self.include_nsb_patches == "auto":
                for particle_id in list(self.simulated_particles.keys())[1:]:
                    self.simulated_particles[particle_id] = int(
                        self.simulated_particles[particle_id]
                        * self.n_classes
                        / (self.n_classes + 1)
                    )
                self.simulated_particles[404] = int(
                    self.simulated_particles["total"] / (self.n_classes + 1)
                )
                self.n_classes += 1
                self._nsb_prob = np.around(1 / self.n_classes, decimals=2)
                self._shower_prob = np.around(1 - self._nsb_prob, decimals=2)

            self.shower_primary_id_to_class = {}
            for p, particle_id in enumerate(list(self.simulated_particles.keys())[1:]):
                self.shower_primary_id_to_class[particle_id] = p
            # Calculate class weights if there are more than 2 classes (particle classification task)
            if len(self.simulated_particles) > 2:
                for particle_id, n_particles in self.simulated_particles.items():
                    if particle_id != "total":
                        self.class_weight[
                            self.shower_primary_id_to_class[particle_id]
                        ] = (1 / n_particles) * (
                            self.simulated_particles["total"] / 2.0
                        )

        # Transform shower primary id to class
        # Create a vectorized function to map the values
        vectorized_map = np.vectorize(self.shower_primary_id_to_class.get)
        # Apply the mapping to the astropy column
        true_shower_primary_class = vectorized_map(table["true_shower_primary_id"])
        table.add_column(true_shower_primary_class, name="true_shower_primary_class")
        return table

    def _transform_to_log_energy(self, table):
        # Transform true energy into the log space
        table.add_column(np.log10(table["true_energy"]), name="log_true_energy")
        return table

    def _transform_to_spherical_offsets(self, table):
        # Transform alt and az into spherical offsets
        # Set the telescope pointing of the SkyOffsetSeparation tranform to the fix pointing
        fix_pointing = SkyCoord(
            table["telescope_pointing_azimuth"],
            table["telescope_pointing_altitude"],
            frame="altaz",
        )
        true_direction = SkyCoord(
            table["true_az"],
            table["true_alt"],
            frame="altaz",
        )
        sky_offset = fix_pointing.spherical_offsets_to(true_direction)
        angular_separation = fix_pointing.separation(true_direction)
        table.add_column(sky_offset[0], name="spherical_offset_az")
        table.add_column(sky_offset[1], name="spherical_offset_alt")
        table.add_column(angular_separation, name="angular_separation")
        table.remove_columns(
            [
                "telescope_pointing_azimuth",
                "telescope_pointing_altitude",
            ]
        )
        return table

    def _get_parameters(self, batch, dl1b_parameter_list) -> np.array:
        dl1b_parameters = []
        for file_idx, img_idx, tel_id in zip(
            batch["file_index"], batch["img_index"], batch["tel_id"]
        ):
            filename = list(self.files)[file_idx]
            with lock:
                tel_table = f"tel_{tel_id:03d}"
                child = self.files[
                    filename
                ].root.dl1.event.telescope.parameters._f_get_child(tel_table)
            parameters = list(child[img_idx][dl1b_parameter_list])
            dl1b_parameters.append([np.stack(parameters)])
        return np.array(dl1b_parameters)

    @abstractmethod
    def batch_generation(
        self, batch_indices, dl1b_parameter_list=None
    ) -> (dict, Table):
        pass


class DLMonoReader(DLDataReader):
    def __init__(
        self,
        file_list,
        tel_types=None,
        tel_ids=None,
        mapping_settings=None,
        quality_selection=None,
    ):

        DLDataReader.__init__(
            self,
            file_list=file_list,
            tel_types=tel_types,
            tel_ids=tel_ids,
            mapping_settings=mapping_settings,
        )

        # Columns to keep in the the example identifiers
        # This are the basic columns one need to do a
        # conventional IACT analysis with CNNs
        self.example_ids_keep_columns = ["img_index", "obs_id", "event_id", "tel_id"]
        if self.process_type == "Simulation":
            self.example_ids_keep_columns.extend(
                ["true_energy", "true_alt", "true_az", "true_shower_primary_id"]
            )

        simulation_info = []
        example_identifiers = []
        for file_idx, (filename, f) in enumerate(self.files.items()):
            if self.process_type == "Simulation":
                # Read simulation information for each observation
                simulation_info.append(read_table(f, "/configuration/simulation/run"))
                # Construct the shower simulation table
                simshower_table = read_table(f, "/simulation/event/subarray/shower")

            # Construct the table containing all events.
            # First, the telescope tables are joined with the shower simulation
            # table and then those joined/merged tables are vertically stacked.
            tel_tables = []
            for tel_id in self.selected_telescopes[self.tel_type]:
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
            events = vstack(tel_tables)

            # Initialize a boolean mask to True for all events
            self.quality_mask = np.ones(len(events), dtype=bool)
            # Quality selection based on the dl1b parameter and MC shower simulation tables
            if quality_selection:
                for filter in quality_selection:
                    # Update the mask for the minimum value condition
                    if "min_value" in filter:
                        self.quality_mask &= (
                            events[filter["col_name"]] >= filter["min_value"]
                        )
                    # Update the mask for the maximum value condition
                    if "max_value" in filter:
                        self.quality_mask &= (
                            events[filter["col_name"]] < filter["max_value"]
                        )
            # Apply the updated mask to filter events
            events = events[self.quality_mask]

            # Construct the example identifiers
            events.keep_columns(self.example_ids_keep_columns)
            tel_pointing = self._get_tel_pointing(f, self.tel_ids)
            events = join(
                left=events,
                right=tel_pointing,
                keys=["obs_id", "tel_id"],
            )
            events = DLDataReader._transform_to_spherical_offsets(self, table=events)
            # Add telescope type id which is always 0 in mono mode
            # Needed to share code with stereo reading mode
            events.add_column(file_idx, name="file_index", index=0)
            events.add_column(0, name="tel_type_id", index=3)
            example_identifiers.append(events)

        # Constrcut the example identifiers for all files
        self.example_identifiers = vstack(example_identifiers)
        # Construct simulation information for all files
        if self.process_type == "Simulation":
            self.simulation_info = vstack(simulation_info)
        # Add index column to the example identifiers to later retrieve batches
        # using the loc functionality
        self.example_identifiers.add_column(
            np.arange(len(self.example_identifiers)), name="index", index=0
        )
        self.example_identifiers.add_index("index")
        # Apply common transformation of MC data
        # Transform shower primary id to class
        self.example_identifiers = DLDataReader._transform_to_primary_class(
            self, table=self.example_identifiers
        )
        # Transform true energy into the log space
        self.example_identifiers = DLDataReader._transform_to_log_energy(
            self, table=self.example_identifiers
        )

    def batch_generation(self, batch_indices, dl1b_parameter_list=None):
        "Generates data containing batch_size samples"
        # Retrieve the batch from the example identifiers via indexing
        batch = self.example_identifiers.loc[batch_indices]
        # Retrieve the features from child classes
        features, batch = self._get_features(batch)
        # Retrieve the dl1b parameters if requested
        if dl1b_parameter_list is not None:
            features["parameters"] = DLDataReader._get_parameters(
                batch,
                dl1b_parameter_list,
            )
        return features, batch

    @abstractmethod
    def _get_features(self, batch) -> (dict, Table):
        pass


class DLStereoReader(DLDataReader):
    def __init__(
        self,
        file_list,
        tel_types=None,
        tel_ids=None,
        mapping_settings=None,
        quality_selection=None,
        multiplicity_selection=None,
    ):
        DLDataReader.__init__(
            self,
            file_list=file_list,
            tel_types=tel_types,
            tel_ids=tel_ids,
            mapping_settings=mapping_settings,
        )

        # Columns to keep in the the example identifiers
        # This are the basic columns one need to do a
        # conventional IACT analysis with CNNs
        self.example_ids_keep_columns = [
            "img_index",
            "obs_id",
            "event_id",
            "tel_id",
            "hillas_intensity",
        ]
        if self.process_type == "Simulation":
            self.example_ids_keep_columns.extend(
                ["true_energy", "true_alt", "true_az", "true_shower_primary_id"]
            )
        elif self.process_type == "Observation":
            self.example_ids_keep_columns.extend(["time", "event_type"])

        simulation_info = []
        example_identifiers = []
        for file_idx, (filename, f) in enumerate(self.files.items()):
            if self.process_type == "Simulation":
                # Read simulation information for each observation
                simulation_info.append(read_table(f, "/configuration/simulation/run"))
                # Construct the shower simulation table
                simshower_table = read_table(f, "/simulation/event/subarray/shower")
            # Read the trigger table.
            trigger_table = read_table(f, "/dl1/event/subarray/trigger")
            if self.process_type == "Simulation":
                # The shower simulation table is joined with the subarray trigger table.
                trigger_table = join(
                    left=trigger_table,
                    right=simshower_table,
                    keys=["obs_id", "event_id"],
                )
            events = []
            for tel_type_id, tel_type in enumerate(self.selected_telescopes):
                table_per_type = []
                for tel_id in self.selected_telescopes[tel_type]:
                    # The telescope table is joined with the selected and merged table.
                    tel_table = read_table(
                        f,
                        f"/dl1/event/telescope/parameters/tel_{tel_id:03d}",
                    )
                    tel_table.add_column(
                        np.arange(len(tel_table)), name="img_index", index=0
                    )
                    # Initialize a boolean mask to True for all events
                    quality_mask = np.ones(len(tel_table), dtype=bool)
                    # Quality selection based on the dl1b parameter and MC shower simulation tables
                    if quality_selection:
                        for filter in quality_selection:
                            # Update the mask for the minimum value condition
                            if "min_value" in filter:
                                quality_mask &= (
                                    tel_table[filter["col_name"]] >= filter["min_value"]
                                )
                            # Update the mask for the maximum value condition
                            if "max_value" in filter:
                                quality_mask &= (
                                    tel_table[filter["col_name"]] < filter["max_value"]
                                )
                    # Merge the telescope table with the trigger table
                    merged_table = join(
                        left=tel_table[quality_mask],
                        right=trigger_table,
                        keys=["obs_id", "event_id"],
                    )
                    table_per_type.append(merged_table)
                table_per_type = vstack(table_per_type)
                table_per_type = table_per_type.group_by(["obs_id", "event_id"])
                table_per_type.keep_columns(self.example_ids_keep_columns)
                if self.process_type == "Simulation":
                    tel_pointing = self._get_tel_pointing(f, self.tel_ids)
                    table_per_type = join(
                        left=table_per_type,
                        right=tel_pointing,
                        keys=["obs_id", "tel_id"],
                    )
                    table_per_type = self._transform_to_spherical_offsets(
                        self, table=table_per_type
                    )
                # Apply the multiplicity cut based on the telescope type
                if tel_type in multiplicity_selection:
                    table_per_type = table_per_type.group_by(["obs_id", "event_id"])

                    def _multiplicity_cut_tel_type(table, key_colnames):
                        return len(table) >= multiplicity_selection[tel_type]

                    table_per_type = table_per_type.groups.filter(
                        _multiplicity_cut_tel_type
                    )
                table_per_type.add_column(tel_type_id, name="tel_type_id", index=3)
                events.append(table_per_type)
            events = vstack(events)
            # Apply the multiplicity cut based on the subarray
            if "Subarray" in multiplicity_selection:
                events = events.group_by(["obs_id", "event_id"])

                def _multiplicity_cut_subarray(table, key_colnames):
                    return len(table) >= multiplicity_selection["Subarray"]

                events = events.groups.filter(_multiplicity_cut_subarray)
            events.add_column(file_idx, name="file_index", index=0)
            example_identifiers.append(events)

        # Constrcut the example identifiers for all files
        self.example_identifiers = vstack(example_identifiers)
        # Construct simulation information for all files
        if self.process_type == "Simulation":
            self.simulation_info = vstack(simulation_info)
        # Unique example identifiers by events
        self.unique_example_identifiers = unique(
            self.example_identifiers, keys=["obs_id", "event_id"]
        )
        # Workaround for the missing multicolumn indexing in astropy:
        # Need this PR https://github.com/astropy/astropy/pull/15826
        # waiting astropy v7.0.0
        # self.example_identifiers.add_index(["obs_id", "event_id"])

        # Apply common transformation of MC data
        # Transform shower primary id to class
        self.simulated_particles, self.class_weight = {}, {}
        self.example_identifiers = DLDataReader._transform_to_primary_class(
            self, table=self.example_identifiers
        )
        # Transform true energy into the log space
        self.example_identifiers = DLDataReader._transform_to_log_energy(
            self, table=self.example_identifiers
        )

    def batch_generation(self, batch_indices, dl1b_parameter_list=None):
        "Generates data containing batch_size samples"
        # Retrieve the batch from the example identifiers via groupd by
        # Workaround for the missing multicolumn indexing in astropy:
        # Need this PR https://github.com/astropy/astropy/pull/15826
        # waiting astropy v7.0.0
        # Once available, the batch_gereration can be shared with "mono subclass"
        example_identifiers_grouped = self.example_identifiers.group_by(
            ["obs_id", "event_id"]
        )
        batch = example_identifiers_grouped.groups[batch_indices]
        # Sort events based on their telescope types by the hillas intensity in a given batch
        batch.sort(
            ["obs_id", "event_id", "tel_type_id", "hillas_intensity"], reverse=True
        )
        batch.sort(["obs_id", "event_id", "tel_type_id"])
        # Retrieve the features from child classes
        features, batch = self._get_features(batch)
        # Retrieve the dl1b parameters if requested
        if dl1b_parameter_list is not None:
            features["parameters"] = DLDataReader._get_parameters(
                batch,
                dl1b_parameter_list,
            )
        return features, batch

    @abstractmethod
    def _get_features(self, batch) -> (dict, Table):
        pass


class DLImageReader(DLMonoReader, DLStereoReader):
    def __init__(
        self,
        file_list,
        image_settings,
        tel_types=None,
        tel_ids=None,
        mapping_settings=None,
        mode=None,
        quality_selection=None,
        multiplicity_selection=None,
    ):

        # Set data loading mode
        # Mono: single images of one telescope type
        # Stereo: events including multiple telescope types
        if mode in ["mono", "stereo"]:
            self.mode = mode
        else:
            raise ValueError(
                f"Invalid mode selection '{mode}'. Valid options: 'mono', 'stereo'"
            )

        # temp fix
        self.include_nsb_patches = "off"
        if self.mode == "mono":
            DLMonoReader.__init__(
                self,
                file_list=file_list,
                tel_types=tel_types,
                tel_ids=tel_ids,
                quality_selection=quality_selection,
                mapping_settings=mapping_settings,
            )
        elif self.mode == "stereo":
            DLStereoReader.__init__(
                self,
                file_list=file_list,
                tel_types=tel_types,
                tel_ids=tel_ids,
                mapping_settings=mapping_settings,
                quality_selection=quality_selection,
                multiplicity_selection=multiplicity_selection,
            )

        # Integrated charges and peak arrival times (DL1a)
        self.image_channels = image_settings["image_channels"]
        for camera_type in self.image_mapper.camera_types:
            self.image_mapper.image_shapes[camera_type] = (
                self.image_mapper.image_shapes[camera_type][0],
                self.image_mapper.image_shapes[camera_type][1],
                len(self.image_channels),  # number of channels
            )
        # Get offset and scaling of images
        self.image_transforms = {}
        self.image_transforms["image_scale"] = 0.0
        self.image_transforms["image_offset"] = 0
        self.image_transforms["peak_time_scale"] = 0.0
        self.image_transforms["peak_time_offset"] = 0
        first_tel_table = f"tel_{self.tel_ids[0]:03d}"
        with lock:
            img_table_v_attrs = (
                self.files[self.first_file]
                .root.dl1.event.telescope.images._f_get_child(first_tel_table)
                ._v_attrs
            )
        # Check the transform value used for the file compression
        if "CTAFIELD_3_TRANSFORM_SCALE" in img_table_v_attrs:
            self.image_transforms["image_scale"] = img_table_v_attrs[
                "CTAFIELD_3_TRANSFORM_SCALE"
            ]
            self.image_transforms["image_offset"] = img_table_v_attrs[
                "CTAFIELD_3_TRANSFORM_OFFSET"
            ]
        if "CTAFIELD_4_TRANSFORM_SCALE" in img_table_v_attrs:
            self.image_transforms["peak_time_scale"] = img_table_v_attrs[
                "CTAFIELD_4_TRANSFORM_SCALE"
            ]
            self.image_transforms["peak_time_offset"] = img_table_v_attrs[
                "CTAFIELD_4_TRANSFORM_OFFSET"
            ]
            for camera_type in self.image_mapper.camera_types:
                self.image_mapper.image_shapes[camera_type] = (
                    self.image_mapper.image_shapes[camera_type][0],
                    self.image_mapper.image_shapes[camera_type][1],
                    len(self.image_channels),  # number of channels
                )

    def _get_features(self, batch) -> (dict, Table):
        features = {}
        images = []
        for file_idx, img_idx, tel_type_id, tel_id in zip(
            batch["file_index"],
            batch["img_index"],
            batch["tel_type_id"],
            batch["tel_id"],
        ):
            filename = list(self.files)[file_idx]
            with lock:
                tel_table = f"tel_{tel_id:03d}"
                child = self.files[
                    filename
                ].root.dl1.event.telescope.images._f_get_child(tel_table)
                unmapped_image = get_unmapped_image(
                    child[img_idx], self.image_channels, self.image_transforms
                )
            # Apply the ImageMapper whenever the mapping method is not indexed_conv
            camera_type = self._get_camera_type(
                list(self.selected_telescopes.keys())[tel_type_id]
            )
            if self.image_mapper.mapping_method[camera_type] != "indexed_conv":
                images.append(self.image_mapper.map_image(unmapped_image, camera_type))
            else:
                images.append(unmapped_image)
        features["images"] = np.array(images)
        return features, batch


class DLWaveformReader(DLMonoReader, DLStereoReader):
    def __init__(
        self,
        file_list,
        waveform_settings,
        tel_types=None,
        tel_ids=None,
        mapping_settings=None,
        mode=None,
        multiplicity_selection=None,
        quality_selection=None,
    ):

        # Set data loading mode
        # Mono: single images of one telescope type
        # Stereo: events including multiple telescope types
        if mode in ["mono", "stereo"]:
            self.mode = mode
        else:
            raise ValueError(
                f"Invalid mode selection '{mode}'. Valid options: 'mono', 'stereo'"
            )

        # temp fix
        self.include_nsb_patches = "off"
        if self.mode == "mono":
            DLMonoReader.__init__(
                self,
                file_list=file_list,
                tel_types=tel_types,
                tel_ids=tel_ids,
                mapping_settings=mapping_settings,
                quality_selection=quality_selection,
            )
        elif self.mode == "stereo":
            DLStereoReader.__init__(
                self,
                file_list=file_list,
                tel_types=tel_types,
                tel_ids=tel_ids,
                mapping_settings=mapping_settings,
                quality_selection=quality_selection,
                multiplicity_selection=multiplicity_selection,
            )

        # Calibrated waveform (R1)
        self.waveform_settings = waveform_settings
        self.waveform_type = waveform_settings["type"]

        first_tel_table = f"tel_{self.tel_ids[0]:03d}"
        with lock:
            wvf_table_v_attrs = (
                self.files[self.first_file]
                .root.r1.event.telescope._f_get_child(first_tel_table)
                ._v_attrs
            )
        self.waveform_settings["sequence_max_length"] = (
            self.files[self.first_file]
            .root.r1.event.telescope._f_get_child(first_tel_table)
            .coldescrs["waveform"]
            .shape[-1]
        )
        self.waveform_settings["waveform_scale"] = 0.0
        self.waveform_settings["waveform_offset"] = 0
        # Check the transform value used for the file compression
        if "CTAFIELD_5_TRANSFORM_SCALE" in wvf_table_v_attrs:
            self.waveform_settings["waveform_scale"] = wvf_table_v_attrs[
                "CTAFIELD_5_TRANSFORM_SCALE"
            ]
            self.waveform_settings["waveform_offset"] = wvf_table_v_attrs[
                "CTAFIELD_5_TRANSFORM_OFFSET"
            ]
        # Check that the waveform sequence length is valid
        if (
            self.waveform_settings["sequence_length"]
            > self.waveform_settings["sequence_max_length"]
        ):
            raise ValueError(
                f"Invalid sequence length '{self.waveform_settings['sequence_length']}' (must be <= '{self.waveform_settings['sequence_max_length']}')."
            )
        # Set the shapes of the waveforms
        self.waveform_settings["shapes"] = {}
        for camera_type in self.image_mapper.camera_types:
            self.image_mapper.image_shapes[camera_type] = (
                self.image_mapper.image_shapes[camera_type][0],
                self.image_mapper.image_shapes[camera_type][1],
                self.waveform_settings["sequence_length"],
            )
            self.waveform_settings["shapes"][camera_type] = (
                self.image_mapper.image_shapes[camera_type]
            )

    def _get_features(self, batch) -> (dict, Table):
        features = {}
        waveforms = []
        for file_idx, img_idx, tel_type_id, tel_id in zip(
            batch["file_index"],
            batch["img_index"],
            batch["tel_type_id"],
            batch["tel_id"],
        ):
            filename = list(self.files)[file_idx]
            with lock:
                tel_table = f"tel_{tel_id:03d}"
                child = self.files[filename].root.r1.event.telescope._f_get_child(
                    tel_table
                )
                dl1_cleaning_mask = None
                if "dl1" in self.files[filename].root:
                    if "images" in self.files[filename].root.dl1.event.telescope:
                        img_child = self.files[
                            filename
                        ].root.dl1.event.telescope.images._f_get_child(tel_table)
                        dl1_cleaning_mask = np.array(
                            img_child[img_idx]["image_mask"], dtype=int
                        )
                unmapped_waveform = get_unmapped_waveform(
                    child[img_idx],
                    self.waveform_settings,
                    dl1_cleaning_mask,
                )
            # Apply the ImageMapper whenever the mapping method is not indexed_conv
            camera_type = self._get_camera_type(
                list(self.selected_telescopes.keys())[tel_type_id]
            )
            if self.image_mapper.mapping_method[camera_type] != "indexed_conv":
                waveforms.append(
                    self.image_mapper.map_image(unmapped_waveform, camera_type)
                )
            else:
                waveforms.append(unmapped_waveform)
        features["waveforms"] = np.array(waveforms)
        return features, batch


class DLTriggerReader(DLMonoReader):
    def __init__(
        self,
        file_list,
        waveform_settings,
        trigger_settings,
        tel_types=None,
        tel_ids=None,
        mapping_settings=None,
    ):

        # Set data loading mode to mono
        self.mode = "mono"
        # AI-based trigger system settings
        self.trigger_settings = trigger_settings
        self.include_nsb_patches = self.trigger_settings["include_nsb_patches"]
        self.get_trigger_patch_from = self.trigger_settings["get_patch_from"]

        DLMonoReader.__init__(
            self,
            file_list=file_list,
            tel_types=tel_types,
            tel_ids=tel_ids,
            mapping_settings=mapping_settings,
        )

        # AI-based trigger system
        # Obtain trigger patch info from an external algorithm (i.e. DBScan)
        # TODO: Make a better iterface to read the trigger patch info
        # Either append the hdf5 file with the trigger patch info or implement
        # the DBscan algorithm here in the reader.
        if self.get_trigger_patch_from == "file":
            trigger_patch_info = []
            for filename in self.files:
                try:
                    # Read csv containing the trigger patch info
                    import pandas as pd

                    trigger_patch_info_csv_file = pd.read_csv(
                        filename.replace("r0.dl1.h5", "npe.csv")
                    )[
                        [
                            "obs_id",
                            "event_id",
                            "tel_id",
                            "trg_pixel_id",
                            "trg_waveform_sample_id",
                        ]
                    ].astype(
                        int
                    )
                    trigger_patch_info.append(
                        Table.from_pandas(trigger_patch_info_csv_file)
                    )
                except:
                    raise IOError(
                        f"There is a problem with '{filename.replace('r0.dl1.h5','npe.csv')}'!"
                    )

            # Join the events table ith the trigger patch info
            self.example_identifiers = join(
                left=vstack(trigger_patch_info),
                right=self.example_identifiers,
                keys=["obs_id", "event_id", "tel_id"],
            )
            # Remove non-trigger events from the self.example_identifiers
            # identified by negative pixel ids
            self.example_identifiers = self.example_identifiers[
                self.example_identifiers["trg_pixel_id"] >= 0
            ]

        # Raw waveform (R0)
        self.waveform_settings = waveform_settings
        self.waveform_type = waveform_settings["type"]
        first_tel_table = f"tel_{self.tel_ids[0]:03d}"
        self.waveform_settings["sequence_max_length"] = (
            self.files[self.first_file]
            .root.r0.event.telescope._f_get_child(first_tel_table)
            .coldescrs["waveform"]
            .shape[-1]
        )
        # Check that the waveform sequence length is valid
        if (
            self.waveform_settings["sequence_length"]
            > self.waveform_settings["sequence_max_length"]
        ):
            raise ValueError(
                f"Invalid sequence length '{self.waveform_settings['sequence_length']}' (must be <= '{self.waveform_settings['sequence_max_length']}')."
            )
        self.waveform_settings["shapes"] = {}
        for camera_type in self.image_mapper.camera_types:
            self.image_mapper.image_shapes[camera_type] = (
                self.image_mapper.image_shapes[camera_type][0],
                self.image_mapper.image_shapes[camera_type][1],
                self.waveform_settings["sequence_length"],
            )
            self.waveform_settings["shapes"][camera_type] = (
                self.image_mapper.image_shapes[camera_type]
            )
            self.trigger_settings["patches_xpos"] = {}
            self.trigger_settings["patches_ypos"] = {}
            # Autoset the trigger patches
            if (
                "patch_size" not in self.trigger_settings
                or "patches" not in self.trigger_settings
            ):
                trigger_patches_xpos = np.linspace(
                    0,
                    self.image_mapper.image_shapes[camera_type][0],
                    num=self.trigger_settings["number_of_patches"][0] + 1,
                    endpoint=False,
                    dtype=int,
                )[1:]
                trigger_patches_ypos = np.linspace(
                    0,
                    self.image_mapper.image_shapes[camera_type][1],
                    num=self.trigger_settings["number_of_patches"][0] + 1,
                    endpoint=False,
                    dtype=int,
                )[1:]
                self.trigger_settings["patch_size"] = {
                    camera_type: [
                        trigger_patches_xpos[0] * 2,
                        trigger_patches_ypos[0] * 2,
                    ]
                }
                self.trigger_settings["patches"] = {camera_type: []}
                for patches in np.array(
                    np.meshgrid(trigger_patches_xpos, trigger_patches_ypos)
                ).T:
                    for patch in patches:
                        self.trigger_settings["patches"][camera_type].append(
                            {"x": patch[0], "y": patch[1]}
                        )

            self.waveform_settings["shapes"][camera_type] = (
                self.trigger_settings["patch_size"][camera_type][0],
                self.trigger_settings["patch_size"][camera_type][1],
                self.waveform_settings["sequence_length"],
            )
            self.trigger_settings["patches_xpos"][camera_type] = np.unique(
                [patch["x"] for patch in trigger_settings["patches"][camera_type]]
            )
            self.trigger_settings["patches_ypos"][camera_type] = np.unique(
                [patch["y"] for patch in trigger_settings["patches"][camera_type]]
            )

    def _get_features(self, batch) -> (dict, Table):
        features = {}

        # Get the trigger patches from
        trg_pixel_id, trg_waveform_sample_id = None, None
        if self.get_trigger_patch_from == "file":
            trg_pixel_ids = batch["trg_pixel_id"]
            trg_waveform_sample_ids = batch["trg_waveform_sample_id"]
        trigger_patches, true_cherenkov_photons = [], []
        random_trigger_patch = False
        for i, (file_idx, img_idx, tel_type_id, tel_id) in enumerate(
            zip(
                batch["file_index"],
                batch["img_index"],
                batch["tel_type_id"],
                batch["tel_id"],
            )
        ):
            filename = list(self.files)[file_idx]
            if self.get_trigger_patch_from == "file":
                trg_pixel_id = trg_pixel_ids[i]
                trg_waveform_sample_id = trg_waveform_sample_ids[i]
            with lock:
                tel_table = f"tel_{tel_id:03d}"
                child = self.files[filename].root.r0.event.telescope._f_get_child(
                    tel_table
                )
                true_image = None
                if self.process_type == "Simulation":
                    if self.include_nsb_patches == "auto":
                        random_trigger_patch = np.random.choice(
                            [False, True], p=[self._shower_prob, self._nsb_prob]
                        )
                    elif self.include_nsb_patches == "all":
                        random_trigger_patch = True
                    if "images" in self.files[filename].root.simulation.event.telescope:
                        sim_child = self.files[
                            filename
                        ].root.simulation.event.telescope.images._f_get_child(tel_table)
                        true_image = np.expand_dims(
                            np.array(sim_child[img_idx]["true_image"], dtype=int),
                            axis=1,
                        )
                camera_type = self._get_camera_type(
                    list(self.selected_telescopes.keys())[tel_type_id]
                )
                waveform, trigger_patch_true_image_sum = get_mapped_triggerpatch(
                    child[img_idx],
                    self.waveform_settings,
                    self.trigger_settings,
                    self.image_mapper,
                    camera_type,
                    true_image,
                    random_trigger_patch,
                    trg_pixel_id,
                    trg_waveform_sample_id,
                )
            trigger_patches.append(waveform)
            if trigger_patch_true_image_sum is not None:
                true_cherenkov_photons.append(trigger_patch_true_image_sum)
        features["waveforms"] = np.array(trigger_patches)
        # Add the true cherenkov photons to the batch if available
        if len(true_cherenkov_photons) > 0:
            batch.add_column(true_cherenkov_photons, name="true_cherenkov_photons")

        return features, batch
