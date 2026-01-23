"""
This module defines the ``DLDataReader`` and ``TableQualityQuery`` classes that hold the basic reading and processing functionality for Deep Learning (DL) analyses.
"""

__all__ = [
    "ProcessType",
    "TableQualityQuery",
    "DLDataReader",
    "DLImageReader",
    "get_unmapped_image",
    "DLWaveformReader",
    "get_unmapped_waveform",
    "clean_waveform",
    "DLTriggerReader",
    "get_true_image",
    "apply_digital_sum",
    "apply_tdscan",
    "quantised_per_feb",
    "quantised_per_flower",
    "DLFeatureVectorReader",
    "get_feature_vectors",
]

from abc import abstractmethod
import atexit
from collections import OrderedDict
from enum import Enum
import numpy as np
import tables
import threading

from astropy import units as u
from astropy.coordinates.earth import EarthLocation
from astropy.coordinates import AltAz, SkyCoord
from astropy.table import (
    Table,
    unique,
    join,
    vstack,
)
from astropy.time import Time
from scipy.sparse import csr_matrix

from ctapipe.coordinates import CameraFrame, NominalFrame
from ctapipe.core import Component, QualityQuery
from ctapipe.core.traits import (
    Bool,
    Dict,
    CInt,
    Int,
    IntTelescopeParameter,
    Set,
    List,
    CaselessStrEnum,
    Unicode,
    UseEnum,
    TelescopeParameter,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.instrument.optics import FocalLengthKind
from ctapipe.io import read_table
from ctapipe.io.datalevels import DataLevel
from dl1_data_handler.image_mapper import ImageMapper

# Reference (dummy) time to insert in the SkyCoord object as the default time
LST_EPOCH = Time("2018-10-01T00:00:00", scale="utc")

lock = threading.Lock()


class ProcessType(Enum):
    Observation = "Observation"
    Simulation = "Simulation"


class TableQualityQuery(QualityQuery):
    """Quality criteria for table-wise dl1b parameters."""

    quality_criteria = List(
        default_value=[
            ("> 50 phe", "hillas_intensity > 50"),
            # ("Positive width", "hillas_width > 0"),
            # ("> 3 pixels", "morphology_n_pixels > 3"),
        ],
        allow_none=True,
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class DLDataReader(Component):
    """
    Base component for reading and processing data from ctapipe HDF5 files for Deep Learning (DL) analyses.

    This class handles the initialization and configuration of the data reader, including setting up quality criteria,
    managing input files, and extracting relevant information from the data files. It supports both observational and
    simulation data, and can operate in ``mono`` and ``stereo`` modes.

    Attributes
    ----------
    quality_query : TableQualityQuery
        An instance of TableQualityQuery to apply quality criteria to the data.
    files : OrderedDict
        A dictionary of filename:file_handle pairs for the input files.
    first_file : str
        The first file in the list of input files, which is used as reference.
    _v_attrs : dict
        Attributes and useful information retrieved from the first file.
    process_type : enum
        The type of data processing (i.e. ``ProcessType.Observation`` or ``ProcessType.Simulation``).
    data_format_version : str
        The version of the ctapipe data format.
    instrument_id : str
        The ID of the instrument.
    subarray : SubarrayDescription
        The description of the subarray.
    tel_ids : list
        List of telescope IDs in the subarray.
    selected_telescopes : dict
        Dictionary of selected telescopes by type.
    tel_type : str
        The type of telescope (used in mono mode).
    image_mappers : dict
        Dictionary of ImageMapper instances for different telescope types.
    telescope_pointings : dict
        Dictionary of telescope pointings.
    tel_trigger_table : Table
        Table of telescope triggers.
    dl1b_parameter_colnames : list
        List of all column names for the DL1b parameter table.
    example_identifiers : list
        List of example identifiers for the dataset.
    class_weight : dict
        Dictionary of class weights for balancing the dataset.

    Parameters
    ----------
    config : traitlets.loader.Config, optional
        Configuration specified by config file or cmdline arguments.
        Used to set traitlet values.
        This is mutually exclusive with passing a ``parent``.
    parent : ctapipe.core.Component or ctapipe.core.Tool, optional
        Parent of this component in the configuration hierarchy,
        this is mutually exclusive with passing ``config``.
    **kwargs : dict
        Additional keyword arguments.

    Methods
    -------
    generate_mono_batch(batch_indices)
        Generate a batch of mono events from list of indices.
    generate_stereo_batch(batch_indices)
        Generate a batch of stereo events from list of indices.
    get_tel_pointing(file, tel_id)
        Retrieve the telescope pointing information for the specified telescope ID.
    close_files()
        Close all open files.
    """

    mode = CaselessStrEnum(
        ["mono", "stereo"],
        default_value="mono",
        help=(
            "Set data loading mode. "
            "``mono``: single images of one telescope type "
            "``stereo``: events including multiple telescope types "
        ),
    ).tag(config=True)

    skip_incompatible_files = Bool(
        default_value=False,
        help="Skip files that are not compatible to the reference instead of raising an error",
    ).tag(config=True)

    enforce_subarray_equality = Bool(
        default_value=True,
        help=(
            "Enforce strict equality of subarray descriptions between files. "
            "If False, a looser check primarily on telescope IDs is performed "
            "to ensure compatibility. Error will be raised if selected check failed "
            "and skip_incompatible_files is False."
        ),
    ).tag(config=True)

    allowed_tel_types = List(
        default_value=None,
        allow_none=True,
        help=(
            "List of allowed tel_types, others will be ignored. "
            "If None, all telescope types in the input stream "
            "will be included restricted by trait ``allowed_tels``"
        ),
    ).tag(config=True)

    allowed_tels = Set(
        trait=CInt(),
        default_value=None,
        allow_none=True,
        help=(
            "List of allowed tel_ids, others will be ignored. "
            "If None, all telescopes in the input stream "
            "will be included restricted by trait ``allowed_tel_types``"
        ),
    ).tag(config=True)

    image_mapper_type = TelescopeParameter(
        trait=Unicode(),
        default_value="BilinearMapper",
        allow_none=True,
        help=(
            "Instances of ``ImageMapper`` transforming a raw 1D vector into a 2D image. "
            "Different mapping methods can be selected for each telescope type."
        ),
    ).tag(config=True)

    focal_length_choice = UseEnum(
        FocalLengthKind,
        default_value=FocalLengthKind.EFFECTIVE,
        help=(
            "If both nominal and effective focal lengths are available in the"
            " SimTelArray file, which one to use for the `~ctapipe.coordinates.CameraFrame`"
            " attached to the `~ctapipe.instrument.CameraGeometry` instances in"
            " the `~ctapipe.instrument.SubarrayDescription`, which will be used in"
            " CameraFrame to TelescopeFrame coordinate transforms. "
            " The 'nominal' focal length is the one used during "
            " the simulation, the 'effective' focal length is computed using specialized "
            " ray-tracing from a point light source"
        ),
    ).tag(config=True)

    force_dl1_lookup = Bool(
        default_value=False,
        allow_none=True,
        help=(
            "Force to retrieve the table indices from the DL1 image table. "
            "Usually the table indices can be retrieved from the DL1 parameter table. "
            "In case a scrict ordering can not be guaranteed, the DL1 image table "
            "has to be used to retrieve the table indices. This results in a "
            "significantly slower initializing."
        ),
    ).tag(config=True)

    min_telescopes = Int(
        default_value=1,
        help=(
            "Minimum number of telescopes required globally after ``TableQualityQuery``. "
            "Events with fewer telescopes will be filtered out completely. "
            "Requires mode to be ``stereo``."
        ),
    ).tag(config=True)

    min_telescopes_of_type = IntTelescopeParameter(
        default_value=0,
        help=(
            "Minimum number of telescopes required for a specific type after ``TableQualityQuery``. "
            "In events with fewer telescopes of that type, "
            "those telescopes will be removed from the array event. "
            "This might result in the event not fulfilling ``min_telescopes`` anymore "
            "and thus being filtered completely. "
            "Requires mode to be ``stereo``. "
        ),
    ).tag(config=True)

    def __init__(
        self,
        input_url_signal,
        input_url_background=[],
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(config=config, parent=parent, **kwargs)

        # Register the destructor to close all open files properly
        atexit.register(self.__destructor)
        # Initialize the Table data quality query
        self.quality_query = TableQualityQuery(parent=self)
        
        # Construct dict of filename:file_handle pairs of an ordered file list
        self.input_url_signal = input_url_signal
        self.input_url_background = input_url_background
        self.files = OrderedDict()
        file_list = (
            self.input_url_signal + self.input_url_background
            if self.input_url_background
            else self.input_url_signal
        )
        for filename in np.sort(file_list):
            with lock:
                self.files[filename] = tables.open_file(filename, mode="r")
        self.first_file = list(self.files)[0]
        # Save the user attributes and useful information retrieved from the first file as a reference
        self._v_attrs = self.files[self.first_file].root._v_attrs
        self.process_type = ProcessType(self._v_attrs["CTA PROCESS TYPE"])
        self.data_format_version = self._v_attrs["CTA PRODUCT DATA MODEL VERSION"]
        self.instrument_id = self._v_attrs["CTA INSTRUMENT ID"]

        # Check for the minimum ctapipe data format version (v6.0.0) for MC sims
        if (
            self.process_type == ProcessType.Simulation
            and int(self.data_format_version.split(".")[0].replace("v", "")) < 6
        ):
            raise IOError(
                f"Provided ctapipe data format version is '{self.data_format_version}' (must be >= v.6.0.0 for Simulation)."
            )
        # Check for the minimum ctapipe data format version (v5.0.0) for real observational data
        if (
            self.process_type == ProcessType.Observation
            and int(self.data_format_version.split(".")[0].replace("v", "")) < 5
        ):
            raise IOError(
                f"Provided ctapipe data format version is '{self.data_format_version}' (must be >= v.5.0.0 for Observation)."
            )
        # Check for real data processing that only a single file is provided.
        if self.process_type == ProcessType.Observation and len(self.files) != 1:
            raise ValueError(
                f"When processing real observational data, please provide a single file (currently: '{len(self.files)}')."
            )

        # Set up the subarray
        self.subarray = SubarrayDescription.from_hdf(self.first_file, focal_length_choice=self.focal_length_choice)
        selected_tel_ids = None
        if self.allowed_tels is not None:
            selected_tel_ids = np.array(list(self.allowed_tels), dtype=np.int16)
        else:
            if self.allowed_tel_types is not None:
                selected_tel_ids = np.ravel(
                    [
                        np.array(self.subarray.get_tel_ids_for_type(str(tel_type)))
                        for tel_type in self.allowed_tel_types
                    ]
                )

        # Filter subarray by selected telescopes
        if selected_tel_ids is not None:
            self.subarray = self.subarray.select_subarray(selected_tel_ids)
        self.tel_ids = self.subarray.tel_ids
        # Copy the pixel rotation of the camera geometry for each telescope of the subarray
        # in a variable since the ImageMapper will derotate the pixels. The pixel rotation
        # is needed to create a rotated camera frame in order to transform the true Alt/Az
        # coordinates to correct camera coordinate offsets.
        self.pix_rotation = {}
        for tel_id in self.tel_ids:
            self.pix_rotation[tel_id] = self.subarray.tel[
                tel_id
            ].camera.geometry.pix_rotation
        self.selected_telescopes = {}
        for tel_type in self.subarray.telescope_types:
            # If is needed here for some sims where the same tel_type is stored twice
            if str(tel_type) not in self.selected_telescopes:
                self.selected_telescopes[str(tel_type)] = np.array(
                    self.subarray.get_tel_ids_for_type(str(tel_type))
                )

        # Check if only one telescope type is selected for any subclass except the 'DLFeatureVectorReader'
        if (
            self.__class__.__name__ != "DLFeatureVectorReader"
            and len(self.selected_telescopes) > 1
        ):
            raise ValueError(
                f"'{self.__class__.__name__}' do not support multiple telescope types: '{self.selected_telescopes}'. "
                "Please select only one telescope type or perform the event reconstruction with multiple telescope "
                "types using the 'DLFeatureVectorReader' subclass. Beforehand, the feature vectors have to be appended "
                "to the DL1 data files using '$ ctlearn-predict-model --dl1-features ...'."
            )
        # Check that all files have the same SubarrayDescription
        for filename in self.files:
            # Read SubarrayDescription from the new file
            subarray = SubarrayDescription.from_hdf(filename, focal_length_choice=self.focal_length_choice)

            # Filter subarray by selected telescopes
            if selected_tel_ids is not None:
                subarray = subarray.select_subarray(self.tel_ids)

            # Check if the subarray matches the reference
            subarrays_match = (
                subarray.__eq__(self.subarray)
                if self.enforce_subarray_equality
                else SubarrayDescription.check_matching_subarrays([self.subarray, subarray])
            )
            if not subarrays_match:
                message = (
                    f"Subarray description of file '{filename}' does not match the reference subarray description."
                )
                if self.skip_incompatible_files:
                    self.log.warning(f"Skipping '{filename}'. {message}")
                    del self.files[filename]
                else:
                    raise ValueError(message)

        # Set the telescope type and camera name as class attributes for mono mode for convenience
        # FIXME Make image mapper not a dict because we only need one since we do not select multiple telescope types for image/wvf reading
        self.tel_type = list(self.selected_telescopes)[0]
        self.cam_name = self._get_camera_type(self.tel_type)
        # Initialize the ImageMapper with the pixel positions and mapping settings
        # TODO: Find a better way for passing the configuration
        self.image_mappers = {}
        cam_geom = {}
        if self.image_mapper_type is not None:
            for i, camera_type in enumerate(self.subarray.camera_types):
                camera_name = self._get_camera_type(camera_type.name)
                if camera_type.name == "UNKNOWN-7987PX":
                    self.subarray.camera_types[i].name = "AdvCamSiPM"
                    self.subarray.camera_types[i].geometry.name = "AdvCamSiPM"
                    self.subarray.camera_types[i].readout.name = "AdvCamSiPM"
                if camera_name not in cam_geom:
                    cam_geom[camera_name] = camera_type.geometry
                    for scope, tel_type, name in self.image_mapper_type:
                        if scope == "type" and camera_name in tel_type:
                            self.image_mappers[camera_name] = ImageMapper.from_name(
                                name,
                                geometry=cam_geom[camera_name],
                                subarray=self.subarray,
                                parent=self,
                            )
                        if tel_type == "*" and camera_name not in self.image_mappers:
                            self.image_mappers[camera_name] = ImageMapper.from_name(
                                name,
                                geometry=cam_geom[camera_name],
                                subarray=self.subarray,
                                parent=self,
                            )

        # Telescope pointings
        self.telescope_pointings = {}
        self.tel_trigger_table, self.subarray_trigger_table = None, None
        if self.process_type == ProcessType.Observation:
            for tel_id in self.tel_ids:
                with lock:
                    # Read the telescope pointing information from the dl0/dl1 monitoring tables.
                    # dl1 monitoring table has priority.
                    if self.files[self.first_file].__contains__(
                        f"/dl0/monitoring/telescope/pointing/tel_{tel_id:03d}"
                    ):
                        self.telescope_pointings[f"tel_{tel_id:03d}"] = read_table(
                            self.files[self.first_file],
                            f"/dl0/monitoring/telescope/pointing/tel_{tel_id:03d}",
                        )
                    if self.files[self.first_file].__contains__(
                        f"/dl1/monitoring/telescope/pointing/tel_{tel_id:03d}"
                    ):
                        self.telescope_pointings[f"tel_{tel_id:03d}"] = read_table(
                            self.files[self.first_file],
                            f"/dl1/monitoring/telescope/pointing/tel_{tel_id:03d}",
                        )
                    # Break if no pointing information is available
                    if not self.files[self.first_file].__contains__(
                        f"/dl0/monitoring/telescope/pointing/tel_{tel_id:03d}"
                    ) and not self.files[self.first_file].__contains__(
                        f"/dl1/monitoring/telescope/pointing/tel_{tel_id:03d}"
                    ):
                        raise IOError(
                            f"Telescope pointing information for telescope '{tel_id}' is not available "
                            f"in the dl0/dl1 monitoring tables of file '{self.first_file}'."
                        )
        with lock:
            self.tel_trigger_table = read_table(
                self.files[self.first_file],
                "/dl1/event/telescope/trigger",
            )
            self.subarray_trigger_table = read_table(
                self.files[self.first_file],
                "/dl1/event/subarray/trigger",
            )
        # Image parameters (DL1b)
        # Retrieve the column names for the DL1b parameter table
        with lock:
            self.dl1b_parameter_colnames = read_table(
                self.files[self.first_file],
                f"/dl1/event/telescope/parameters/tel_{self.tel_ids[0]:03d}",
            ).colnames

        # Columns to keep in the example identifiers
        # This are the basic columns one need to do a
        # conventional IACT analysis with CNNs
        self.example_ids_keep_columns = ["table_index", "obs_id", "event_id", "tel_id"]
        if self.process_type == ProcessType.Simulation:
            self.example_ids_keep_columns.extend(
                [
                    "true_energy",
                    "true_shower_primary_id",
                    "true_az",
                    "telescope_pointing_azimuth",
                    "true_alt",
                    "telescope_pointing_altitude",
                    "cam_coord_offset_x",
                    "cam_coord_offset_y",
                    "cam_coord_distance",
                    "true_core_x",
                    "true_core_y",
                    "true_h_first_int",
                    "true_x_max"
                ]
            )
        elif self.process_type == ProcessType.Observation:
            self.example_ids_keep_columns.extend(["time", "event_type"])

        # Construct the example identifiers
        if self.mode == "mono":
            self._construct_mono_example_identifiers()
        elif self.mode == "stereo":
            self._construct_stereo_example_identifiers()

        # Handling the class weights calculation.
        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        # The RawTriggerReader in patch/sector mode retrieves the bkg from the signal events.
        self.class_weight = None
        if self.process_type == ProcessType.Simulation:
            if self.input_url_background or (isinstance(self, DLTriggerReader) and not self.one_class):
                self.class_weight = {
                    0: (1.0 / self.n_bkg_events) * (self._get_n_events() / 2.0),
                    1: (1.0 / self.n_signal_events) * (self._get_n_events() / 2.0),
                }

    def _get_camera_type(self, tel_type):
        """Extract the camera type from the telescope type string."""
        if tel_type.split("_")[-1] == "UNKNOWN-7987PX":
            return "AdvCamSiPM"
        else:
            return tel_type.split("_")[-1]

    def _get_n_events(self):
        """Return the number of events in the dataset."""
        if self.mode == "mono":
            return len(self.example_identifiers)
        elif self.mode == "stereo":
            return len(self.unique_example_identifiers)

    def _construct_mono_example_identifiers(self):
        """
        Construct example identifiers for mono mode.

        This method generates a list of example identifiers for the mono mode
        of operation. It processes the DL1b parameter tables for each telescope
        and constructs identifiers based on the event and telescope IDs. These
        identifiers are used to uniquely reference each example in the dataset.
        """
        simulation_info, example_identifiers = [], []
        for file_idx, (filename, f) in enumerate(self.files.items()):
            # Read the trigger table.
            trigger_table = read_table(f, "/dl1/event/subarray/trigger")
            if self.process_type == ProcessType.Simulation:
                # Read simulation information for each observation
                simulation_info.append(read_table(f, "/configuration/simulation/run"))
                # Construct the shower simulation table
                simshower_table = read_table(f, "/simulation/event/subarray/shower")
                # The shower simulation table is joined with the subarray trigger table.
                trigger_table = join(
                    left=trigger_table,
                    right=simshower_table,
                    keys=["obs_id", "event_id"],
                )

            # Construct the table containing all events.
            # First, the telescope tables are joined with the shower simulation
            # table and then those joined/merged tables are vertically stacked.
            tel_tables = []
            for tel_id in self.selected_telescopes[self.tel_type]:
                tel_table = read_table(
                    f, f"/dl1/event/telescope/parameters/tel_{tel_id:03d}"
                )
                if self.force_dl1_lookup:
                    # Read the DL1 image table
                    dl1_tel_table = read_table(
                        f, f"/dl1/event/telescope/images/tel_{tel_id:03d}",
                    )
                    # Keep only the columns needed for the join
                    dl1_tel_table.keep_columns(["obs_id", "event_id", "tel_id"])
                    # Add the table index to the DL1 image table
                    dl1_tel_table.add_column(
                        np.arange(len(dl1_tel_table)), name="table_index", index=0
                    )
                    # Unique the table to remove unwanted duplication
                    dl1_tel_table = unique(dl1_tel_table, keys=["obs_id", "event_id", "tel_id"])
                    # Join the DL1 image table with the DL1 parameter table
                    tel_table = join(
                        left=tel_table,
                        right=dl1_tel_table,
                        keys=["obs_id", "event_id", "tel_id"],
                    )
                else:
                    # Add the table index to the DL1 parameter table
                    tel_table.add_column(
                        np.arange(len(tel_table)), name="table_index", index=0
                    )
                if self.process_type == ProcessType.Simulation:
                    tel_table = join(
                        left=tel_table,
                        right=trigger_table,
                        keys=["obs_id", "event_id"],
                    )
                    # Add the spherical offsets w.r.t. to the telescope pointing
                    tel_pointing = self.get_tel_pointing(f, tel_id)
                    tel_table = join(
                        left=tel_table,
                        right=tel_pointing,
                        keys=["obs_id", "tel_id"],
                    )
                    tel_table = self._transform_to_cam_coord_offsets(tel_table)
                tel_tables.append(tel_table)
            events = vstack(tel_tables)

            # Initialize a boolean mask to True for all events
            # Todo: Does not have to be class attribute. This needed at the momment
            # for real data which is processed per file.
            self.passes_quality_checks = np.ones(len(events), dtype=bool)
            # Quality selection based on the dl1b parameter and MC shower simulation tables
            if self.quality_query:
                self.passes_quality_checks = self.quality_query.get_table_mask(events)

            # Apply the mask to filter events that are not fufilling the quality criteria
            events = events[self.passes_quality_checks]

            # Construct the example identifiers
            events.keep_columns(self.example_ids_keep_columns)
            if self.process_type == ProcessType.Simulation:
                # Add the spherical offsets w.r.t. to the telescope pointing
                array_pointing = self.get_array_pointing(f)
                # Join the prediction table with the telescope pointing table
                events = join(
                    left=events,
                    right=array_pointing,
                    keys=["obs_id"],
                )
                events = self._transform_to_sky_spher_offsets(events)
                # Add the logarithm of the true energy in TeV
                events = self._transform_to_log_energy(events)
                # Add the impact radius
                events = self._transform_to_impact_radius(events)
                # Add the true shower primary class to the table based on the filename
                # is signal or background input file list
                true_shower_primary_class = (
                    1 if filename in self.input_url_signal else 0
                )
                events.add_column(
                    true_shower_primary_class, name="true_shower_primary_class"
                )
            # Add telescope type id which is always 0 in mono mode
            # This is needed to share code with stereo reading mode later on
            events.add_column(file_idx, name="file_index", index=0)
            events.add_column(0, name="tel_type_id", index=3)
            # Appending the events to the list of example identifiers
            example_identifiers.append(events)

        # Constrcut the example identifiers for all files
        self.example_identifiers = vstack(example_identifiers)
        # For the RawTriggerReader patches option we need extra columns and rows to retrieve 
        # more than one patch per event.
        if isinstance(self,DLTriggerReader):
            if self.input_trigger_files:
                self.example_identifiers = self._add_trigger_table(self.example_identifiers)
                # Keep only the events that passed the low level trigger (digital sum or digital sum + tdscan).
                if self.trigger_cuts:
                    trigger_array = np.stack(self.example_identifiers['trigger_per_sample'])
                    cpe = np.asarray(self.example_identifiers['true_image_sum'])
                    length = len(trigger_array[0])
                    center = length // 2
                    if self.apply_trigger:
                        if self.trigger_length is None:
                            raise ValueError(
                                "trigger_length must be defined when applying trigger."
                            )
                        length -= self.trigger_length + 1
                    # Look for real triggers for gammas, 20 ns around the center of the simulation window.
                    # For the NSB we just need the first trigger to be in the second retrieved sample.
                    combined_mask = (
                        (cpe >= self.cpe_threshold) & np.any(trigger_array[:, center-10:center+11], axis=1) | 
                        (cpe == 0) & np.any(trigger_array[:, :length], axis=1)                        
                    )
                    self.example_identifiers = self.example_identifiers[combined_mask]
            # Retrieve the example identifiers for multiple patches option.
            self.example_identifiers = self._get_raw_example(self.example_identifiers)

        self.example_identifiers.sort(["file_index", "obs_id", "event_id", "tel_id", "tel_type_id"])
        # Construct simulation information for all files
        if self.process_type == ProcessType.Simulation:
            self.simulation_info = vstack(simulation_info)
            if isinstance(self,DLTriggerReader):
                class_column = "patch_class"
            else:
                class_column = "true_shower_primary_class"
            self.n_signal_events = np.count_nonzero(
                self.example_identifiers[class_column] == 1
            )
            if self.input_url_background or (isinstance(self, DLTriggerReader) and not self.one_class):
                self.n_bkg_events = np.count_nonzero(
                    self.example_identifiers[class_column] == 0
                )
        # Add index column to the example identifiers to later retrieve batches
        # using the loc functionality
        self.example_identifiers.add_column(
            np.arange(len(self.example_identifiers)), name="index", index=0
        )
        self.example_identifiers.add_index("index")

    def _construct_stereo_example_identifiers(self):
        """
        Construct example identifiers for stereo mode.

        This method generates a list of example identifiers for the stereo mode
        of operation. It processes the DL1b parameter tables for each event and constructs
        identifiers based on the event ID and the combination of telescope IDs that participated
        (triggered and passed quality cuts) in the event. These identifiers are used to uniquely
        reference each example in the dataset.
        """
        # Extend the columns to keep in the example identifiers
        self.example_ids_keep_columns.extend(["hillas_intensity"])
        simulation_info, example_identifiers = [], []
        for file_idx, (filename, f) in enumerate(self.files.items()):
            # Read the trigger table.
            trigger_table = read_table(f, "/dl1/event/subarray/trigger")
            if self.process_type == ProcessType.Simulation:
                # Read simulation information for each observation
                simulation_info_table = read_table(f, "/configuration/simulation/run")
                # Append the simulation information to the list of simulation information
                simulation_info.append(simulation_info_table)
                # Construct the shower simulation table
                simshower_table = read_table(f, "/simulation/event/subarray/shower")
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
                    if self.force_dl1_lookup:
                        # Read the DL1 image table
                        dl1_tel_table = read_table(
                            f, f"/dl1/event/telescope/images/tel_{tel_id:03d}",
                        )
                        # Keep only the columns needed for the join
                        dl1_tel_table.keep_columns(["obs_id", "event_id", "tel_id"])
                        # Add the table index to the DL1 image table
                        dl1_tel_table.add_column(
                            np.arange(len(dl1_tel_table)), name="table_index", index=0
                        )
                        # Unique the table to remove unwanted duplication
                        dl1_tel_table = unique(dl1_tel_table, keys=["obs_id", "event_id", "tel_id"])
                        # Join the DL1 image table with the DL1 parameter table
                        tel_table = join(
                            left=tel_table,
                            right=dl1_tel_table,
                            keys=["obs_id", "event_id", "tel_id"],
                        )
                    else:
                        # Add the table index to the DL1 parameter table
                        tel_table.add_column(
                            np.arange(len(tel_table)), name="table_index", index=0
                        )
                    # Initialize a boolean mask to True for all events
                    passes_quality_checks = np.ones(len(tel_table), dtype=bool)
                    # Quality selection based on the dl1b parameter and MC shower simulation tables
                    if self.quality_query:
                        passes_quality_checks = self.quality_query.get_table_mask(
                            tel_table
                        )
                    # Merge the telescope table with the trigger table
                    merged_table = join(
                        left=tel_table[passes_quality_checks],
                        right=trigger_table,
                        keys=["obs_id", "event_id"],
                    )
                    if self.process_type == ProcessType.Simulation:
                        tel_pointing = self.get_tel_pointing(f, tel_id)
                        merged_table = join(
                            left=merged_table,
                            right=tel_pointing,
                            keys=["obs_id", "tel_id"],
                        )
                        merged_table = self._transform_to_cam_coord_offsets(
                            merged_table
                        )
                    table_per_type.append(merged_table)
                table_per_type = vstack(table_per_type)
                table_per_type.keep_columns(self.example_ids_keep_columns)
                # Apply the multiplicity cut based on the telescope type
                table_per_type = table_per_type.group_by(["obs_id", "event_id"])

                def _multiplicity_cut_tel_type(table, key_colnames):
                    self.min_telescopes_of_type.attach_subarray(self.subarray)
                    return len(table) >= self.min_telescopes_of_type.tel[tel_type]

                table_per_type = table_per_type.groups.filter(
                    _multiplicity_cut_tel_type
                )

                table_per_type.add_column(tel_type_id, name="tel_type_id", index=3)
                events.append(table_per_type)
            events = vstack(events)
            # Apply the multiplicity cut based on the subarray
            events = events.group_by(["obs_id", "event_id"])

            def _multiplicity_cut_subarray(table, key_colnames):
                return len(table) >= self.min_telescopes

            events = events.groups.filter(_multiplicity_cut_subarray)
            events.add_column(file_idx, name="file_index", index=0)
            if self.process_type == ProcessType.Simulation:
                # Add the logarithm of the true energy in TeV
                events = self._transform_to_log_energy(events)
                # Add the impact radius
                events = self._transform_to_impact_radius(events)
                # Add the true shower primary class to the table based on the filename
                # is signal or background input file list
                true_shower_primary_class = (
                    1 if filename in self.input_url_signal else 0
                )
                events.add_column(
                    true_shower_primary_class, name="true_shower_primary_class"
                )
                array_pointing = self.get_array_pointing(f)
                # Join the prediction table with the telescope pointing table
                events = join(
                    left=events,
                    right=array_pointing,
                    keys=["obs_id"],
                )
                events = self._transform_to_sky_spher_offsets(events)
            # Appending the events to the list of example identifiers
            example_identifiers.append(events)

        # Constrcut the example identifiers for all files
        self.example_identifiers = vstack(example_identifiers)
        self.example_identifiers.sort(["obs_id", "event_id", "tel_id", "tel_type_id"])
        self.example_identifiers_grouped = self.example_identifiers.group_by(
            ["obs_id", "event_id"]
        )
        # Unique example identifiers by events
        self.unique_example_identifiers = unique(
            self.example_identifiers, keys=["obs_id", "event_id"]
        )
        self.unique_example_identifiers.remove_columns(["table_index", "tel_type_id", "tel_id", "hillas_intensity"])
        # Construct simulation information for all files
        if self.process_type == ProcessType.Simulation:
            self.simulation_info = vstack(simulation_info)
            self.n_signal_events = np.count_nonzero(
                self.unique_example_identifiers["true_shower_primary_class"] == 1
            )
            if self.input_url_background:
                self.n_bkg_events = np.count_nonzero(
                    self.unique_example_identifiers["true_shower_primary_class"] == 0
                )
        # Workaround for the missing multicolumn indexing in astropy:
        # Need this PR https://github.com/astropy/astropy/pull/15826
        # waiting astropy v7.1.0
        # self.example_identifiers.add_index(["obs_id", "event_id"])

    def get_tel_pointing(self, file, tel_id) -> Table:
        """
        Retrieve the telescope pointing information for the specified telescope ID.

        This method extracts the pointing information (azimuth and altitude)
        for the given telescope ID from the provided file.

        Parameters:
        -----------
        file : str
            Path to the file containing the telescope pointing data.
        tel_id : int
            Telescope ID for which the pointing information is to be retrieved.

        Returns:
        --------
        tel_pointing : astropy.table.Table
            A table containing pointing information (azimuth and altitude)
            for the specified telescope ID.
        """
        with lock:
            tel_pointing = read_table(
                file,
                f"/configuration/telescope/pointing/tel_{tel_id:03d}",
            )
        return tel_pointing

    def get_array_pointing(self, file) -> Table:
        """
        Retrieve the array pointing information.

        This method extracts the array pointing information (azimuth and altitude)
        from the provided file.

        Parameters:
        -----------
        file : str
            Path to the file containing the array pointing data.

        Returns:
        --------
        array_pointing : astropy.table.Table
            A table containing array pointing information (azimuth and altitude).
        """
        # Read simulation information for each observation
        array_pointing = read_table(file, "/configuration/simulation/run")
        # Assuming min_az = max_az and min_alt = max_alt
        array_pointing.keep_columns(["obs_id", "min_az", "min_alt"])
        array_pointing.rename_column("min_az", "pointing_azimuth")
        array_pointing.rename_column("min_alt", "pointing_altitude")
        return array_pointing

    def _transform_to_log_energy(self, table):
        """
        Transform energy values to their logarithmic scale.

        This method converts the energy values in the provided table to their logarithmic scale.

        Parameters:
        -----------
        table : astropy.table.Table
            A Table containing the energy values.

        Returns:
        --------
        table : astropy.table.Table
            A Table with the logarithmic energy values added as a new column.
        """
        table.add_column(np.log10(table["true_energy"]), name="log_true_energy")
        return table

    def _transform_to_impact_radius(self, table) -> Table:
        """
        Transform core coordinates to impact radius.

        This method calculates the impact radius from the core coordinates
        in the provided table. The impact radius is the distance from the
        true core position to the subarray center.

        Parameters:
        -----------
        table : astropy.table.Table
            A Table containing the true core coordinates.

        Returns:
        --------
        table : astropy.table.Table
            A Table with the impact radius added as a new column.
        """
        # Calculate the impact radius
        impact_radius = np.sqrt(table["true_core_x"]**2 + table["true_core_y"]**2)
        # Add the impact radius to the table
        table.add_column(impact_radius, name="impact_radius")
        return table

    def _transform_to_cam_coord_offsets(self, table) -> Table:
        """
        Transform Alt/Az coordinates to camera coordinate offsets w.r.t. the telescope pointing.

        This method converts the Alt/Az coordinates in the provided table to spherical offsets
        w.r.t. the telescope pointing. It also calculates the angular separation between the
        true and telescope pointing directions.

        Parameters:
        -----------
        table : astropy.table.Table
            A Table containing the true Alt/Az coordinates and telescope pointing.

        Returns:
        --------
        table : astropy.table.Table
            A Table with the spherical offsets and the angular separation added as new columns.
        """
        # Get the telescope ID from the table
        tel_id = table["tel_id"][0]
        # Set the telescope pointing
        tel_ground_frame = self.subarray.tel_coords[
            self.subarray.tel_ids_to_indices(tel_id)
        ]
        # Set the AltAz frame with the tel location and reference time
        altaz = AltAz(
            location=tel_ground_frame.to_earth_location(),
            obstime=LST_EPOCH,
        )
        fix_tel_pointing = SkyCoord(
            table["telescope_pointing_azimuth"],
            table["telescope_pointing_altitude"],
            frame=altaz,
        )
        # Set a new camera frame with the pixel rotation of the camera
        camera_frame = CameraFrame(
            focal_length=self.subarray.tel[tel_id].camera.geometry.frame.focal_length,
            rotation=self.pix_rotation[tel_id],
            telescope_pointing=fix_tel_pointing,
        )
        # Transform the true Alt/Az coordinates to camera coordinates
        true_direction = SkyCoord(
            table["true_az"],
            table["true_alt"],
            frame=altaz,
        )
        # Calculate the camera coordinate offsets and distance
        true_cam_position = true_direction.transform_to(camera_frame)
        true_cam_distance = np.sqrt(true_cam_position.x**2 + true_cam_position.y**2)
        # Add the camera coordinate offsets and distance to the table
        table.add_column(true_cam_position.x, name="cam_coord_offset_x")
        table.add_column(true_cam_position.y, name="cam_coord_offset_y")
        table.add_column(true_cam_distance, name="cam_coord_distance")
        return table

    def _transform_to_sky_spher_offsets(self, table) -> Table:
        """
        Transform Alt/Az coordinates to sky spherical offsets w.r.t. the array pointing.

        This method converts the Alt/Az coordinates in the provided table to sky spherical offsets
        w.r.t. the array pointing. It also calculates the angular separation between the
        true and array pointing directions.

        Parameters:
        -----------
        table : astropy.table.Table
            A Table containing the true Alt/Az coordinates and array pointing.

        Returns:
        --------
        table : astropy.table.Table
            A Table with the spherical offsets and the angular separation added as new columns.
        """
        # Set the AltAz frame with the reference location and time
        altaz = AltAz(
            location=self.subarray.reference_location,
            obstime=LST_EPOCH,
        )
        # Set the array pointing
        fix_array_pointing = SkyCoord(
            az=table["pointing_azimuth"],
            alt=table["pointing_altitude"],
            frame=altaz,
        )
        # Set the nominal frame with the array pointing
        nom_frame = NominalFrame(
            origin=fix_array_pointing,
            location=self.subarray.reference_location,
            obstime=LST_EPOCH,
        )
        # Set the true direction in (alt, az) coordinates
        true_direction = SkyCoord(
            az=table["true_az"],
            alt=table["true_alt"],
            frame=altaz,
        )
        # Transform the true direction to the nominal frame
        sky_coord = true_direction.transform_to(nom_frame)
        # Add the spherical offsets to the table
        table.add_column(sky_coord.fov_lon.to(u.deg), name="fov_lon")
        table.add_column(sky_coord.fov_lat.to(u.deg), name="fov_lat")
        # Calculate the angular separation between the true and array pointing directions
        angular_separation = fix_array_pointing.separation(true_direction)
        # Add the angular separation to the table
        table.add_column(angular_separation, name="angular_separation")
        return table

    def get_parameters(self, batch, parameter_list=None) -> Dict:
        """
        Retrieve a dictionary of existing DL1b parameters for a given batch.

        Parameters
        ----------
        batch : astropy.table.Table
            A Table containing the batch with columns `file_index`, `table_index`, and `tel_id`.
        parameter_list : list
            List of DL1b parameters to retrieve.

        Returns
        -------
        param_dict : dict
            Dictionary where keys are parameter names and values are lists of parameter values across the batch.
        """
        if not parameter_list:
            parameter_list = self.dl1b_parameter_colnames
            
        param_dict = {param: [] for param in parameter_list}
        initialized = False
        available_params = set()

        for file_idx, table_idx, tel_id in batch.iterrows(
            "file_index", "table_index", "tel_id"
        ):
            filename = list(self.files)[file_idx]
            tel_table = f"tel_{tel_id:03d}"

            with lock:
                child = self.files[
                    filename
                ].root.dl1.event.telescope.parameters._f_get_child(tel_table)

                if not initialized:
                    available_params = set(child.dtype.names)
                    param_dict = {
                        param: []
                        for param in parameter_list
                        if param in available_params
                    }
                    initialized = True

                row = child[table_idx]

                for param in param_dict:
                    param_dict[param].append(row[param])

        if parameter_list != list(param_dict.keys()):
            self.log.warning("The parameter list does not match with the output.")
 
        return param_dict

    def generate_mono_batch(self, batch_indices) -> Table:
        """
        Generate a batch of events for mono mode.

        This method generates a batch of examples for the mono mode of operation.
        It retrieves the DL1b parameters and other relevant data for the specified
        batch indices and constructs a dictionary of input features optionally with
        a table of DL1b parameters.

        Parameters
        ----------
        batch_indices : list of int
            List of indices specifying the examples to include in the batch.
        dl1b_parameter_list : list of str, optional
            List of DL1b parameter names to include in the output table. If ``None``,
            no DL1b parameters are included.

        Returns
        -------
        dict
            Dictionary containing the input features for the batch. The keys are
            the feature names and the values are the corresponding data arrays.
        Table
            Table containing the DL1b parameters for the batch. The columns are
            the specified DL1b parameters and the rows correspond to the examples
            in the batch.
        """
        # Check that the batch generation call is consistent with the mode
        if self.mode != "mono":
            raise ValueError("Mono batch generation is not supported in stereo mode.")
        # Retrieve the batch from the example identifiers via indexing
        batch = self.example_identifiers.loc[batch_indices]
        # If the batch is a single event loc returns a Rows object and not a Table.
        # Convert the batch to a Table in order to append the features later
        if not isinstance(batch, Table):
            batch = Table(rows=batch)
        # Append the features from child classes to the batch
        batch = self._append_features(batch)
        return batch

    def generate_stereo_batch(self, batch_indices) -> Table:
        """
        Generate a batch of events for stereo mode.

        This method generates a batch of stereo examples based on the provided batch indices.
        It retrieves the DL1b parameters for the selected events and telescopes, and constructs
        the input data and labels for the batch.

        Parameters
        ----------
        batch_indices : list of int
            List of indices specifying the examples to include in the batch.
        dl1b_parameter_list : list of str, optional
            List of DL1b parameter names to include in the feature dictionary. If ``None``,
            no DL1b parameters are included.

        Returns
        -------
        dict
            Dictionary containing the feature for the batch. The keys are the parameter names
            and the values are the corresponding data arrays.
        Table
            Table containing the labels and additional infor for the batch examples.
        """
        # Check that the batch generation call is consistent with the mode
        if self.mode != "stereo":
            raise ValueError("Stereo batch generation is not supported in mono mode.")
        # Retrieve the batch from the example identifiers via groupd by
        # Workaround for the missing multicolumn indexing in astropy:
        # Need this PR https://github.com/astropy/astropy/pull/15826
        # waiting astropy v7.1.0
        # Once available, the batch_generation can be shared with "mono"
        batch = self.example_identifiers_grouped.groups[batch_indices]
        # This may returns a Rows object and not a Table if the batch is a single event.
        # Convert the batch to a Table in order to append the features later
        if not isinstance(batch, Table):
            batch = Table(rows=batch)
        # Append the features from child classes to the batch
        batch = self._append_features(batch)
        # Add blank inputs for missing telescopes in the batch
        if self.process_type == ProcessType.Simulation:
            batch_grouped = batch.group_by(
                ["obs_id", "event_id", "true_shower_primary_class"]
            )
        elif self.process_type == ProcessType.Observation:
            batch_grouped = batch.group_by(["obs_id", "event_id"])
        for group_element in batch_grouped.groups:
            for tel_type_id, tel_type in enumerate(self.selected_telescopes):
                if "features" in group_element.colnames:
                    blank_input = np.zeros(self.input_shape[tel_type][1:])
                if "mono_feature_vectors" in group_element.colnames:
                    blank_mono_feature_vectors = np.zeros(
                        group_element["mono_feature_vectors"][0].shape
                    )
                if "stereo_feature_vectors" in group_element.colnames:
                    blank_stereo_feature_vectors = np.zeros(
                        group_element["stereo_feature_vectors"][0].shape
                    )
                for tel_id in self.selected_telescopes[tel_type]:
                    # Check if the telescope is missing in the batch
                    if tel_id not in group_element["tel_id"]:
                        blank_input_row = group_element.copy()[0]
                        blank_input_row["table_index"] = -1
                        blank_input_row["tel_type_id"] = tel_type_id
                        blank_input_row["tel_id"] = tel_id
                        blank_input_row["hillas_intensity"] = 0.0
                        if "features" in group_element.colnames:
                            blank_input_row["features"] = blank_input
                        if "mono_feature_vectors" in group_element.colnames:
                            blank_input_row["mono_feature_vectors"] = (
                                blank_mono_feature_vectors
                            )
                        if "stereo_feature_vectors" in group_element.colnames:
                            blank_input_row["stereo_feature_vectors"] = (
                                blank_stereo_feature_vectors
                            )
                        batch.add_row(blank_input_row)
        # Sort the batch with the new rows of blank inputs
        batch.sort(["obs_id", "event_id", "tel_type_id", "tel_id"])
        return batch

    def __destructor(self):
        """Destructor to ensure all opened HDF5 files are properly closed."""
        if hasattr(
            self, "files"
        ):  # Ensure self.files exists before attempting to close
            for file_name in list(self.files.keys()):
                if self.files[file_name].isopen:  # Check if file is still open
                    self.files[file_name].close()

    @abstractmethod
    def _append_features(self, batch) -> Table:
        pass    

def get_unmapped_image(dl1_event, channels, transforms) -> np.ndarray:
    """
    Generate unmapped image from a DL1 event.

    This function processes the DL1 event data to generate an image array
    based on the specified channels and transformation parameters. It handles
    different types of channels such as 'image' and 'peak_time', and
    applies the necessary transformations to recover the original floating
    point values if the file was compressed.

    Parameters
    ----------
    dl1_event : astropy.table.Table
        A table containing DL1 event data, including ``image``, ``image_mask``,
        and ``peak_time``.
    channels : list of str
        A list of channels to be processed, such as ``image`` and ``peak_time``
        with optional ``cleaned_``-prefix for for the cleaned versions of the channels
        and ``relative_``-prefix for the relative peak arrival times.
    transforms : dict
        A dictionary containing scaling and offset values for image and peak time
        transformations.

    Returns
    -------
    image : np.ndarray
        The processed image data image for the specific channels.
    """
    # Initialize the image array
    image = np.zeros(
        shape=(
            len(dl1_event["image"]),
            len(channels),
        ),
        dtype=np.float32,
    )
    # Process the channels and apply the necessary transformations
    for i, channel in enumerate(channels):
        # Save the cleaning mask to be applied to the channels in various cases
        mask = dl1_event["image_mask"]
        # TODO: Check here if the mask is valid
        # and return NaNs if not and cleaned is requested
        # Process the integrated charges if specified
        if "image" in channel:
            image[:, i] = dl1_event["image"]
        # Process the peak arrival times if specified
        if "peak_time" in channel:
            # Calculate the relative peak arrival times if specified
            if "relative" in channel:
                peak_times = dl1_event["peak_time"]
                # Apply the cleaning mask to the peak times if specified
                if "cleaned" in channel:
                    peak_times *= mask
                image[:, i] = (
                    dl1_event["peak_time"] - peak_times[np.nonzero(peak_times)].mean()
                )
            else:
                image[:, i] = dl1_event["peak_time"]
        # Apply the cleaning mask to the image if specified
        if "cleaned" in channel:
            image[:, i] *= mask
        # Apply the transform to recover orginal floating point values if the file were compressed
        if "image" in channel:
            if transforms["image_scale"] > 0.0:
                image[:, i] /= transforms["image_scale"]
            if transforms["image_offset"] > 0:
                image[:, i] -= transforms["image_offset"]
        if "peak_time" in channel:
            if transforms["peak_time_scale"] > 0.0:
                image[:, i] /= transforms["peak_time_scale"]
            if transforms["peak_time_offset"] > 0:
                image[:, i] -= transforms["peak_time_offset"]
    return image


class DLImageReader(DLDataReader):
    """
    A data reader class for handling DL1 image data from telescopes.

    This class extends the ``DLDataReader`` to specifically handle the reading,
    transformation, and mapping of DL1 image data, including integrated charges
    and peak arrival times. It supports both ``mono`` and ``stereo`` data loading modes
    and can apply DL1 cleaning masks to the images if specified.

    Attributes
    ----------
    channels : list of str
        Specifies the input channels to be loaded, such as ``image`` and/or ``peak_time``.
        Also supports ``cleaned_``-prefix for the cleaned versions of the channels and
        ``relative_``-prefix for the relative peak arrival times.
    transforms : dict
        Contains scaling and offset values for image and peak time transformations.
    """

    channels = List(
        trait=CaselessStrEnum(
            [
                "image",
                "cleaned_image",
                "peak_time",
                "relative_peak_time",
                "cleaned_peak_time",
                "cleaned_relative_peak_time",
            ]
        ),
        default_value=["image", "peak_time"],
        allow_none=False,
        help=(
            "Set the input channels to be loaded from the DL1 event data. "
            "image: integrated charges, "
            "cleaned_image: integrated charges cleaned with the DL1 cleaning mask, "
            "peak_time: extracted peak arrival times, "
            "relative_peak_time: extracted relative peak arrival times, "
            "cleaned_peak_time: extracted peak arrival times cleaned with the DL1 cleaning mask,"
            "cleaned_relative_peak_time: extracted relative peak arrival times cleaned with the DL1 cleaning mask."
        ),
    ).tag(config=True)

    def __init__(
        self,
        input_url_signal,
        input_url_background=[],
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            input_url_signal=input_url_signal,
            input_url_background=input_url_background,
            config=config,
            parent=parent,
            **kwargs,
        )

        # Set the input shape based on the selected mode
        if self.mode == "mono":
            if self.image_mappers[self.cam_name].cam_neighbor_array is None:
                self.input_shape = (
                    self.image_mappers[self.cam_name].image_shape,
                    self.image_mappers[self.cam_name].image_shape,
                    len(self.channels),
                )
            else:
                self.input_shape = (self.image_mappers[self.cam_name].geometry.n_pixels, len(self.channels))
                
        elif self.mode == "stereo":
            self.input_shape = {}
            for tel_type in self.selected_telescopes:
                camera_name = super()._get_camera_type(tel_type)
                input_shape = (
                    len(self.subarray.get_tel_ids_for_type(tel_type)),
                    self.image_mappers[camera_name].image_shape,
                    self.image_mappers[camera_name].image_shape,
                    len(self.channels),
                )
                self.input_shape[tel_type] = input_shape

        # Get offset and scaling of images
        self.transforms = {}
        self.transforms["image_scale"] = 0.0
        self.transforms["image_offset"] = 0
        self.transforms["peak_time_scale"] = 0.0
        self.transforms["peak_time_offset"] = 0
        first_tel_table = f"tel_{self.tel_ids[0]:03d}"
        with lock:
            img_table_v_attrs = (
                self.files[self.first_file]
                .root.dl1.event.telescope.images._f_get_child(first_tel_table)
                ._v_attrs
            )
        # Check the transform value used for the file compression
        if "CTAFIELD_3_TRANSFORM_SCALE" in img_table_v_attrs:
            self.transforms["image_scale"] = img_table_v_attrs[
                "CTAFIELD_3_TRANSFORM_SCALE"
            ]
            self.transforms["image_offset"] = img_table_v_attrs[
                "CTAFIELD_3_TRANSFORM_OFFSET"
            ]
        if "CTAFIELD_4_TRANSFORM_SCALE" in img_table_v_attrs:
            self.transforms["peak_time_scale"] = img_table_v_attrs[
                "CTAFIELD_4_TRANSFORM_SCALE"
            ]
            self.transforms["peak_time_offset"] = img_table_v_attrs[
                "CTAFIELD_4_TRANSFORM_OFFSET"
            ]
        # If "HexagonalPatchMapper" retrieve the neighbor array of he complete camera.
        if self.image_mappers[self.cam_name].cam_neighbor_array is not None:
            self.neighbor_array = self.image_mappers[self.cam_name].cam_neighbor_array
        else:
            self.neighbor_array = None

    def _append_features(self, batch) -> Table:
        """
        Append images to a given batch as features.

        This method processes a batch of events to append images as input features for the neural networks.
        It reads the image data from the specified files, applies any necessary transformations, and maps
        the images using the appropriate ``ImageMapper``.

        Parameters
        ----------
        batch : astropy.table.Table
            A table containing information at minimum the following columns:
            - "file_index": List of indices corresponding to the files.
            - "table_index": List of indices corresponding to the event tables.
            - "tel_type_id": List of telescope type IDs.
            - "tel_id": List of telescope IDs.

        Returns
        -------
        batch : astropy.table.Table
            The input batch with the appended processed images as features.
        """
        images = []
        for file_idx, table_idx, tel_type_id, tel_id in batch.iterrows(
            "file_index", "table_index", "tel_type_id", "tel_id"
        ):
            filename = list(self.files)[file_idx]
            with lock:
                tel_table = f"tel_{tel_id:03d}"
                child = self.files[
                    filename
                ].root.dl1.event.telescope.images._f_get_child(tel_table)
                unmapped_image = get_unmapped_image(
                    child[table_idx], self.channels, self.transforms
                )
            # Apply the 'ImageMapper' whenever the index matrix is not None.
            # Otherwise, return the unmapped image for the 'IndexedConv' package.
            camera_type = self._get_camera_type(
                list(self.selected_telescopes.keys())[tel_type_id]
            )
            if self.image_mappers[camera_type].cam_neighbor_array is None and self.image_mappers[camera_type].index_matrix is None:
                images.append(self.image_mappers[camera_type].map_image(unmapped_image))
            else:
                images.append(unmapped_image)
        batch.add_column(images, name="features", index=7)
        return batch


def get_unmapped_waveform(
    r1_event,
    settings,
    camera_geometry=None,
    dl1_cleaning_mask=None,
) -> np.ndarray:
    """
    Retrieve and process the unmapped waveform from an R1 event.

    This function extracts the waveform data from an R1 event, applies necessary transformations
    based on the provided settings, and optionally applies a DL1 cleaning mask. The function
    supports handling waveforms with one or two gain channels and can crop the waveform sequence
    based on the specified sequence length and position.

    Parameters
    ----------
    r1_event : astropy.table.Table
        A table containing the R1 event data, including ``waveform`` and ``selected_gain_channel``.
    settings : dict
        Dictionary containing settings for waveform processing, including:
        - ``waveform_scale`` (float): Scale factor for waveform values.
        - ``waveform_offset`` (int): Offset value for waveform values.
        - ``cleaning_type`` (str or None): Data level on which the cleaning mask(s) are obtained for cleaning the waveforms.
        - ``seq_length`` (int): Length of the waveform sequence to be extracted.
        - ``readout_length`` (int): Total length of the readout window.
        - ``seq_position`` (str): Position of the sequence within the readout window (``center`` or ``maximum``).
    camera_geometry : ctapipe.instrument.CameraGeometry, optional
        The geometry of the camera, including pixel positions and camera type. Default is ``None``.
    dl1_cleaning_mask : numpy.ndarray, optional
        Array containing the DL1 cleaning mask to be applied to the waveform to find the shower maximum
        to center the sequence. Default is ``None``.

    Returns
    -------
    waveform : numpy.ndarray
        The processed and optionally cropped waveform data.
    """

    waveform = np.float32(r1_event["waveform"])
    # Check if camera has one or two gain(s) and apply selection
    if waveform.shape[0] == 1:
        waveform = waveform[0]
    else:
        selected_gain_channel = r1_event["selected_gain_channel"][:, np.newaxis]
        waveform = np.where(selected_gain_channel == 0, waveform[0], waveform[1])
    # Apply the transform to recover orginal floating point values if the file were compressed
    if settings["waveform_scale"] > 0.0:
        waveform /= settings["waveform_scale"]
    if settings["waveform_offset"] > 0:
        waveform -= settings["waveform_offset"]
    # Apply the DL1 cleaning mask if selected
    if settings["cleaning_type"] == "image":
        waveform *= dl1_cleaning_mask[:, None]
    elif settings["cleaning_type"] == "waveform":
        waveform = clean_waveform(waveform, camera_geometry, settings["DBSCAN_params"])
    # Retrieve the sequence around the center of the readout window or the shower maximum
    if settings["seq_length"] < settings["readout_length"]:
        if settings["seq_position"] == "center":
            sequence_position = waveform.shape[1] // 2 - 1
        elif settings["seq_position"] == "maximum":
            sequence_position = np.argmax(np.sum(waveform, axis=0))
        # Calculate start and stop positions
        start = max(0, int(1 + sequence_position - settings["seq_length"] / 2))
        stop = min(
            settings["readout_length"],
            int(1 + sequence_position + settings["seq_length"] / 2),
        )
        # Adjust the start and stop if bound overflows
        if stop > settings["readout_length"]:
            start -= stop - settings["readout_length"]
            stop = settings["readout_length"]
        # Crop the unmapped waveform in samples
        waveform = waveform[:, int(start) : int(stop)]
    return waveform


def clean_waveform(waveform, camera_geometry, DBSCAN_config):
    pass


class DLWaveformReader(DLDataReader):
    """
    A data reader class for handling R1 calibrated waveform data from telescopes.

    This class extends the ``DLDataReader`` to specifically handle the reading,
    transformation, and mapping of R1 calibrated waveform data. It supports both ``mono``
    and ``stereo`` data loading modes and can perform a cleaning to the waveforms
    if specified.

    Attributes
    ----------
    sequence_length : int or None
        Number of waveform samples considered in the selected sequence. If None,
        the sequence length is set to the readout length.
    sequence_position : str
        Position of the sequence within the readout window. Can be ``center`` or ``maximum``.
    cleaning_type : str or None
        Data level on which the cleaning mask(s) are obtained for cleaning the waveforms.
        Can be ``image`` or ``waveform``.
    DBSCAN_params : dict or None
        Dictionary containing the DBSCAN clustering parameters for waveform cleaning.
    waveform_settings : dict
        Contains settings for waveform processing, including cleaning type (with DBSCAN parameters),
        sequence length, readout length, sequence position, scale, and offset.
    """

    sequence_length = Int(
        default_value=None,
        allow_none=True,
        help="Number of waveform samples considered in the selected sequence.",
    ).tag(config=True)

    sequence_position = CaselessStrEnum(
        ["center", "maximum"],
        default_value="center",
        help=(
            "Set where to position the sequence if ``sequence_length`` is selected. "
            "``center``: sequence is extracted around the center of the readout window. "
            "``maximum``: sequence is extracted around the shower maximum. "
        ),
    ).tag(config=True)

    cleaning_type = CaselessStrEnum(
        ["image", "waveform"],
        allow_none=True,
        default_value=None,
        help=(
            "Set whether to apply cleaning of the calibrated waveforms. "
            "Two cleaning types are supported obtained from different data levels."
            "``image``: apply the DL1 cleaning mask to the calibrated waveforms. "
            "``waveform``: perform a digital sum and a DBSCAN clustering on-the-fly. "
        ),
    ).tag(config=True)

    DBSCAN_params = Dict(
        default_value={"eps": 0.5, "min_samples": 5, "metric": "euclidean"},
        allow_none=True,
        help=(
            "Set the DBSCAN clustering parameters for waveform cleaning. "
            "Only required when ``cleaning_type`` is set to ``waveform``. "
            "``eps``: The maximum distance between two samples for one to be considered as in the neighborhood of the other."
            "``min_samples``: The number of samples in a neighborhood for a point to be considered as a core point."
            "``metric``: The metric to use when calculating distance between instances in a feature array."
        ),
    ).tag(config=True)

    data_level = DataLevel.R1

    def __init__(
        self,
        input_url_signal,
        input_url_background=[],
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            input_url_signal=input_url_signal,
            input_url_background=input_url_background,
            config=config,
            parent=parent,
            **kwargs,
        )

        # Read the readout length from the first file
        data_group = getattr(self.files[self.first_file].root, self.data_level.name.lower())
        self.readout_length = int(
            data_group
            .event.telescope._f_get_child(f"tel_{self.tel_ids[0]:03d}")
            .coldescrs["waveform"]
            .shape[-1]
        )

        # Set the sequence length to the readout length if not selected
        if self.sequence_length is None:
            self.sequence_length = self.readout_length
        else:
            # Check that the waveform sequence length is valid
            if self.sequence_length > self.readout_length:
                raise ValueError(
                    f"Invalid sequence length '{self.sequence_length}' (must be <= '{self.readout_length}')."
                )

        # Set the input shape based on the selected mode
        if self.mode == "mono":
            if self.image_mappers[self.cam_name].cam_neighbor_array is None:
                self.input_shape = (
                    self.image_mappers[self.cam_name].image_shape,
                    self.image_mappers[self.cam_name].image_shape,
                    self.sequence_length,
                )
            else:
                self.input_shape = (self.image_mappers[self.cam_name].geometry.n_pixels, self.sequence_length)
        elif self.mode == "stereo":
            self.input_shape = {}
            for tel_type in self.selected_telescopes:
                camera_name = super()._get_camera_type(tel_type)
                input_shape = (
                    len(self.subarray.get_tel_ids_for_type(tel_type)),
                    self.image_mappers[camera_name].image_shape,
                    self.image_mappers[camera_name].image_shape,
                    self.sequence_length,
                )
                self.input_shape[tel_type] = input_shape

        # Construct settings dict for the calibrated waveforms
        self.waveform_settings = {
            "cleaning_type": self.cleaning_type,
            "seq_length": self.sequence_length,
            "readout_length": self.readout_length,
            "seq_position": self.sequence_position,
            "DBSCAN_params": self.DBSCAN_params,
        }

        # Check the transform value used for the file compression
        self.waveform_settings["waveform_scale"] = 0.0
        self.waveform_settings["waveform_offset"] = 0
        with lock:
            wvf_table_v_attrs = (
                data_group
                .event.telescope._f_get_child(f"tel_{self.tel_ids[0]:03d}")
                ._v_attrs
            )
        if "CTAFIELD_5_TRANSFORM_SCALE" in wvf_table_v_attrs:
            self.waveform_settings["waveform_scale"] = wvf_table_v_attrs[
                "CTAFIELD_5_TRANSFORM_SCALE"
            ]
            self.waveform_settings["waveform_offset"] = wvf_table_v_attrs[
                "CTAFIELD_5_TRANSFORM_OFFSET"
            ]

        # If "HexagonalPatchMapper" retrieve the neighbor array of he complete camera.
        if self.image_mappers[self.cam_name].cam_neighbor_array is not None:
            self.neighbor_array = self.image_mappers[self.cam_name].cam_neighbor_array
        else:
            self.neighbor_array = None
        
    def _append_features(self, batch) -> Table:
        """
        Append waveforms to a given batch as features.

        This method processes a batch of events to append waveforms as input features for the neural networks.
        It reads the waveform data from the specified files, applies any necessary transformations, and maps
        the waveforms using the appropriate ``ImageMapper``.

        Parameters
        ----------
        batch : astropy.table.Table
            A table containing information at minimum the following columns:
            - ``file_index``: List of indices corresponding to the files.
            - ``table_index``: List of indices corresponding to the event tables.
            - ``tel_type_id``: List of telescope type IDs.
            - ``tel_id``: List of telescope IDs.

        Returns
        -------
        batch : astropy.table.Table
            The input batch with the appended processed waveforms as features.
        """
        waveforms = []
        for file_idx, table_idx, tel_type_id, tel_id in batch.iterrows(
            "file_index", "table_index", "tel_type_id", "tel_id"
        ):
            filename = list(self.files)[file_idx]
            camera_type = self._get_camera_type(
                list(self.selected_telescopes.keys())[tel_type_id]
            )
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
                            img_child[table_idx]["image_mask"], dtype=int
                        )
                unmapped_waveform = get_unmapped_waveform(
                    child[table_idx],
                    self.waveform_settings,
                    self.image_mappers[camera_type].geometry,
                    dl1_cleaning_mask,
                )
            # Apply the 'ImageMapper' whenever the index matrix is not None.
            # Otherwise, return the unmapped image for the 'IndexedConv' package.
            if self.image_mappers[self.cam_name].cam_neighbor_array is None and self.image_mappers[camera_type].index_matrix is None:
                waveforms.append(
                    self.image_mappers[camera_type].map_image(unmapped_waveform)
                )
            else:
                waveforms.append(unmapped_waveform)
        batch.add_column(waveforms, name="features", index=7)
        return batch

def get_true_image(sim_event) -> np.ndarray:
    """
    Retrieve the true image from the simulated event.

    This method retrieves the simulated true image from a given event, i.e. only 
    Cherenkov p.e.

    Parameters
    ----------
    sim_event : dl1_event : astropy.table.Table
        A table containing the simulated event data, including ``true_image`` and
        ``true_image_sum``. 

    Returns
    -------
    true_image : np.ndarray
        The simulated image with only Cherenkov p.e.
    """

    return np.array(sim_event["true_image"], dtype=np.int32).reshape(-1, 1)


def apply_digital_sum(waveform, mapper, l1_settings):
    """
    Apply digital sum to the raw waveform.

    This method applies a digital sum to the raw waveform, the region sum size options are per flower,
    superflower or front end board (hardware module with the size of a superflower).

    Parameters
    ----------
    waveform : r0_event : np.ndarray
        An array containing the complete raw waveform.
    mapper : image_mapper object
        Object to retrieve all the neighbors and patches information.
    l1_settings : dict
        Dictionary with the digital sum trigger settings.
            - ``eps`` : regions on which performing the digital sums options are ``flower``, ``superflower`` or ``feb``.
            - ``threshold`` : threshold in ADC counts for the digital sum.

    Returns
    -------
    bin_flowers : np.ndarray
        The trigger mask from the thresholded digital sum, with shape (num_flowers, time).
    flower_sums : np.ndarray
        The l1 digital sums result, with shape (num_flowers, time).
    """
    # Assign the l1 neighbor list, summing on flowers or superflowers for each flower.
    # For the superflowers, the outter flowers may not have 6 flower neighbors so need to mask.
    if l1_settings["eps"] == "flower":
        l1_list = mapper.fl_neighbor_array_l1
        flower_sums = waveform[np.array(l1_list)].sum(axis=1) # shape: (n_flowers, time)
    elif l1_settings["eps"] == "superflower":
        l1_list = mapper.supfl_neighbor_array_l1
        l1_mask = mapper.supfl_neighbor_l1_mask
        flower_sums = (waveform[l1_list] * l1_mask[..., None]).sum(axis=1) # shape: (n_flowers, time)
    else:
        l1_list = mapper.feb_indices
        flower_sums = waveform[np.array(l1_list)].sum(axis=1)
    # Thresholding
    bin_flowers = (flower_sums > l1_settings["threshold"]).astype(int)
    return bin_flowers, flower_sums

def apply_tdscan(bin_flowers, mapper, l1_settings, tdscan_settings):
    """
    Parameters
    ----------
    bin_flowers : np.ndarray
        An array containing the thresholded digital sums.
    mapper : image_mapper object
        Object to retrieve all the neighbors and patches information.
    l1_settings : dict
        Dictionary with the digital sum trigger settings.
            - ``eps`` : regions on which performing the digital sums options are ``flower``, ``superflower`` or ``feb``.
            - ``threshold`` : threshold in ADC counts for the digital sum.
    tdscan_settings : dict
        Dictionary with the tdscan trigger settings.
            - ``eps_xy`` : 1 to perform tdscan with 1st order flowers neighbors, 2 for 2nd order.
            - ``eps_t`` : Set the the number of samples before and after to consider in the convolution. "
            - ``min_pts`` : Threshold in binary flower counts above which the convolution is going to keep the central flower. "

    Returns
    -------
    tdscan_flowers : np.ndarray
        The cleaned trigger mask from TDSCAN, with shape (num_flowers, time).
    """
    num_flowers, time = bin_flowers.shape
    # TDSCAN mask cleaning:
    tdscan_flowers = np.zeros((num_flowers, time), dtype=np.float32)
    # Precompute cumulative sum over time for each pixel
    cumsum_l1 = np.pad(np.cumsum(bin_flowers, axis=1), ((0, 0), (1, 0)))  # pad to handle start_t = 0 cleanly
    # Select the neighbor list depending on epsxy
    eps_xy_neighbors = (
        mapper.neighbor_tdscan_eps1_list
        if l1_settings["eps"] in ["flower", "superflower"]
        else mapper.feb_neighbors
    )
    for i in range(num_flowers):
        neighbors = eps_xy_neighbors[i]
        if len(neighbors) == 0:
            continue
        # Compute rolling sum using cumsum
        start = np.arange(time) - tdscan_settings["eps_t"]
        end = np.arange(time) + tdscan_settings["eps_t"] + 1
        start = np.clip(start, 0, time)
        end = np.clip(end, 0, time)
        # Get total values in the window for each time frame using broadcasting
        total = cumsum_l1[neighbors][:, end] - cumsum_l1[neighbors][:, start]  # shape: (num_neighbors, time)
        # sum across neighbors and threshold.
        tdscan_flowers[i] = (np.sum(total, axis=0) >= tdscan_settings["min_pts"]).astype(np.float32)
    return tdscan_flowers

def quantised_per_flower(flower_sums, bin_flowers, l1_settings, quant_step):
    """
    Retrieve the flower sums and triggered flowers per FEB.

    This method retrieves per FEB, the number of flowers that were triggered with the digital sum + OR (3 bits), 
    and the total amplitude above threshold (7 bits) of all the flowers quantising if asked.

    Parameters
    ----------
    flower_sums : np.ndarray
        An array containing the digital sums.
    bin_flowers : np.ndarray
        An array containing the thresholded digital sums.
    l1_settings : dict
        Dictionary with the digital sum trigger settings.
            - ``eps`` : regions on which performing the digital sums options are ``flower``, ``superflower`` or ``feb``.
            - ``threshold`` : threshold in ADC counts for the digital sum.

    Returns
    -------
    output : np.ndarray
        The trigger information with total amplitude and number of triggered flowers, with shape (163, time, 2).
    """
    # Sum all the flowers of a FEB
    flower_sums_red = np.maximum(flower_sums.astype(np.int32) - l1_settings["threshold"], 0)
    feb_sums = flower_sums_red.reshape(163, 7, flower_sums.shape[-1])
    feb_summed = feb_sums.sum(axis=1)
    # Quantize the amplitudes
    vals_q = np.floor_divide(feb_summed, quant_step)
    vals_q = np.clip(vals_q, 0, 127).astype(np.uint16)
    # Retrieve number of triggers per flower
    mask_feb = bin_flowers.reshape(163, 7, flower_sums.shape[-1])
    mask_feb_summed = mask_feb.sum(axis=1)
    output = np.stack([mask_feb_summed, vals_q], axis=-1)
    return output

def quantised_per_feb(feb_sums, l1_settings, quant_step):
    """
    Retrieve the quantised digital sum per FEB.

    This method retrieves per FEB, the amplitude above threshold of the FEB (10 bits) quantising if asked.

    Parameters
    ----------
    feb_sums : np.ndarray
        An array containing the digital sums.
    l1_settings : dict
        Dictionary with the digital sum trigger settings.
            - ``eps`` : regions on which performing the digital sums options are ``flower``, ``superflower`` or ``feb``.
            - ``threshold`` : threshold in ADC counts for the digital sum.

    Returns
    -------
    output : np.ndarray
        The trigger information with total amplitudeper FEB, with shape (163, time).
    """
    vals_thr = np.maximum(feb_sums.astype(np.int32) - l1_settings["threshold"], 0)
    vals_q = np.floor_divide(vals_thr, quant_step)
    return np.clip(vals_q, 0, 1023).astype(np.uint16)

class DLTriggerReader(DLWaveformReader):
    """
    A data reader class for handling R0 raw waveform data from telescopes.

    This class extends the ``DLWaveformReader`` to specifically handle the reading,
    transformation, and mapping of R0 raw waveform data. It supports ``mono`` data loading 
    mode and can multiple output settings for different trigger experiments.

    """

    output_settings = CaselessStrEnum(
        ["waveform", "balanced_patches", "all_patches", "double_patches"],
        default_value="waveform",
        help=(
            "Set the way to retrieve data from each event. "
            "``waveform``: extract the sequence of selected samples for the complete camera. "
            "``balanced_patches``: extract the sequence of selected samples for equilibrated number of cosmic and nsb patches. "
            "``all_patches``: extract the sequence of selected samples for all patches. "
            "``double_patches``: extract the sequence of selected samples for a random nsb patch and the patch containing the brightest pixel. "
        ),
    ).tag(config=True)

    output_size = CaselessStrEnum(
        ["patch", "sector", "camera"],
        default_value = "camera",
        help=(
            "Set the number of pixels in the output vector, patch and sector only available for the Advanced SiPM camera (AdvCam)."
            "``patch``: 343 pixels trigger patches."
            "``sector``: 2989 pixels patches, approx. 1/3 of the camera. "
            "``camera``: complete camera. "
        ),
    ).tag(config=True) 

    add_dummy_channel = Bool(
        default_value = False,
        allow_none = False,
        help=("Boolean variable to add or not an extra dummy dimension for the channel. This will allow the data to be passed and "
        "treated as 3 dimensional data to the DL package to perform 3D convolutions. This channel dimension is already created "
        "when selecting  ``flower_feb_quantised`` or '`stack'' as ``trigger_output'' configuration. "
        ),
    ).tag(config=True) 

    subtract_baseline = Int(
        default_value = None,
        allow_none = True,
        help=("Value in ADC counts to subtract to each pixel. ")
    ).tag(config=True) 

    quantisation_step = Int(
        default_value = 1,
        allow_none = True,
        help=("If quantisation per flower or FEB mode activated, step to quantise the data. ")
    ).tag(config=True)

    apply_trigger = CaselessStrEnum(
        ["l1", "tdscan"],
        default_value = None,
        allow_none = True,
        help=(
            "Variable to apply or not a low level trigger on the waveform patches."
            "``l1``: Apply a l1 sum trigger (per flower or superflower) on the waveforms. "
            "``tdscan``: Apply on top of the l1 trigger a Trigger Distributed Spatial Convolution Accelerator Network. "
        ),
    ).tag(config=True) 

    trigger_output = CaselessStrEnum(
        ["waveform", "mask", "stack", "binary", "feb_quantised", "flower_feb_quantised"],
        default_value = "waveform",
        allow_none = True,
        help=(
            "If apply trigger not none, variable to define the trigger output."
            "``mask``: Mask the input waveform with the flowers with positive trigger. "
            "``stack``: Add the trigger mask as a second channel. "
            "``binary``: Take exclusively the trigger mask as an output. "
        ),
    ).tag(config=True) 

    input_trigger_files = List(
        default_value = None,
        allow_none = True,
        help=(
            "h5 files with the obs_id, event_id, tel_id, trigger_mask, low_trigger and trigger_per_sample columns. "
            "These files can be generated with the code in https://github.com/jbuces/low_trigger "
        ),
    ).tag(config=True)

    trigger_length = Int(
        default_value = None,
        allow_none = True,
        help=("Sequence length to be retrieved in which there is a trigger. Only with apply trigger cuts")
    ).tag(config=True) 

    trigger_cuts = Bool(
        default_value = False,
        allow_none = False,
        help=("Boolean variable to filter the table with events with positive triggers")
    ).tag(config=True) 

    l1_settings = Dict(
        default_value = {"eps": "flower", "threshold": 2140},
        allow_none = True,
        help=(
            "Set the L1 trigger settings, only required when apply_trigger is not None. "
            "``eps``: Set whether to perform the ADC digital sum per ``flower`` or per ``superflower`` centered in each flower. "
            "``threshold``: Threshold in ADC counts above which flowers are going to be considered as triggered. ")
    ).tag(config=True)

    tdscan_settings = Dict(
        default_value ={"eps_xy":1, "eps_t":1, "min_pts":6},
        allow_none = True,
        help=(
            "Set the TDSCAN trigger settings, only required when apply_trigger is tdscan. "
            "``eps_xy``: Set the flower neighboring level to consider in the convolution. "
            "``eps_t``: Set the the number of samples before and after to consider in the convolution. "
            "``min_pts``: Threshold in binary flower counts above which the convolution is going to keep the central flower. ")
    ).tag(config=True)

    cpe_threshold = Int(
        default_value = 0,
        allow_none = False,
        help=("Threshold in simulated number of photoelectrons above which events are going to be labelled as shower or NSB")
    ).tag(config=True) 

    data_level = DataLevel.R0

    def __init__(
        self,
        input_url_signal,
        input_url_background=[],
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            input_url_signal=input_url_signal,
            input_url_background=input_url_background,
            config=config,
            parent=parent,
            **kwargs,
        )
        if self.trigger_length:
            final_time = self.trigger_length
            self.sequence_length = self.readout_length
        else:
            final_time = self.sequence_length
        # Load neighbors arrays and input shapes
        if self.output_settings in ["all_patches", "balanced_patches", "double_patches"]:
            if self.output_size == "patch":                 
                self.input_shape = (self.image_mappers[self.cam_name].patch_size, final_time)  
                self.neighbor_array = self.image_mappers[self.cam_name].neighbor_array
            elif self.output_size == "sector":
                self.input_shape = (self.image_mappers[self.cam_name].sector_size, final_time)
                self.neighbor_array = self.image_mappers[self.cam_name].sect0_neighbors
            else:
                raise ValueError(f"Output settings cannot be in patches mode: {self.output_settings} with output size in camera mode: {self.output_size}.")

        elif self.output_settings == "waveform" and self.output_size == "camera":
            # Different shapes in case HexagonalPatchMapper or other square pixel mapper is selected.
            if self.image_mapper_type != "HexagonalPatchMapper":
                self.input_shape = (
                    self.image_mappers[self.cam_name].image_shape,
                    self.image_mappers[self.cam_name].image_shape,
                    final_time,
                )
                self.neighbor_array = None
            elif self.apply_trigger is None or self.trigger_output in ["waveform", "mask", "stack"]:
                self.input_shape = (self.image_mappers[self.cam_name].n_pixels, final_time)
                self.neighbor_array = self.image_mappers[self.cam_name].cam_neighbor_array
            elif self.trigger_output in ["flower_feb_quantised"]:
                self.input_shape = (163, final_time, 2)
                self.neighbor_array = self.image_mappers[self.cam_name].feb_neighbors
            elif self.trigger_output in ["feb_quantised"]: 
                self.input_shape = (163, final_time)
                self.neighbor_array = self.image_mappers[self.cam_name].feb_neighbors
            elif self.trigger_output in ["binary"]:
                if self.l1_settings["eps"] in ["flower", "superflower"]:
                    self.input_shape = (1141, final_time)
                    self.neighbor_array = self.image_mappers[self.cam_name].fl_neighbor_array_tdscan
                else:
                    self.input_shape = (163, final_time)
                    self.neighbor_array = self.image_mappers[self.cam_name].feb_neighbors

        if self.add_dummy_channel:
            self.input_shape = self.input_shape + (1,)
        elif self.trigger_output == 'stack':
            self.input_shape = self.input_shape + (2,)

        if self.apply_trigger and self.cam_name != "AdvCamSiPM":
            raise ValueError(
                f"Apply trigger option only valid for the Adv SiPM camera, currently using '{self.cam_name}'."
            )
        if self.output_settings != "waveform" and self.cam_name != "AdvCamSiPM":
            raise ValueError(
                f"Patches options are only valid for the Adv SiPM camera, currently using '{self.cam_name}'."
            )
        if self.output_size != "camera" and self.trigger_output in ["feb_quantised", "flower_feb_quantised"]:
           raise ValueError(
                f"Output size {self.output_size} not compatible with trigger output {self.trigger_output}."
            )
        if self.image_mapper_type != "HexagonalPatchMapper" and self.trigger_output in ["feb_quantised", "flower_feb_quantised"]:
           raise ValueError(
                f"Only HexagonalPatchMapper method compatible with trigger output {self.trigger_output}, currently using {self.image_mapper_type}."
            )

    def _waveform_mode(self, true_image):
        # Retrieves the complete waveform, adds the number of pe and the class depending on pe.
        pe = np.int64(true_image.sum())
        lbl = np.int64(pe >= self.cpe_threshold)
        return {
            "patch_idx": [0],
            "cherenkov": [pe],
            "classes": [lbl],
        }

    def _all_patches_mode(self, true_image, sparse, n_patches):
        # Retrieves all patches per event, adds the number of pe and the class depending on pe.
        sums = sparse @ true_image
        labels = (sums > self.cpe_threshold).astype(np.int64)
        idxs = np.arange(n_patches, dtype=np.int64)
        return {
            "patch_idx": idxs,
            "cherenkov": sums.astype(np.int64),
            "classes": labels,
        }

    def _double_patches_mode(self, true_image, sparse, *args):
        # Retrieves the brightest patch and a random nsb patch per event.
        sums = sparse @ true_image
        # find brightest patch and one random NSB or fallback
        bright = np.argmax(sums)
        idxs, labels = [], []
        if sums[bright] >= self.cpe_threshold:
            idxs.append(bright); labels.append(1)
        nsb = np.where(sums == 0)[0]
        if nsb.size:
            idxs.append(np.random.choice(nsb)); labels.append(0)
        if not idxs:
            idxs.append(bright); labels.append(1)
        return {
            "patch_idx": np.array(idxs, np.int64),
            "cherenkov": sums[idxs].astype(np.int64),
            "classes": np.array(labels, np.int64),
        }

    def _balanced_patches(self, row_idxs, patch_idxs, ch_pe, patch_cls):
        # Works differently as the other methods, first retrieve all patches then select the exact
        # same number of shower and nsb patches.
        idx_bright = np.where(np.array(ch_pe) >= self.cpe_threshold)[0]
        idx_nsb = np.where(np.array(ch_pe) == 0)[0]
        n_min = min(len(idx_bright), len(idx_nsb))
        # Take the first n_min of each class (nsb > shower)
        idx_bright = idx_bright[:n_min]
        idx_nsb = idx_nsb[:n_min]
        balanced_idx = np.sort(np.concatenate([idx_bright, idx_nsb]))
        # Filter balanced data
        return (
            np.array(row_idxs)[balanced_idx],
            np.array(patch_idxs)[balanced_idx],
            np.array(ch_pe)[balanced_idx],
            np.array(patch_cls)[balanced_idx],
        )

    def _add_trigger_table(self, batch):
        # Add to the example identifiers the precomputed trigger information.
        # The h5 trigger files can be generated with the code in https://github.com/jbuces/low_trigger
        tdscan_tables = []

        # Packing info
        with tables.open_file(self.input_trigger_files[0], 'r') as h5file:
            node = h5file.get_node('/table')
            if hasattr(node._v_attrs, 'trigger_mask_shape'):
                self._trigger_mask_shape = tuple(node._v_attrs.trigger_mask_shape)
                self._trigger_mask_bits = int(node._v_attrs.trigger_mask_bits)
                self._trigger_mask_packed_len = int(node._v_attrs.trigger_mask_packed_len)
                self._trigger_mask_bitorder = str(node._v_attrs.trigger_mask_bitorder)

        for file_idx, trigger_file in enumerate(self.input_trigger_files):
            tdscan_table = read_table(trigger_file, "/table")
            if "trigger_per_patch" in tdscan_table.colnames:
                tdscan_table.remove_column("trigger_per_patch")
            if "max_pix" and "triggered_febs" in tdscan_table.colnames:
                tdscan_table.remove_column("max_pix")
                tdscan_table.remove_column("triggered_febs")
            # Add file index column
            tdscan_table.add_column(file_idx, name="file_index", index=0)
            tdscan_tables.append(tdscan_table)
            
        # Stack all trigger tables
        tdscan_table_all = vstack(tdscan_tables)

        n_rows = len(tdscan_table_all)
        if n_rows != len(batch):
            raise ValueError("The events tables and the loaded trigger tables have not the same length!")
        # Join stacked trigger table to batch
        merged = join(
            left=batch,
            right=tdscan_table_all,
            keys=["file_index", "obs_id", "event_id", "tel_id", "table_index", "true_energy"],
            join_type="left"
        )
        return merged

    def _get_raw_example(self, batch):
        """
        Adds the patch_index, patch_class and cherenkov_pe columns to the example identifiers
        depending on the selected 'output_settings'.

        This method processes the events in the example identifiers file and computes and adds to the 
        file a column with the patch_index, patch_class and true Cherenkov p.e. for each selected output setting. 
        Parameters
        ----------
        batch : astropy.table.Table
            A table containing information at minimum the following columns:
            - ``file_index``: List of indices corresponding to the files.
            - ``table_index``: List of indices corresponding to the event tables.
            - ``tel_type_id``: List of telescope type IDs.
            - ``tel_id``: List of telescope IDs.

        Returns
        -------
        batch : astropy.table.Table
            The input batch with the appended patch index, class and true Cherenkov p.e.
        """
        mapper = self.image_mappers[self.cam_name]
        sparse, n_patches = None, 0
        # Load the patches/sector info.
        if mapper.cam_neighbor_array is not None:
            patches = {
                "patch": mapper.trigger_patches,
                "sector": mapper.sectors_bool
            }.get(self.output_size)
            if patches is not None:
                sparse = csr_matrix(patches)
                n_patches = patches.shape[0]                

        records = []
        mode = self.output_settings
        if self.output_settings == "balanced_patches":
            mode = "all_patches"

        for row_idx, (file_idx, table_idx, tel_id) in enumerate(batch.iterrows(
            "file_index", "table_index", "tel_id")
        ):
            filename = list(self.files)[file_idx]
            tel_table = f"tel_{tel_id:03d}"
            with lock:
                sim_child = self.files[filename].root.simulation.event.telescope.images._f_get_child(tel_table)
                true_img = get_true_image(sim_child[table_idx])

            # Call functions depending on the output settings.
            if self.output_settings == 'waveform':
                out = self._waveform_mode(true_img)
            else:
                out = getattr(self, f"_{mode}_mode")(true_img, sparse, n_patches)

            for p, pe, cl in zip(out['patch_idx'], out['cherenkov'], out['classes']):
                records.append((row_idx, p, pe, cl))

        row_idxs, patch_idxs, ch_pe, patch_cls = zip(*records)
        self.one_class = np.all(patch_cls == 0) or np.all(patch_cls == 1)
        if self.output_settings == "balanced_patches":
            row_idxs, patch_idxs, ch_pe, patch_cls = self._balanced_patches(
                row_idxs, patch_idxs, ch_pe, patch_cls
            )

        batch = batch[list(row_idxs)]
        batch.add_column(np.array(patch_idxs),name="patch_index", index=6)
        batch.add_column(np.array(ch_pe),   name="cherenkov_pe", index=7)
        batch.add_column(np.array(patch_cls),name="patch_class", index=8)

        return batch

    def extract_triggered_window(self, waveforms, trigger_per_sample, true_show, window_size=10):
        """
        Extract the window of a given with the first trigger in the second sample.

        This method extracts the triggered window using the information from the trigger files.

        Parameters
        ----------
        waveform : r0_event : np.ndarray
            An array containing the complete raw waveform.
        trigger_per_sample : np.ndarray
            Binary array with the length of the simulated event indicating where does the trigger method triggered.
        true_show : bin
            1 for real event, 0 for NSB event.
        window_size : int
            Length of the window to extract.

        Returns
        -------
        window : np.ndarray
            Cropped array, with shape (num_pixels, window_size).
        """
        n_samples = waveforms.shape[1]
        center = n_samples // 2
        if true_show:
            search_start, search_end = center-10, center+11
        else:
            search_start, search_end = 0, n_samples - self.trigger_length + 1
        # Find the first triggered sample
        trigger_indices = np.where(trigger_per_sample[search_start:search_end] == 1)[0]
        # Global index
        trigger_index = trigger_indices[0] + search_start
        # Want it to be in position 1
        start = trigger_index - 1
        if start < 0:
            start = 0
        end = start + window_size
        # Just in case
        if end > n_samples:
            end = n_samples
            start = n_samples - window_size
        return waveforms[:, start:end]

    def _append_features(self, batch) -> Table:
        """
        Append waveforms to a given batch as features.

        This method processes a batch of events to append waveforms as input features for the neural networks.
        It reads the waveform data from the specified files and maps the waveforms using the appropriate ``ImageMapper``.
        Divides the waveform into patches and append the needed patch for a given sequence length.

        Parameters
        ----------
        batch : astropy.table.Table
            A table containing information at minimum the following columns:
            - ``file_index``: List of indices corresponding to the files.
            - ``table_index``: List of indices corresponding to the event tables.
            - ``tel_type_id``: List of telescope type IDs.
            - ``tel_id``: List of telescope IDs.
            - ``patch_index``: The index of the patch which will be processed.
            - ``patch_class``: 0 for patches with a number of Cherenkov p.e. above cpe_threshold, 1 for nsb patches.
            - ``cherenkov_pe``: Number of Cherenkov p.e. in the selected patch.

        Returns
        -------
        batch : astropy.table.Table
            The input batch with the appended following column:
            - ``waveforms``: Processed waveforms.
            
        """
        waveforms = []
        for i, (file_idx, table_idx, tel_type_id, tel_id, ptch_idx) in enumerate(batch.iterrows(
            "file_index", "table_index", "tel_type_id", "tel_id", "patch_index")
        ):
            filename = list(self.files)[file_idx]
            camera_type = self._get_camera_type(
                list(self.selected_telescopes.keys())[tel_type_id]
            )
            # Load only the r0 waveform.
            with lock:
                tel_table = f"tel_{tel_id:03d}"
                child = self.files[filename].root.r0.event.telescope._f_get_child(tel_table)
                unmapped_waveform = get_unmapped_waveform(
                    child[table_idx],
                    self.waveform_settings,
                    self.image_mappers[camera_type].geometry,
                    )
                waveform = unmapped_waveform.astype(np.int64)
                true_show = batch["cherenkov_pe"][i]>0
                if self.trigger_length:
                    waveform = self.extract_triggered_window(waveform, batch["trigger_per_sample"][i], true_show, window_size = self.trigger_length)
                # If requested apply trigger option for the complete waveform always (not patches), also valid for image mapping methods.
                if self.apply_trigger:
                    # If file provided get the mask from it.
                    if self.input_trigger_files and "tdscan" in self.apply_trigger and "trigger_mask" in batch.colnames:
                        packed = batch["trigger_mask"][i]   # 1D array uint8 length packed_len
                        flat = np.unpackbits(packed, bitorder="big")[:self._trigger_mask_bits]   # vector 0/1
                        mask = flat.reshape(self._trigger_mask_shape).astype(int)
                        mask = self.extract_triggered_window(mask, batch["trigger_per_sample"][i], true_show, window_size = self.trigger_length)
                    else:
                        # For all cases compute digital sum
                        mask, flower_sums = apply_digital_sum(waveform, self.image_mappers[camera_type], self.l1_settings) # shape: (n_trigger_regions, time)
                        # TDSCAN
                        if self.apply_trigger == 'tdscan':
                            mask = apply_tdscan(mask, self.image_mappers[camera_type], self.l1_settings, self.tdscan_settings) # shape: (n_trigger_regions, time)
                        rep = 49 if self.l1_settings["eps"] == "feb" else 7
                        bin_pixels = np.repeat(mask, rep, axis=0)  # shape: (n_pixels, time)
                        # Quantised data using flowers or FEBs
                        if self.l1_settings["eps"] == "feb" and self.trigger_output == "feb_quantised":
                            waveform = quantised_per_feb(flower_sums, self.l1_settings, self.quantisation_step) # shape: (n_febs, time)
                        elif self.l1_settings["eps"] in ['flower', 'superflower'] and self.trigger_output == "flower_feb_quantised":
                            waveform = quantised_per_flower(flower_sums, mask, self.l1_settings, self.quantisation_step) # shape: (n_febs, time, 2)
                    # Masked, binary or stacked output                    
                    if self.trigger_output == 'mask':
                        if self.subtract_baseline:
                            waveform -= self.subtract_baseline                            
                        waveform *= np.array(bin_pixels, dtype=np.int64) # shape: (n_pixels, time)
                    elif self.trigger_output == 'binary':
                        if self.image_mapper_type != "HexagonalPatchMapper" or self.output_size != "camera":
                            waveform = bin_pixels # shape: (n_pixels, time)
                        else:
                            waveform = np.array(mask) # shape: (n_trigger_regions, time)
                    elif self.trigger_output == 'stack':
                        if self.subtract_baseline:
                            waveform -= self.subtract_baseline
                        waveform = np.stack([waveform, bin_pixels], axis=-1) # shape: (n_pixels, time, 2)
                elif self.subtract_baseline:
                    waveform -= self.subtract_baseline
                # Apply the 'ImageMapper' whenever the index matrix is not None or 'HexagonalPatchMapper' not called (only for the 'waveform' option).
                # Otherwise, return the unmapped waveform if 'waveform' in ouput options.
                if (self.image_mappers[self.cam_name].cam_neighbor_array is None 
                    and self.image_mappers[camera_type].index_matrix is None
                ):
                    waveform = self.image_mappers[camera_type].map_image(waveform).astype(np.int64) 
                # If 'HexagonalPatchMapper' and one of the patches option, crop and reorder the image.
                elif (self.image_mappers[self.cam_name].cam_neighbor_array is not None 
                    and self.output_settings in ["all_patches", "balanced_patches", "double_patches"]
                ):
                    waveform = self.image_mappers[self.cam_name].get_reordered_patch(waveform, ptch_idx, self.output_size)
                
                # Option to add a dummy channel to perform 3D convolutions
                if self.add_dummy_channel:
                    waveform = np.expand_dims(waveform, axis=-1)
                waveforms.append(waveform)

        # Append everything to the selected batch
        batch.add_column(waveforms, name="features", index=8)
        if "waveform" in self.output_settings:
            batch.remove_column("patch_index")
        return batch

def get_feature_vectors(dl1_event, prefix, feature_vector_types) -> list:
    """
    Retrieve selected feature vectors from a DL1 event.

    This function processes the DL1 event data to retrieve feature vectors
    based on the specified feature vector types and prefix. It returns a list
    of feature vectors for the selected types, which can be used as input features
    for the neural networks.

    Parameters
    ----------
    dl1_event : astropy.table.Table
        A table containing DL1 event data, including feature vectors for classification,
        energy regression, and geometry/direction regression.
    prefix : str
        A prefix for the feature vector group in the HDF5 file.
    feature_vector_types : list of str
        A list of feature vector types to be loaded from the DL1 data, such as
        ``classification``, ``energy``, and ``geometry``.

    Returns
    -------
    feature_vectors : list of np.ndarray
        A list of feature vectors for the selected types.
    """
    feature_vectors = []
    for feature_vector_type in feature_vector_types:
        feature_vectors.append(
            dl1_event[f"{prefix}_{feature_vector_type}_feature_vectors"]
        )
    return feature_vectors


class DLFeatureVectorReader(DLDataReader):
    """
    A data reader class for handling DL1 feature vector data.

    This class extends the ``DLDataReader`` to specifically handle the reading of
    DL1 feature vectors, obtained from a previous CTLearnModel. It supports the reading
    of both ``telescope``- and ``subarray``-level feature vectors. This reader class only
    supports the reading in stereo mode.
    """

    prefixes = List(
        trait=Unicode(),
        default_value=["CTLearn"],
        allow_none=False,
        help="List of prefixes for the feature vector group in the HDF5 file.",
    ).tag(config=True)

    feature_vector_types = List(
        trait=CaselessStrEnum(
            [
                "classification",
                "energy",
                "geometry",
            ]
        ),
        allow_none=False,
        help=(
            "Set the type of the feature vector to be loaded from the DL1 data. "
            "classification: load feature vectors used for particle classification, "
            "energy: load feature vectors used for energy regression, "
            "geometry: load feature vectors used for geometry/direction regression."
        ),
    ).tag(config=True)

    load_telescope_features = Bool(
        default_value=True,
        help="Set whether to load telescope feature vectors from the DL1 data.",
    ).tag(config=True)

    load_subarray_features = Bool(
        default_value=False,
        help="Set whether to load subarray feature vectors from the DL1 data.",
    ).tag(config=True)

    def __init__(
        self,
        input_url_signal,
        input_url_background=[],
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            input_url_signal=input_url_signal,
            input_url_background=input_url_background,
            config=config,
            parent=parent,
            **kwargs,
        )
        # Check that the mode is consistent with the feature reader.
        # The feature reader only supports stereo mode.
        if self.mode != "stereo":
            raise ValueError(
                f"'{self.__class__.__name__}' only supports 'stereo' mode. "
                "Please set the mode to 'stereo' or use one of the other subclasses."
            )
        # Check that at least one of the feature vector types is selected
        if not self.load_telescope_features and not self.load_subarray_features:
            raise ValueError(
                "No loading of feature vectors selected. Please set 'load_telescope_features' "
                "and/or 'load_subarray_features' to 'True'."
            )

    def _append_features(self, batch) -> Table:
        """
        Append previous obtained feature vectors to a given batch as features.

        This method processes a batch of events to append feature vectors as input features
        for the neural networks. It reads the feature vector data from the specified files
        and appends the feature vectors to the batch. The feature vectors can be loaded
        for both ``telescope``- and ``subarray``-level.

        Parameters
        ----------
        batch : astropy.table.Table
            A table containing information at minimum the following columns:
            - "file_index": List of indices corresponding to the files.
            - "table_index": List of indices corresponding to the event tables.
            - "tel_type_id": List of telescope type IDs.
            - "tel_id": List of telescope IDs.

        Returns
        -------
        batch : astropy.table.Table
            The input batch with the appended mono and stereo feature vectors.
        """
        mono_fvs, stereo_fvs = [], []
        for file_idx, table_idx, tel_type_id, tel_id in batch.iterrows(
            "file_index", "table_index", "tel_type_id", "tel_id"
        ):
            filename = list(self.files)[file_idx]
            if self.load_telescope_features:
                with lock:
                    mono_fvs_per_prefix = []
                    tel_table = f"tel_{tel_id:03d}"
                    for prefix in self.prefixes:
                        telescope_child = (
                            self.files[filename]
                            .root.dl1.event.telescope.features.__getitem__(prefix)
                            ._f_get_child(tel_table)
                        )
                        mono_fvs_per_prefix.append(
                            get_feature_vectors(
                                telescope_child[table_idx],
                                f"{prefix}_tel",
                                self.feature_vector_types,
                            )
                        )
                mono_fvs.append(mono_fvs_per_prefix)
            if self.load_subarray_features:
                with lock:
                    stereo_fvs_per_prefix = []
                    for prefix in self.prefixes:
                        subarray_child = self.files[
                            filename
                        ].root.dl1.event.subarray.features._f_get_child(prefix)
                        stereo_fvs_per_prefix.append(
                            get_feature_vectors(
                                subarray_child[table_idx],
                                prefix,
                                self.feature_vector_types,
                            )
                        )
                stereo_fvs.append(stereo_fvs_per_prefix)
        # Append the features to the batch
        if self.load_telescope_features:
            batch.add_column(np.array(mono_fvs), name="mono_feature_vectors", index=7)
        if self.load_subarray_features:
            batch.add_column(
                np.array(stereo_fvs), name="stereo_feature_vectors", index=7
            )
        return batch
