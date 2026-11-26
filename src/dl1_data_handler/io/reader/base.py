"""
This module defines the ``DLDataReader`` (abstract base class) and ``TableQualityQuery`` classes that hold the basic reading and processing functionality for Deep Learning (DL) analyses.
"""

__all__ = [
    "ProcessType",
    "TableQualityQuery",
    "DLDataReader",
]

import atexit
import threading
from abc import abstractmethod
from collections import OrderedDict
from enum import Enum

import numpy as np
import tables
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.table import (
    Table,
    join,
    unique,
    vstack,
)
from astropy.time import Time
from ctapipe.coordinates import CameraFrame, NominalFrame
from ctapipe.core import Component, QualityQuery
from ctapipe.core.traits import (
    Bool,
    CaselessStrEnum,
    CInt,
    Dict,
    Int,
    IntTelescopeParameter,
    List,
    Set,
    TelescopeParameter,
    Unicode,
    UseEnum,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.instrument.optics import FocalLengthKind
from ctapipe.io import read_table
from ctapipe.io.metadata import read_ctao_metadata
from traitlets import TraitError

from dl1_data_handler.core import ImageMapper

# Reference (dummy) time to insert in the SkyCoord object as the default time
LST_EPOCH = Time("2018-10-01T00:00:00", scale="utc")

lock = threading.Lock()


class ProcessType(Enum):
    Observation = "Observation"
    Simulation = "ObservationSim"


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
    ctao_metadata : ctao_datamodel.models.dataproducts.Product or tables.attributeset.AttributeSet
        Metadata from the first input file.
        With ctapipe v0.33.0 or later, this is a data product instance of the ctao_datamodel package;
        with older versions, it is an attribute set of the HDF5 file.
    process_type : enum
        The type of data processing (i.e. ``ProcessType.Observation`` or ``ProcessType.Simulation``).
    data_format_version : str
        The version of the ctapipe data format.
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

    subarray_check = CaselessStrEnum(
        values=["equality", "telescopes", "disabled"],
        default_value="equality",
        help=(
            "Specify the level of subarray compatibility checking between files. "
            "'equality' enforces strict equality of subarray descriptions. "
            "'telescopes' performs a looser check based on telescope IDs only. "
            "'disabled' disables subarray compatibility checks entirely. "
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

    # Column of the example identifiers with the class (1 signal, 0 background)
    # used to count the examples for the class weights.
    class_column = "true_shower_primary_class"

    def __init__(
        self,
        input_url_signal,
        input_url_background=[],
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(config=config, parent=parent, **kwargs)

        if not input_url_signal:
            raise TraitError("The 'input_url_signal' list cannot be empty.")

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
        # Load CTAO metadata from the first file as a reference
        self.ctao_metadata = read_ctao_metadata(self.first_file)
        self.process_type = ProcessType(self.ctao_metadata.data.type)
        self.data_format_version = self.ctao_metadata.model.version
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
        # Set the paths for the telescope and subarray trigger tables based on the data format version
        self.tel_trigger_table_path = "dl1/event/telescope/trigger"
        self.subarray_trigger_table_path = "dl1/event/subarray/trigger"
        if int(self.data_format_version.split(".")[0].replace("v", "")) >= 8:
            self.tel_trigger_table_path = "dl0/event/telescope/trigger"
            self.subarray_trigger_table_path = "dl0/event/subarray/trigger"

        # Set up the subarray
        self.subarray = SubarrayDescription.from_hdf(
            self.first_file, focal_length_choice=self.focal_length_choice
        )
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
        # Check that all files have compatible SubarrayDescriptions
        if self.subarray_check != "disabled":
            for filename in list(self.files):
                subarray = SubarrayDescription.from_hdf(
                    filename, focal_length_choice=self.focal_length_choice
                )
                # Filter subarray by selected telescopes
                if selected_tel_ids is not None:
                    subarray = subarray.select_subarray(self.tel_ids)
                # Check if the subarray matches the reference
                subarrays_match = (
                    subarray.__eq__(self.subarray)
                    if self.subarray_check == "equality"
                    else SubarrayDescription.check_matching_subarrays(
                        [self.subarray, subarray]
                    )
                )
                if not subarrays_match:
                    message = f"Subarray description of file '{filename}' does not match the reference subarray description."
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
                self.tel_trigger_table_path,
            )
            self.subarray_trigger_table = read_table(
                self.files[self.first_file],
                self.subarray_trigger_table_path,
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
                    "true_x_max",
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
        # The class weights are only computed when there are examples of both classes.
        self.class_weight = None
        if self.process_type == ProcessType.Simulation:
            if self.n_signal_events > 0 and self.n_bkg_events > 0:
                self.class_weight = {
                    0: (1.0 / self.n_bkg_events) * (self.get_n_events() / 2.0),
                    1: (1.0 / self.n_signal_events) * (self.get_n_events() / 2.0),
                }

    def _get_camera_type(self, tel_type):
        """Extract the camera type from the telescope type string."""
        if tel_type.split("_")[-1] == "UNKNOWN-7987PX":
            return "AdvCamSiPM"
        else:
            return tel_type.split("_")[-1]

    def get_n_events(self):
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
            trigger_table = read_table(f, self.subarray_trigger_table_path)
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
                node_path = f"/dl1/event/telescope/parameters/tel_{tel_id:03d}"
                if node_path not in f:
                    # Files can contain only a subset of the selected telescopes (see the subarray check).
                    self.log.warning(
                        f"Telescope {tel_id:03d} not found in '{filename}'. Skipping it."
                    )
                    continue
                tel_table = read_table(f, node_path)
                if self.force_dl1_lookup:
                    # Read the DL1 image table
                    dl1_tel_table = read_table(
                        f,
                        f"/dl1/event/telescope/images/tel_{tel_id:03d}",
                    )
                    # Keep only the columns needed for the join
                    dl1_tel_table.keep_columns(["obs_id", "event_id", "tel_id"])
                    # Add the table index to the DL1 image table
                    dl1_tel_table.add_column(
                        np.arange(len(dl1_tel_table)), name="table_index", index=0
                    )
                    # Unique the table to remove unwanted duplication
                    dl1_tel_table = unique(
                        dl1_tel_table, keys=["obs_id", "event_id", "tel_id"]
                    )
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
                # Join the telescope table with the trigger table
                tel_table = join(
                    left=tel_table,
                    right=trigger_table,
                    keys=["obs_id", "event_id"],
                )
                if self.process_type == ProcessType.Simulation:
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
            # Todo: Does not have to be class attribute. This needed at the moment
            # for real data which is processed per file.
            self.passes_quality_checks = np.ones(len(events), dtype=bool)
            # Quality selection based on the dl1b parameter and MC shower simulation tables
            if self.quality_query:
                self.passes_quality_checks = self.quality_query.get_table_mask(events)

            # Apply the mask to filter events that are not fulfilling the quality criteria
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

        # Construct the example identifiers for all files
        self.example_identifiers = vstack(example_identifiers)
        # Subclasses can label or select the examples (e.g. DLTriggerReader).
        self.example_identifiers = self._process_example_identifiers(
            self.example_identifiers
        )

        self.example_identifiers.sort(
            ["file_index", "obs_id", "event_id", "tel_id", "tel_type_id"]
        )
        # Construct simulation information for all files
        if self.process_type == ProcessType.Simulation:
            self.simulation_info = vstack(simulation_info)
            self.n_signal_events = np.count_nonzero(
                self.example_identifiers[self.class_column] == 1
            )
            self.n_bkg_events = np.count_nonzero(
                self.example_identifiers[self.class_column] == 0
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
            trigger_table = read_table(f, self.subarray_trigger_table_path)
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
                            f,
                            f"/dl1/event/telescope/images/tel_{tel_id:03d}",
                        )
                        # Keep only the columns needed for the join
                        dl1_tel_table.keep_columns(["obs_id", "event_id", "tel_id"])
                        # Add the table index to the DL1 image table
                        dl1_tel_table.add_column(
                            np.arange(len(dl1_tel_table)), name="table_index", index=0
                        )
                        # Unique the table to remove unwanted duplication
                        dl1_tel_table = unique(
                            dl1_tel_table, keys=["obs_id", "event_id", "tel_id"]
                        )
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

        # Construct the example identifiers for all files
        self.example_identifiers = vstack(example_identifiers)
        # Events are identified by (obs_id, event_id, file_index). The pair
        # (obs_id, event_id) alone is not unique across files, e.g. signal and
        # background productions can share run numbers and event IDs.
        self.example_identifiers.sort(
            ["obs_id", "event_id", "file_index", "tel_id", "tel_type_id"]
        )
        self.example_identifiers_grouped = self.example_identifiers.group_by(
            ["obs_id", "event_id", "file_index"]
        )
        # Unique example identifiers by events
        self.unique_example_identifiers = unique(
            self.example_identifiers, keys=["obs_id", "event_id", "file_index"]
        )
        self.unique_example_identifiers.remove_columns(
            ["table_index", "tel_type_id", "tel_id", "hillas_intensity"]
        )
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
            else:
                self.n_bkg_events = 0
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
        impact_radius = np.sqrt(table["true_core_x"] ** 2 + table["true_core_y"] ** 2)
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
            Table containing the labels and additional information for the batch examples.
        """
        # Check that the batch generation call is consistent with the mode
        if self.mode != "stereo":
            raise ValueError("Stereo batch generation is not supported in mono mode.")
        # Retrieve the batch from the example identifiers via group by
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
                ["obs_id", "event_id", "file_index", "true_shower_primary_class"]
            )
        elif self.process_type == ProcessType.Observation:
            batch_grouped = batch.group_by(["obs_id", "event_id", "file_index"])
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
        batch.sort(["obs_id", "event_id", "file_index", "tel_type_id", "tel_id"])
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
    def _process_example_identifiers(self, example_identifiers) -> Table:
        """
        Label or select the mono example identifiers.

        It is called once all files are read and before the examples are sorted.
        Subclasses that do not need it return the example identifiers unchanged.
        """
        pass

    @abstractmethod
    def _append_features(self, batch) -> Table:
        pass
