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

            # Check if it matches the reference
            if not subarray.__eq__(self.subarray):
                if self.skip_incompatible_files:
                    self.log.warning(
                        f"Skipping '{filename}'. Subarray description does not match the reference subarray description."
                    )
                    del self.files[filename]
                else:
                    raise ValueError(
                        f"Subarray description of file '{filename}' does not match the reference subarray description."
                    )

        # Set the telescope type and camera name as class attributes for mono mode for convenience
        # FIXME Make image mapper not a dict because we only need one since we do not select multiple telescope types for image/wvf reading
        self.tel_type = list(self.selected_telescopes)[0]
        self.cam_name = self._get_camera_type(self.tel_type)
        # Initialize the ImageMapper with the pixel positions and mapping settings
        # TODO: Find a better way for passing the configuration
        self.image_mappers = {}
        cam_geom = {}
        if self.image_mapper_type is not None:
            for camera_type in self.subarray.camera_types:
                camera_name = self._get_camera_type(camera_type.name)
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
        self.class_weight = None
        if self.process_type == ProcessType.Simulation:
            if self.input_url_background:
                self.class_weight = {
                    0: (1.0 / self.n_bkg_events) * (self._get_n_events() / 2.0),
                    1: (1.0 / self.n_signal_events) * (self._get_n_events() / 2.0),
                }

    def _get_camera_type(self, tel_type):
        """Extract the camera type from the telescope type string."""
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
        self.example_identifiers.sort(["obs_id", "event_id", "tel_id", "tel_type_id"])
        # Construct simulation information for all files
        if self.process_type == ProcessType.Simulation:
            self.simulation_info = vstack(simulation_info)
            self.n_signal_events = np.count_nonzero(
                self.example_identifiers["true_shower_primary_class"] == 1
            )
            if self.input_url_background:
                self.n_bkg_events = np.count_nonzero(
                    self.example_identifiers["true_shower_primary_class"] == 0
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

    def get_parameters(self, batch, dl1b_parameter_list) -> np.array:
        """
        Retrieve DL1b parameters for a given batch of events.

        This method extracts the specified DL1b parameters for each event in the batch.

        Parameters:
        -----------
        batch : astropy.table.Table
            A Table containing the batch with columns ``file_index``, ``table_index``, and ``tel_id``.
        dl1b_parameter_list : list
            A list of DL1b parameters to be retrieved for each event.

        Returns:
        --------
        dl1b_parameters : np.array
            An array of DL1b parameters for the batch of events.
        """
        dl1b_parameters = []
        for file_idx, table_idx, tel_id in batch.iterrows(
            "file_index", "table_index", "tel_id"
        ):
            filename = list(self.files)[file_idx]
            with lock:
                tel_table = f"tel_{tel_id:03d}"
                child = self.files[
                    filename
                ].root.dl1.event.telescope.parameters._f_get_child(tel_table)
            parameters = list(child[table_idx][dl1b_parameter_list])
            dl1b_parameters.append([np.stack(parameters)])
        return np.array(dl1b_parameters)

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
            self.input_shape = (
                self.image_mappers[self.cam_name].image_shape,
                self.image_mappers[self.cam_name].image_shape,
                len(self.channels),
            )
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
            if self.image_mappers[camera_type].index_matrix is None:
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
        self.readout_length = int(
            self.files[self.first_file]
            .root.r1.event.telescope._f_get_child(f"tel_{self.tel_ids[0]:03d}")
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
            self.input_shape = (
                self.image_mappers[self.cam_name].image_shape,
                self.image_mappers[self.cam_name].image_shape,
                self.sequence_length,
            )
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
                self.files[self.first_file]
                .root.r1.event.telescope._f_get_child(f"tel_{self.tel_ids[0]:03d}")
                ._v_attrs
            )
        if "CTAFIELD_5_TRANSFORM_SCALE" in wvf_table_v_attrs:
            self.waveform_settings["waveform_scale"] = wvf_table_v_attrs[
                "CTAFIELD_5_TRANSFORM_SCALE"
            ]
            self.waveform_settings["waveform_offset"] = wvf_table_v_attrs[
                "CTAFIELD_5_TRANSFORM_OFFSET"
            ]

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
            if self.image_mappers[camera_type].index_matrix is None:
                waveforms.append(
                    self.image_mappers[camera_type].map_image(unmapped_waveform)
                )
            else:
                waveforms.append(unmapped_waveform)
        batch.add_column(waveforms, name="features", index=7)
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
