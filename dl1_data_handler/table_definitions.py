"""Describe table row formats for CTAMLDataDumper output files.

These row classes below define the structure of the tables in
the output PyTables .h5 files.

"""

from tables import (
    IsDescription,
    UInt32Col,
    UInt16Col,
    UInt8Col,
    Float32Col,
    StringCol,
    Int32Col,
)


class EventTableRow(IsDescription):
    """Describe row format for event table.

    Contains event-level information, mostly from Monte Carlo simulation
    parameters. NOTE: Additional columns are added dynamically to some tables,
    see the github wiki page for the full table/data format descriptions.

    Attributes
    ----------
    event_id : tables.UInt32Col
        Shower event id.
    obs_id : tables.UInt32Col
        Shower observation (run) id. Replaces old "run_id" in ctapipe r0
        container.
    shower_primary_id : tables.UInt8Col
        Particle type id for the shower primary particle. From Monte Carlo
        simulation parameters.
    core_x : tables.Float32Col
        Shower core position x coordinate. From Monte Carlo simulation
        parameters.
    core_y : tables.Float32Col
        Shower core position y coordinate. From Monte Carlo simulation
        parameters.
    h_first_int : tables.Float32Col
        Height of shower primary particle first interaction. From Monte Carlo
        simulation parameters.
    mc_energy : tables.Float32Col
        Energy of the shower primary particle in TeV. From Monte Carlo simulation
        parameters.
    log_mc_energy : tables.Float32Col
        Energy of the shower primary particle in log(TeV). From Monte Carlo simulation
        parameters.
    az : tables.Float32Col
        Shower azimuth angle. From Monte Carlo simulation parameters.
    alt : tables.Float32Col
        Shower altitude (zenith) angle. From Monte Carlo simulation parameters.
    array_pointing_az : tables.Float32Col
        Array pointing azimuth angle.
    array_pointing_alt : tables.Float32Col
        Array pointing altitude (zenith) angle.
    delta_direction : tables.Float32Col(2)
        Angular distance of the shower azimuth and the array pointing.

    """

    event_id = UInt32Col()
    obs_id = UInt32Col()
    shower_primary_id = UInt8Col()
    core_x = Float32Col()
    core_y = Float32Col()
    h_first_int = Float32Col()
    x_max = Float32Col()
    mc_energy = Float32Col()
    log_mc_energy = Float32Col()
    az = Float32Col()
    alt = Float32Col()
    array_pointing_az = Float32Col()
    array_pointing_alt = Float32Col()
    delta_direction = Float32Col(2)


class TelTableRow(IsDescription):
    """Describe row format for telescope type table.

    Contains parameter information for each telescope type in the data.  NOTE:
    Additional columns are added dynamically to some tables,
    see the github wiki page for the full table/data format descriptions.

    Attributes
    ----------
    type : tables.StringCol
        Telescope type name (i.e. 'LST:LSTCam')
    optics : tables.StringCol
        Telescope optics type name (i.e. 'LST').
    camera : tables.StringCol
        Telescope camera type name (i.e. 'LSTCam').
    num_pixels: tables.UInt32Col
        Number of pixels in the telescope camera.
    pix_rotation: tables.Float32Col
        Rotation angle in deg.
    cam_rotation: tables.Float32Col
        Overall camera rotation in deg.
    """

    type = StringCol(20)
    optics = StringCol(20)
    camera = StringCol(20)
    num_pixels = UInt32Col()
    pix_rotation = Float32Col()
    cam_rotation = Float32Col()


class ArrayTableRow(IsDescription):
    """Describe row format for telescope array table.

    Contains parameter information for each telescope in the array.
    NOTE: Additional columns are added dynamically to some tables, see the
    github wiki page for the full table/data format descriptions.

    Attributes
    ----------
    id : tables.UInt8Col
        Telescope id (unique).
    type : tables.StringCol
        Telescope type name (i.e. 'LST:LSTCam').
    x : tables.Float32Col
        Telescope position x coordinate relative to the center of the array.
    y : tables.Float32Col
        Telescope position y coordinate relative to the center of the array.
    z : tables.Float32Col
        Telescope position z coordinate (height) relative to the CORSIKA
        observatory altitude.

    """

    id = UInt16Col()
    type = StringCol(20)
    x = Float32Col()
    y = Float32Col()
    z = Float32Col()


class ParametersTableRow(IsDescription):
    """Describe row format for parameter table.

    Contains parameters values for each image of each event of each telescope in the array.
    Parameters are calculated after image cleaning (i.e. with for example tailcut_clean method)
    There are Hillas, Leakage, Concentration, Timing and Morphology parameters.

    Attributes
    ----------
    event_index : tables.Int32Col
        Event id of file (from -1 to N )

    leakage_* : tables.Float32Col
        see at https://cta-observatory.github.io/ctapipe/api/ctapipe.containers.LeakageContainer.html?highlight=leakagecontainer#ctapipe.containers.LeakageContainer

    hillas_* : tables.Float32Col
        see at https://cta-observatory.github.io/ctapipe/api/ctapipe.containers.HillasParametersContainer.html#ctapipe.containers.HillasParametersContainer

    concentration_* :
        see at https://cta-observatory.github.io/ctapipe/api/ctapipe.containers.ConcentrationContainer.html#ctapipe.containers.ConcentrationContainer

    timing_* :
        see at https://cta-observatory.github.io/ctapipe/api/ctapipe.containers.TimingParametersContainer.html#ctapipe.containers.TimingParametersContainer

    morphology_* :
        see at https://cta-observatory.github.io/ctapipe/api/ctapipe.containers.MorphologyContainer.html#ctapipe.containers.MorphologyContainer

    log_hillas_intensity : tables.Float32Col
        logaritmic hillas intensity

    """

    event_index = Int32Col()

    leakage_intensity_1 = Float32Col()
    leakage_intensity_2 = Float32Col()
    leakage_pixels_1 = Float32Col()
    leakage_pixels_2 = Float32Col()

    hillas_intensity = Float32Col()
    hillas_log_intensity = Float32Col()
    hillas_x = Float32Col()
    hillas_y = Float32Col()
    hillas_r = Float32Col()
    hillas_phi = Float32Col()
    hillas_width = Float32Col()
    hillas_length = Float32Col()
    hillas_psi = Float32Col()
    hillas_skewness = Float32Col()
    hillas_kurtosis = Float32Col()

    concentration_cog = Float32Col()
    concentration_core = Float32Col()
    concentration_pixel = Float32Col()

    timing_slope = Float32Col()  # time gradient
    timing_slope_err = Float32Col()
    timing_intercept = Float32Col()
    timing_intercept_err = Float32Col()
    timing_deviation = Float32Col()

    morphology_num_pixels = Int32Col()
    morphology_num_islands = Int32Col()
    morphology_num_small_islands = Int32Col()
    morphology_num_medium_islands = Int32Col()
    morphology_num_large_islands = Int32Col()
