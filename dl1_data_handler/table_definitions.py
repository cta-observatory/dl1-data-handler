"""Describe table row formats for CTAMLDataDumper output files.

These row classes below define the structure of the tables in
the output PyTables .h5 files.

"""

from tables import (IsDescription, UInt32Col, UInt16Col, UInt8Col,
                    Float32Col, StringCol)


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
        Energy of the shower primary particle. From Monte Carlo simulation
        parameters.
    az : tables.Float32Col
        Shower azimuth angle. From Monte Carlo simulation parameters.
    alt : tables.Float32Col
        Shower altitude (zenith) angle. From Monte Carlo simulation parameters.

    """

    event_id = UInt32Col()
    obs_id = UInt32Col()
    shower_primary_id = UInt8Col()
    core_x = Float32Col()
    core_y = Float32Col()
    h_first_int = Float32Col()
    x_max = Float32Col()
    mc_energy = Float32Col()
    az = Float32Col()
    alt = Float32Col()


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

    """

    type = StringCol(20)
    optics = StringCol(20)
    camera = StringCol(20)
    num_pixels = UInt32Col()


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
