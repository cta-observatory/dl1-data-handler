"""Module for custom PyTables row classes.

These row classes below define the structure of the tables in
the output PyTables .h5 files.

"""

from tables import (IsDescription, UInt32Col, UInt16Col, UInt8Col,
                    Float64Col, Float32Col, StringCol)


class EventTableRow(IsDescription):

    """Row descriptor class for event table.

    Contains event-level information, mostly from Monte Carlo simulation
    parameters. NOTE: Additional columns are added dynamically to some tables,
    see the github wiki page for the full table/data format descriptions.

    Attributes
    ----------
    event_id : UInt32Col
        Shower event id.
    obs_id : UInt32Col
        Shower observation (run) id. Replaces "run_id"
    shower_primary_id : UInt8Col
        Particle type id for the shower primary particle. From Monte Carlo simulation parameters.
    core_x : Float64Col
        Shower core position x coordinate. From Monte Carlo simulation parameters.
    core_y : Float64Col
        Shower core position y coordinate. From Monte Carlo simulation parameters.
    h_first_int : Float64Col
        Height of shower primary particle first interaction. From Monte Carlo simulation parameters.
    mc_energy : Float64Col
        Energy of the shower primary particle. From Monte Carlo simulation parameters.
    az : Float32Col
        Shower azimuth angle. From Monte Carlo simulation parameters.
    alt : Float32Col
        Shower altitude (zenith) angle. From Monte Carlo simulation parameters.
    """
    event_id = UInt32Col()
    obs_id = UInt32Col()
    shower_primary_id = UInt8Col()
    core_x = Float32Col()
    core_y = Float32Col()
    h_first_int = Float32Col()
    mc_energy = Float32Col()
    az = Float32Col()
    alt = Float32Col()


class TelTableRow(IsDescription):
    """Row descriptor class for telescope type table.

    Contains parameter information for each telescope type in the data.  NOTE:
    Additional columns are added dynamically to some tables,
    see the github wiki page for the full table/data format descriptions.

    Attributes
    ----------
    type : StringCol
        Telescope type name (i.e. 'LST:LSTCam')
    num_pixels: UInt32Col
        Number of pixels in the telescope camera.
    """

    type = StringCol(20)
    num_pixels = UInt32Col()

class ArrayTableRow(IsDescription):
    """Row descriptor class for telescope array table.

    Contains parameter information for each telescope
    in the array. NOTE: Additional columns are added dynamically to some tables,
    see the github wiki page for the full table/data format descriptions.

    Attributes
    ----------
    id : UInt8Col
        Telescope id (unique).
    x : Float32Col
        Telescope position x coordinate relative to the center of the array.
    y : Float32Col
        Telescope position y coordinate relative to the center of the array.
    z : Float32Col
        Telescope position z coordinate (height) relative to the CORSIKA observatory altitude.
    type : StringCol
        Telescope type name (i.e. 'LST:LSTCam').
    camera : StringCol
        Telescope camera type name (i.e. 'LSTCam').
    optics : StringCol
        Telescope optics type name (i.e. 'LST').
    """

    id = UInt16Col()
    x = Float32Col()
    y = Float32Col()
    z = Float32Col()
    type = StringCol(20)
    camera = StringCol(20)
    optics = StringCol(20)
