"""Module for custom PyTables row classes.

The row classes below define the structure of the tables in
the output file.

"""

from tables import (IsDescription, UInt32Col, UInt16Col, UInt8Col,
                    Float64Col, Float32Col, StringCol)


class Event(IsDescription):

    """Row descriptor class for Pytables event table.

    Contains event-level parameters, mostly from Monte Carlo simulation
    parameters.

    Attributes
    ----------
    event_number : UInt32Col
        UInt32 placeholder type for the shower event id
    run_number : UInt32Col
        UInt32 placeholder type for the run number
    particle_id : UInt8Col
        UInt8 placeholder type for the CORSIKA-simulated primary particle type
        code.
    core_x : Float64Col
        Float64 placeholder type for the Monte Carlo shower core
        position x coordinate.
    core_y : Float64Col
        Float64 placeholder type for the Monte Carlo shower core
        position y coordinate.
    h_first_int : Float64Col
        Float64 placeholder type for the Monte Carlo height of first
        interaction of the primary particle.
    mc_energy : Float64Col
        Float64 placeholder type for the Monte Carlo energy of the
        primary particle.
    az : Float32Col
        Float32 placeholder type for the shower azimuth angle.
    alt : Float32Col
        Float32 placeholder type for the shower altitude (zenith) angle
    """
    event_number = UInt32Col()
    run_number = UInt32Col()
    particle_id = UInt8Col()
    core_x = Float32Col()
    core_y = Float32Col()
    h_first_int = Float32Col()
    mc_energy = Float32Col()
    az = Float32Col()
    alt = Float32Col()


class Tel(IsDescription):
    """Row descriptor class for Pytables telescope data table.

    Contains parameter information for each selected telescope
    in the data.

    Attributes
    ----------
    tel_type : StringCol
        String placeholder type for the telescope type name (i.e. 'LST')
    num_pixels: UInt32Col
        UInt32 placeholder type for telescope's number of pixels
    pixel_pos: Float32Col
        Float32Col placeholder type for pixel's coordinates
    """

    tel_type = StringCol(8)
    num_pixels = UInt32Col()
    pixel_pos = Float32Col(2)

class Array(IsDescription):
    """Row descriptor class for Pytables array data table.

    Contains parameter information for each selected telescope
    in the data.

    Attributes
    ----------
    tel_id : UInt8Col
        UInt8 placeholder type for the telescope id (in the array)
    tel_x : Float32Col
        Float32 placeholder type for the telescope position x coordinate
        relative to the center of the array.
    tel_y : Float32Col
        Float32 placeholder type for the telescope position y coordinate
        relative to the center of the array.
    tel_z : Float32Col
        Float32 placeholder type for the telescope position z coordinate
        (height) relative to the CORSIKA observatory altitude.
    tel_type : StringCol
        String placeholder type for the telescope type name (i.e. 'LST')
    run_array_direction: Float32Col(2)
        Float32 tuple placeholder type for the array pointing direction for
        a given run (az,alt)
    """

    tel_id = UInt16Col()
    tel_x = Float32Col()
    tel_y = Float32Col()
    tel_z = Float32Col()
    tel_type = StringCol(8)
    run_array_direction = Float32Col(2)
