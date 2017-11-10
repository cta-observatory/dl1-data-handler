from tables import IsDescription, UInt32Col, UInt8Col, Float64Col, Float32Col, StringCol


class Event(IsDescription):

    """
    Row descriptor class for Pytables event table.
    """
    event_number = UInt32Col()
    run_number = UInt32Col()
    gamma_hadron_label = UInt8Col()
    core_x = Float64Col()
    core_y = Float64Col()
    h_first_int = Float64Col()
    mc_energy = Float64Col()
    az = Float32Col()
    alt = Float32Col()
    reconstructed_energy = Float64Col()


class Tel(IsDescription):

    """
    Row descriptor class for Pytables telescope data table.
    """
    tel_id = UInt8Col()
    tel_x = Float32Col()
    tel_y = Float32Col()
    tel_z = Float32Col()
    tel_type = StringCol(8)
    run_array_direction = Float32Col(2)
