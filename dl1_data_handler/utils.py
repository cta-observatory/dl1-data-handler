import numpy as np
import ctapipe.image.cleaning as cleaning
from ctapipe.instrument import CameraGeometry


#################
# Event Filters #
#################

def event_intensity_filter(reader, file, i_min=-np.inf, i_max=np.inf):
    """
    Prototype of Filter events on intensity (in pe)
    Parameters
    ----------
    reader (DL1DataReader) : the reader to filter
    file
    i_min
    i_max

    Returns
    -------
    the filtered indices
    """
    # TODO define a physically correct strategy
    tel_types = [reader.tel_type] if reader.mode in ['mono', 'stereo'] else list(reader.selected_telescopes)
    total_intensity = np.zeros(len(file.root.Event_Info))
    for tel_type in tel_types:
        indices = file.root.Event_Info[:][tel_type + '_indices']
        images = file.root._f_get_child(tel_type)[:]['image_charge']
        images = images[indices]
        total_intensity += images.sum(axis=(1, 2))
    mask1 = i_min < total_intensity
    mask2 = total_intensity < i_max
    return set(np.arange(len(file.root.Event_Info))[mask1 & mask2])

def event_multiplicity_filter(reader, file, m_min=-np.inf, m_max=np.inf):
    """
    Prototype of Filter events on telescope multiplicity
    Parameters
    ----------
    reader (DL1DataReader) : the reader to filter
    file
    m_min
    m_max

    Returns
    -------
    the filtered indices
    """
    # FIXME This implementation only works for mono or stereo reader mode, otherwise it only
    # mask based on the last telescope type
    tel_types = [reader.tel_type] if reader.mode in ['mono', 'stereo'] else list(reader.selected_telescopes)
    multiplicity = np.zeros(len(file.root.Event_Info))
    for tel_type in tel_types:
        indices = file.root.Event_Info[:][tel_type + '_indices']
        multiplicity = np.count_nonzero(indices,axis=1)
    mask1 = m_min <= multiplicity
    mask2 = multiplicity <= m_max
    return set(np.arange(len(file.root.Event_Info))[mask1 & mask2])

#################
# Image Filters #
#################

def image_intensity_filter(reader, images, i_min=-np.inf, i_max=np.inf):
    """
    Filter images on intensity (in pe)
    Parameters
    ----------
    reader (DL1DataReader) : the reader to filter
    images
    i_min
    i_max

    Returns
    -------
    mask (Array of bool)
    """
    amps = images.sum(axis=-1)
    mask1 = i_min < amps
    mask2 = amps < i_max
    return mask1 & mask2


def image_cleaning_filter(reader, images, **opts):
    """
    Filter images according to a cleaning operation.

    Parameters
    ----------
    reader: `DL1DataReader`
    images

    Returns
    -------
    mask
    """

    geom = CameraGeometry.from_name(reader.tel_type.split('_')[1])

    def clean(img):
        return cleaning.tailcuts_clean(geom, img, **opts)

    clean_mask = np.apply_along_axis(clean, 1, images)
    return clean_mask.any(axis=1)

