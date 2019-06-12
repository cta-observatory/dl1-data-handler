import numpy as np
import ctapipe.image.cleaning as cleaning
from ctapipe.instrument import CameraGeometry


#################
# Event Filters #
#################

def event_energy_filter(reader, file, e_min=-np.inf, e_max=np.inf):
    """
    Filter events on energy
    Parameters
    ----------
    reader
    file
    e_min
    e_max

    Returns
    -------
    the filtered indices
    """
    mask1 = (file.root.Events[:]['mc_energy'] > e_min) & (file.root.Events[:]['mc_energy'] < e_max)
    return set(np.arange(len(file.root.Events))[mask1])


def event_intensity_filter(reader, file, i_min=-np.inf, i_max=np.inf):
    """
    Filter events on intensity (in pe)
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
    tel_types = [reader.tel_type] if reader.mode == 'mono' else reader.tel_type
    total_intensity = np.zeros(len(file.root.Events))
    for tel_type in tel_types:
        indices = file.root.Events[:][tel_type + '_indices']
        images = file.root[tel_type][:]['charge']
        images = images[indices]
        total_intensity += images.sum(axis=(1, 2))
    mask1 = i_min < total_intensity
    mask2 = total_intensity < i_max
    return set(np.arange(len(file.root.Events))[mask1 & mask2])

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

