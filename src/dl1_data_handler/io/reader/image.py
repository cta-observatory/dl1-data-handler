"""
This module defines the ``DLDataReader`` subclass ``DLImageReader`` that hold the basic reading and processing functionality for image data.
"""

__all__ = [
    "DLImageReader",
    "get_unmapped_image",
]

import numpy as np
import threading
from astropy.table import Table

from ctapipe.core.traits import (
    List,
    CaselessStrEnum,
)
from dl1_data_handler.io import DLDataReader

lock = threading.Lock()

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
        and ``relative_``-prefix for the relative peak arrival times. The ``log``-prefix
        can be used to apply a logarithmic transformation to the image data.
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
        # Apply the cleaning mask to the image if specified
        if "cleaned" in channel:
            image[:, i] *= mask
        # Apply the logarithmic transformation to the image if specified
        if "log" in channel:
            # Only apply log to positive values
            valid = image[:, i] > 0
            image[valid, i] = np.log10(image[valid, i])
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
                "log_image",
                "cleaned_image",
                "log_cleaned_image",
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
            "log_image: logarithm of the integrated charges, "
            "cleaned_image: integrated charges cleaned with the DL1 cleaning mask, "
            "log_cleaned_image: logarithm of the integrated charges cleaned with the DL1 cleaning mask, "
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

    def _process_example_identifiers(self, example_identifiers) -> Table:
        """No additional labelling or selection of the example identifiers."""
        return example_identifiers
