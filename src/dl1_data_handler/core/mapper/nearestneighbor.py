"""
This module defines the ``ImageMapper`` subclass ``NearestNeighborMapper`` that hold the basic functionality for mapping raw 1D vectors into 2D mapped images using the nearest neighbor assignment technique.
"""

from ctapipe.core.traits import Int
from ctapipe.instrument.camera import PixelShape

from dl1_data_handler.core.mapper.image import ImageMapper

__all__ = ["NearestNeighborMapper"]


class NearestNeighborMapper(ImageMapper):
    """
    NearestNeighborMapper maps images from a hexagonal pixel grid to
    a square pixel grid using the nearest neighbor assignment technique.

    This class extends the functionality of ImageMapper by implementing
    methods to generate a nearest neighbor mapping table and perform the
    interpolation. It is particularly useful for applications where simplicity
    and computational efficiency is prioritized over interpolation accuracy.
    """

    interpolation_image_shape = Int(
        default_value=None,
        allow_none=True,
        help=(
            "Integer to overwrite the default shape of the resulting mapped image."
            "Only available for interpolation and rebinning methods."
        ),
    ).tag(config=True)

    def __init__(
        self,
        geometry,
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            geometry=geometry,
            config=config,
            parent=parent,
            **kwargs,
        )

        if geometry.pix_type != PixelShape.HEXAGON:
            raise ValueError(
                f"NearestNeighborMapper is only available for hexagonal pixel cameras. Pixel type of the selected camera is '{geometry.pix_type}'."
            )

        # At the edges of the cameras the mapping methods run into issues.
        # Therefore, we are using a default padding to ensure that the camera pixels aren't affected.
        # The default padding is removed after the conversion is finished.
        self.internal_pad = 3
        if self.interpolation_image_shape is not None:
            self.image_shape = self.interpolation_image_shape
        self.internal_shape = self.image_shape + self.internal_pad * 2
        # Creating the hexagonal and the output grid for the conversion methods.
        input_grid, output_grid = super()._get_grids_for_interpolation()
        # Calculate the mapping table
        self.mapping_table = super()._generate_nearestneighbor_table(
            input_grid, output_grid, pixel_weight=1.0
        )
