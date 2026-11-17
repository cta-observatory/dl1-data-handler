"""
This module defines the ``ImageMapper`` subclass ``OversamplingMapper`` that hold the basic functionality for mapping raw 1D vectors into 2D mapped images using the oversampling method.
"""

from ctapipe.instrument.camera import PixelShape
from dl1_data_handler.core import ImageMapper

__all__ = ["OversamplingMapper"]

class OversamplingMapper(ImageMapper):
    """
    OversamplingMapper maps images from a hexagonal pixel grid to
    a square pixel grid using oversampling techniques.

    This class extends the functionality of ImageMapper by implementing
    methods to generate an oversampling mapping table and perform the transformation.
    One hexganoal pixel is split into four square pixels, which are then weighted
    by one quarter of the intensity of the hexagonal pixel. The resulting
    image is stretched in one direction. It is particularly useful for applications
    where interpolation effects want to be surpressed.
    """

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
                f"OversamplingMapper is only available for hexagonal pixel cameras. Pixel type of the selected camera is '{geometry.pix_type}'."
            )

        self.internal_shape = self.image_shape
        # Creating the hexagonal and the output grid for the conversion methods.
        input_grid, output_grid = super()._get_grids_for_oversampling()
        # Calculate the mapping table
        self.mapping_table = super()._generate_nearestneighbor_table(
            input_grid, output_grid, pixel_weight=0.25
        )
