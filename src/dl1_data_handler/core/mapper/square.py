"""
This module defines the ``ImageMapper`` subclass ``SquareMapper`` that hold the basic functionality for mapping raw 1D vectors into 2D mapped images using the square pixel grid.
"""

import numpy as np

from ctapipe.instrument.camera import PixelShape
from dl1_data_handler.core import ImageMapper

__all__ = ["SquareMapper"]

class SquareMapper(ImageMapper):
    """
    SquareMapper maps images to a square pixel grid without any modifications.

    This class extends the functionality of ImageMapper by implementing
    methods to generate a direct mapping table and perform the transformation.
    It is particularly useful for applications where a direct one-to-one
    mapping is sufficient for converting pixel data.for square pixel cameras
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

        if geometry.pix_type != PixelShape.SQUARE:
            raise ValueError(
                f"SquareMapper is only available for square pixel cameras. Pixel type of the selected camera is '{geometry.pix_type}'."
            )

        # Set shape of the image for the square camera
        self.image_shape //= 2
        self.internal_shape = self.image_shape

        # Create square grid
        input_grid, output_grid = self._get_square_grid()
        # Calculate the mapping table
        self.mapping_table = super()._generate_nearestneighbor_table(
            input_grid, output_grid, pixel_weight=1.0
        )

    def _get_square_grid(
        self,
    ):
        """
        :return: two 2D numpy arrays (input grid and squared output grid)
        """
        # Create the virtual pixels outside of the camera with square pixels
        virtual_pixels = super()._get_virtual_pixels(
            self.x_ticks, self.y_ticks, self.pix_x, self.pix_y
        )
        pix_x = np.concatenate((self.pix_x, virtual_pixels[:, 0]))
        pix_y = np.concatenate((self.pix_y, virtual_pixels[:, 1]))
        # Stack the pixel positions to create the input grid and set output grid
        input_grid = np.column_stack([pix_x, pix_y])
        # Create the squared output grid
        x_grid = np.linspace(
            np.min(pix_x), np.max(pix_x), num=self.image_shape, endpoint=True
        )
        y_grid = np.linspace(
            np.min(pix_y), np.max(pix_y), num=self.image_shape, endpoint=True
        )
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
        output_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        return input_grid, output_grid
