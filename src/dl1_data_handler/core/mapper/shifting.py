"""
This module defines the ``ImageMapper`` classes that hold the basic functionality for mapping raw 1D vectors into 2D mapped images.
"""

import numpy as np
from ctapipe.instrument.camera import PixelShape

from dl1_data_handler.core import ImageMapper

__all__ = ["ShiftingMapper"]


class ShiftingMapper(ImageMapper):
    """
    ShiftingMapper applies a shifting transformation to map images
    from a hexagonal pixel grid to a square pixel grid.

    This class extends the functionality of ImageMapper by implementing
    methods to generate a shifting mapping table and perform the transformation.
    It is particularly useful for applications where a simple shift-based
    transformation is sufficient for mapping hexagonal pixel data.
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
                f"ShiftingMapper is only available for hexagonal pixel cameras. Pixel type of the selected camera is '{geometry.pix_type}'."
            )

        # Creating the hexagonal and the output grid for the conversion methods.
        input_grid, output_grid = self._get_grids()
        # Set shape for the axial addressing method
        self.internal_shape = self.image_shape
        # Calculate the mapping table
        self.mapping_table = super()._generate_nearestneighbor_table(
            input_grid, output_grid, pixel_weight=1.0
        )

    def _get_grids(
        self,
    ):
        """
        :param pos: a 2D numpy array of pixel positions, which were taken from the CTApipe.
        :param camera_type: a string specifying the camera type
        :param grid_size_factor: a number specifying the grid size of the output grid. Only if 'rebinning' is selected, this factor differs from 1.
        :return: two 2D numpy arrays (hexagonal grid and squared output grid)
        """

        # Check orientation of the hexagonal pixels
        first_ticks, first_pos, second_ticks, second_pos = (
            (self.x_ticks, self.pix_x, self.y_ticks, self.pix_y)
            if len(self.x_ticks) < len(self.y_ticks)
            else (self.y_ticks, self.pix_y, self.x_ticks, self.pix_x)
        )
        # Create the virtual pixels outside of the camera with hexagonal pixels
        (
            first_pos,
            second_pos,
            dist_first,
            dist_second,
        ) = super()._create_virtual_hex_pixels(
            first_ticks, second_ticks, first_pos, second_pos
        )
        # Get the number of extra ticks
        tick_diff = len(first_ticks) * 2 - len(second_ticks)
        tick_diff_each_side = tick_diff // 2
        # Extend second_ticks on both sides
        for _ in np.arange(tick_diff_each_side):
            second_ticks.append(
                np.around(
                    second_ticks[-1] + dist_second,
                    decimals=self.constants.decimal_precision,
                )
            )
            second_ticks.insert(
                0,
                np.around(
                    second_ticks[0] - dist_second,
                    decimals=self.constants.decimal_precision,
                ),
            )
        # If tick_diff is odd, add one more tick to the beginning
        if tick_diff % 2 != 0:
            second_ticks.insert(
                0,
                np.around(
                    second_ticks[0] - dist_second,
                    decimals=self.constants.decimal_precision,
                ),
            )
        # Create the input and output grid
        for i in np.arange(len(second_pos)):
            if second_pos[i] in second_ticks[::2]:
                second_pos[i] = second_ticks[second_ticks.index(second_pos[i]) + 1]
        grid_first = np.unique(first_pos).tolist()
        # Overwrite image_shape with the new shape of axial addressing
        self.image_shape = len(grid_first)
        grid_second = np.unique(second_pos).tolist()
        if len(self.x_ticks) < len(self.y_ticks):
            input_grid = np.column_stack([first_pos, second_pos])
            x_grid, y_grid = np.meshgrid(grid_first, grid_second)
        else:
            input_grid = np.column_stack([second_pos, first_pos])
            x_grid, y_grid = np.meshgrid(grid_second, grid_first)
        output_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        return input_grid, output_grid
