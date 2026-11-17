"""
This module defines the ``ImageMapper`` subclass ``AxialMapper`` that hold the basic functionality for the axial addressing scheme.
"""

import numpy as np
from ctapipe.core.traits import Bool
from ctapipe.instrument.camera import PixelShape
from scipy import spatial

from dl1_data_handler.core.mapper.image import ImageMapper

__all__ = ["AxialMapper"]


class AxialMapper(ImageMapper):
    """
    AxialMapper applies a transformation to axial coordinates to map images
    from a hexagonal pixel grid to a square pixel grid.

    This class extends the functionality of ImageMapper by implementing
    methods to generate an axial mapping table and perform the transformation.
    It is particularly useful for applications where axial coordinate
    transformations are required for mapping pixel data.
    """

    set_index_matrix = Bool(
        default_value=False,
        help=(
            "Whether to calculate and store the index matrix or not. "
            "For the 'IndexedConv' package, the index matrix is needed "
            "and the DLDataReader will return an unmapped image."
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
                f"AxialMapper is only available for hexagonal pixel cameras. Pixel type of the selected camera is '{geometry.pix_type}'."
            )
        # Creating the hexagonal and the output grid for the conversion methods.
        (
            input_grid,
            output_grid,
        ) = self._get_grids()
        # Set shape and padding for the axial addressing method
        self.internal_shape = self.image_shape
        # Calculate the mapping table
        self.mapping_table = super()._generate_nearestneighbor_table(
            input_grid, output_grid, pixel_weight=1.0
        )
        # Calculate and store the index matrix for the 'IndexedConv' package
        if self.set_index_matrix:
            tree = spatial.cKDTree(input_grid)
            nn_index = np.reshape(
                tree.query(output_grid)[1], (self.internal_shape, self.internal_shape)
            )
            nn_index[nn_index >= self.n_pixels] = -1
            self.index_matrix = np.flip(nn_index, axis=0)

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

        dist_first = np.around(
            abs(first_ticks[0] - first_ticks[1]),
            decimals=self.constants.decimal_precision,
        )
        dist_second = np.around(
            abs(second_ticks[0] - second_ticks[1]),
            decimals=self.constants.decimal_precision,
        )

        # manipulate y ticks with extra ticks
        num_extra_ticks = len(self.y_ticks)
        for i in np.arange(num_extra_ticks):
            second_ticks.append(
                np.around(
                    second_ticks[-1] + dist_second,
                    decimals=self.constants.decimal_precision,
                )
            )
        first_ticks = reversed(first_ticks)
        for shift, ticks in enumerate(first_ticks):
            for i in np.arange(len(second_pos)):
                if first_pos[i] == ticks and second_pos[i] in second_ticks:
                    second_pos[i] = second_ticks[
                        second_ticks.index(second_pos[i]) + shift
                    ]

        grid_first = np.unique(first_pos).tolist()
        grid_second = np.unique(second_pos).tolist()

        # Squaring the output image if grid axes have not the same length.
        if len(grid_first) > len(grid_second):
            for i in np.arange(len(grid_first) - len(grid_second)):
                grid_second.append(
                    np.around(
                        grid_second[-1] + dist_second,
                        decimals=self.constants.decimal_precision,
                    )
                )
        elif len(grid_first) < len(grid_second):
            for i in np.arange(len(grid_second) - len(grid_first)):
                grid_first.append(
                    np.around(
                        grid_first[-1] + dist_first,
                        decimals=self.constants.decimal_precision,
                    )
                )

        # Overwrite image_shape with the new shape of axial addressing
        self.image_shape = len(grid_first)

        # Create the virtual pixels outside of the camera.
        # This can not be done with general super()._create_virtual_hex_pixels() method
        # because for axial addressing the image is tilted and we need add extra ticks
        # to one axis (y-axis).
        virtual_pixels = super()._get_virtual_pixels(
            grid_first, grid_second, first_pos, second_pos
        )

        first_pos = np.concatenate((first_pos, np.array(virtual_pixels[:, 0])))
        second_pos = np.concatenate((second_pos, np.array(virtual_pixels[:, 1])))

        if len(self.x_ticks) < len(self.y_ticks):
            input_grid = np.column_stack([first_pos, second_pos])
            x_grid, y_grid = np.meshgrid(grid_first, grid_second)
        else:
            input_grid = np.column_stack([second_pos, first_pos])
            x_grid, y_grid = np.meshgrid(grid_second, grid_first)
        output_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])

        return input_grid, output_grid
