"""
This module defines the ``ImageMapper`` subclass ``BilinearMapper`` that hold the basic functionality for mapping raw 1D vectors into 2D mapped images using the bilinear interpolation method.
"""

import numpy as np
from ctapipe.core.traits import Int
from ctapipe.instrument.camera import PixelShape
from scipy import spatial

from dl1_data_handler.core import ImageMapper

__all__ = ["BilinearMapper"]


class BilinearMapper(ImageMapper):
    """
    BilinearMapper maps images from a hexagonal pixel grid to
    a square pixel grid using bilinear interpolation.

    This class extends the functionality of ImageMapper by implementing
    methods to generate a bilinear interpolation mapping table and perform the transformation.
    It leverages Delaunay triangulation to find the nearest neighbors for the interpolation process.
    The mapping table is normalized to ensure that the intensity of the pixels is preserved.
    It is particularly useful for applications where smooth and continuous mapping
    of pixel data is required. Recommended to use as default for hexagonal pixel cameras.
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
                f"BilinearMapper is only available for hexagonal pixel cameras. Pixel type of the selected camera is '{geometry.pix_type}'."
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
        self.mapping_table = self._generate_table(input_grid, output_grid)

    def _generate_table(self, input_grid, output_grid):
        # Drawing Delaunay triangulation on the hex grid
        tri = spatial.Delaunay(input_grid)
        corner_indexes = tri.simplices[tri.find_simplex(output_grid)]
        corner_points = input_grid[corner_indexes]
        weights = super()._get_weights(corner_points, output_grid)
        weights = weights.reshape(self.internal_shape, self.internal_shape, -1)
        corner_indexes = corner_indexes.reshape(
            self.internal_shape, self.internal_shape, -1
        )
        # Construct the mapping matrix from the calculated weights
        mapping_matrix = np.zeros(
            (input_grid.shape[0], self.internal_shape, self.internal_shape),
            dtype=np.float32,
        )
        for j in range(self.internal_shape):
            for i in range(self.internal_shape):
                mapping_matrix[corner_indexes[j, i], j, i] = weights[j, i]
        return super()._get_sparse_mapping_matrix(mapping_matrix, normalize=True)
