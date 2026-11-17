"""
This module defines the ``ImageMapper`` subclass ``BicubicMapper`` that hold the basic functionality for mapping raw 1D vectors into 2D mapped images using bicubic interpolation.
"""

import numpy as np
from ctapipe.core.traits import Int
from ctapipe.instrument.camera import PixelShape
from scipy import spatial

from dl1_data_handler.core import ImageMapper

__all__ = ["BicubicMapper"]


class BicubicMapper(ImageMapper):
    """
    BicubicMapper maps images from a hexagonal pixel grid to
    a square pixel grid using bicubic interpolation.

    This class is used to generate a mapping table that maps input grid points
    to output grid points using bicubic interpolation. The mapping table is
    normalized to ensure that the intensity of the pixels is preserved. It
    leverages Delaunay triangulation to find the nearest neighbors and second
    nearest neighbors for the interpolation process.
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
                f"BicubicMapper is only available for hexagonal pixel cameras. Pixel type of the selected camera is '{geometry.pix_type}'."
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
        #
        #                 /\        /\
        #                /  \      /  \
        #               /    \    /    \
        #              / 2NN  \  / 2NN  \
        #             /________\/________\
        #            /\        /\        /\
        #           /  \  NN  /  \  NN  /  \
        #          /    \    /    \    /    \
        #         / 2NN  \  /  .   \  /  2NN \
        #        /________\/________\/________\
        #                 /\        /\
        #                /  \  NN  /  \
        #               /    \    /    \
        #              / 2NN  \  / 2NN  \
        #             /________\/________\
        #

        # Drawing Delaunay triangulation on the hex grid
        tri = spatial.Delaunay(input_grid)

        # Get all relevant simplex indices
        simplex_index = tri.find_simplex(output_grid)
        simplex_index_NN = tri.neighbors[simplex_index]
        simplex_index_2NN = tri.neighbors[simplex_index_NN]
        table_simplex = tri.simplices[simplex_index]

        # NN
        weights_NN = []
        simplexes_NN = []
        for i in np.arange(simplex_index.shape[0]):
            if -1 in simplex_index_NN[i] or all(
                ind >= self.n_pixels for ind in table_simplex[i]
            ):
                w = np.array([0, 0, 0])
                weights_NN.append(w)
                corner_simplexes_2NN = np.array([-1, -1, -1])
                simplexes_NN.append(corner_simplexes_2NN)
            else:
                corner_points_NN, corner_simplexes_NN = self._get_triangle(
                    tri, input_grid, simplex_index_NN[i], table_simplex[i]
                )
                target = output_grid[i]
                target = np.expand_dims(target, axis=0)
                w = super()._get_weights(corner_points_NN, target)
                w = np.squeeze(w, axis=0)
                weights_NN.append(w)
                simplexes_NN.append(corner_simplexes_NN)

        weights_NN = np.array(weights_NN)
        simplexes_NN = np.array(simplexes_NN)

        # 2NN
        weights_2NN = []
        simplexes_2NN = []
        for i in np.arange(3):
            weights = []
            simplexes = []
            for j in np.arange(simplex_index.shape[0]):
                table_simplex_NN = tri.simplices[simplex_index_NN[j][i]]
                if (
                    -1 in simplex_index_2NN[j][i]
                    or -1 in simplex_index_NN[j]
                    or all(ind >= self.n_pixels for ind in table_simplex_NN)
                ):
                    w = np.array([0, 0, 0])
                    weights.append(w)
                    corner_simplexes_2NN = np.array([-1, -1, -1])
                    simplexes.append(corner_simplexes_2NN)
                else:
                    corner_points_2NN, corner_simplexes_2NN = self._get_triangle(
                        tri, input_grid, simplex_index_2NN[j][i], table_simplex_NN
                    )
                    target = output_grid[j]
                    target = np.expand_dims(target, axis=0)
                    w = super()._get_weights(corner_points_2NN, target)
                    w = np.squeeze(w, axis=0)
                    weights.append(w)
                    simplexes.append(corner_simplexes_2NN)

            weights = np.array(weights)
            simplexes = np.array(simplexes)
            weights_2NN.append(weights)
            simplexes_2NN.append(simplexes)

        weights_2NN.append(weights_NN)
        simplexes_2NN.append(simplexes_NN)
        weights_2NN = np.array(weights_2NN)
        simplexes_2NN = np.array(simplexes_2NN)
        weights = np.reshape(
            weights_2NN,
            (
                weights_2NN.shape[0],
                self.internal_shape,
                self.internal_shape,
                weights_2NN.shape[2],
            ),
        )
        corner_indexes = np.reshape(
            simplexes_2NN,
            (
                simplexes_2NN.shape[0],
                self.internal_shape,
                self.internal_shape,
                simplexes_2NN.shape[2],
            ),
        )
        # Construct the mapping matrix from the calculated weights
        mapping_matrix = np.zeros(
            (input_grid.shape[0], self.internal_shape, self.internal_shape),
            dtype=np.float32,
        )
        for i in range(4):
            for j in range(self.internal_shape):
                for k in range(self.internal_shape):
                    for corner in range(weights.shape[3]):
                        index = (
                            corner_indexes[i][k][j][corner]
                            if weights.shape[3] == 3
                            else corner_indexes[k][j][i][corner]
                        )
                        mapping_matrix[index][k][j] = (
                            weights[i][k][j][corner] / 4
                            if weights.shape[3] == 3
                            else weights[k][j][i][corner] / 4
                        )
        return super()._get_sparse_mapping_matrix(mapping_matrix)

    def _get_triangle(self, tri, hex_grid, simplex_index_NN, table_simplex):
        """
        :param tri: a Delaunay triangulation.
        :param hex_grid: a 2D numpy array (hexagonal grid).
        :param simplex_index_NN: a numpy array containing the indexes of the three neighboring simplexes.
        :param table_simplex: a numpy array containing the three indexes (hexaganol grid) of the target simplex.
        :return: two numpy array containing the three corner points and simplexes.
        """
        # This function is calculating the corner points (marked as 'X') and simplexes
        # for the nearest neighbor (NN) triangles. The function returns a bigger triangle,
        # which contains four Delaunay triangles.
        #
        #            X--------------------X
        #             \        /\        /
        #              \  NN  /  \  NN  /
        #               \    /    \    /
        #                \  /  .   \  /
        #                 \/________\/
        #                  \        /
        #                   \  NN  /
        #                    \    /
        #                     \  /
        #                      X
        #

        corner_points = []
        corner_simplexes = []
        for neighbors in np.arange(3):
            table_simplex_NN = tri.simplices[simplex_index_NN[neighbors]]
            simplex = np.array(list(set(table_simplex_NN) - set(table_simplex)))
            simplex = np.squeeze(simplex, axis=0)
            corner_simplexes.append(simplex)
            corner_points.append(hex_grid[simplex])
        corner_points = np.array(corner_points)
        corner_simplexes = np.array(corner_simplexes)
        corner_points = np.expand_dims(corner_points, axis=0)
        return corner_points, corner_simplexes
