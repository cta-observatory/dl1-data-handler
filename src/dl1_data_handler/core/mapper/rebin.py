"""
This module defines the ``ImageMapper`` subclass ``RebinMapper`` that hold the basic functionality for mapping raw 1D vectors into 2D mapped images using the rebinning method.
"""

from collections import Counter

import numpy as np
from ctapipe.core.traits import Float, Int
from ctapipe.instrument.camera import PixelShape
from scipy import spatial

from dl1_data_handler.core import ImageMapper

__all__ = ["RebinMapper"]


class RebinMapper(ImageMapper):
    """
    RebinMapper maps images from a hexagonal pixel grid to
    a square pixel grid using a rebinning technique.

    This class extends the functionality of ImageMapper by implementing
    methods to generate a rebinning mapping table and perform the transformation.
    Hexagonal pixels are rebinned or reshaped to square pixels, which preserve
    the intensity of the pixels. It is particularly useful for
    applications where a simple and efficient rebinning of pixel data is required.
    """

    interpolation_image_shape = Int(
        default_value=None,
        allow_none=True,
        help=(
            "Integer to overwrite the default shape of the resulting mapped image."
            "Only available for interpolation and rebinning methods."
        ),
    ).tag(config=True)

    max_memory_gb = Float(
        default_value=10,
        allow_none=True,
        help=(
            "Maximum memory in GB that RebinMapper is allowed to allocate. "
            "Set to None to disable memory checks. Default is 10 GB. "
            "Note: RebinMapper uses approximately (image_shape * 10)^2 * image_shape^2 * 4 bytes."
        ),
    ).tag(config=True)

    def __init__(
        self,
        geometry,
        config=None,
        parent=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent : ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``config``
        """

        super().__init__(
            geometry=geometry,
            config=config,
            parent=parent,
            **kwargs,
        )

        if geometry.pix_type != PixelShape.HEXAGON:
            raise ValueError(
                f"RebinMapper is only available for hexagonal pixel cameras. Pixel type of the selected camera is '{geometry.pix_type}'."
            )

        # At the edges of the cameras the mapping methods run into issues.
        # Therefore, we are using a default padding to ensure that the camera pixels aren't affected.
        # The default padding is removed after the conversion is finished.
        self.internal_pad = 3
        if self.interpolation_image_shape is not None:
            self.image_shape = self.interpolation_image_shape
        self.internal_shape = self.image_shape + self.internal_pad * 2
        self.rebinning_mult_factor = 10

        # Validate memory requirements before proceeding (if max_memory_gb is set)
        if self.max_memory_gb is not None:
            # RebinMapper uses a fine grid (internal_shape * rebinning_mult_factor)^2
            # and creates a mapping matrix of shape (fine_grid_size, internal_shape, internal_shape)
            fine_grid_size = (self.internal_shape * self.rebinning_mult_factor) ** 2
            estimated_memory_gb = (
                fine_grid_size * self.internal_shape * self.internal_shape * 4
            ) / (1024**3)  # 4 bytes per float32

            if estimated_memory_gb > self.max_memory_gb:
                raise ValueError(
                    f"RebinMapper with image_shape={self.image_shape} would require "
                    f"approximately {estimated_memory_gb:.1f} GB of memory, which exceeds "
                    f"the limit of {self.max_memory_gb:.1f} GB. "
                    f"To allow this allocation, set max_memory_gb to a higher value or None. "
                    f"Alternatively, consider using a smaller interpolation_image_shape (recommended < 60) "
                    f"or use BilinearMapper or BicubicMapper instead, which are more memory-efficient."
                )

        # Creating the hexagonal and the output grid for the conversion methods.
        input_grid, output_grid = super()._get_grids_for_interpolation()
        # Calculate the mapping table
        self.mapping_table = self._generate_table(input_grid, output_grid)

    def _generate_table(self, input_grid, output_grid):
        # Finding the nearest point in the hexagonal grid for each point in the square grid
        tree = spatial.cKDTree(input_grid)
        nn_index = np.reshape(
            tree.query(output_grid)[1],
            (
                self.internal_shape * self.rebinning_mult_factor,
                self.internal_shape * self.rebinning_mult_factor,
            ),
        )
        # Calculating the overlapping area/weights for each square pixel
        mapping_matrix = np.zeros(
            (output_grid.shape[0], self.internal_shape, self.internal_shape),
            dtype=np.float32,
        )
        # Create a grid of indices
        y_indices, x_indices = np.meshgrid(
            np.arange(
                0,
                self.internal_shape * self.rebinning_mult_factor,
                self.rebinning_mult_factor,
            ),
            np.arange(
                0,
                self.internal_shape * self.rebinning_mult_factor,
                self.rebinning_mult_factor,
            ),
            indexing="ij",
        )
        # Flatten the grid indices
        y_indices = y_indices.flatten()
        x_indices = x_indices.flatten()
        # Iterate over the flattened grid indices
        for y_grid, x_grid in zip(y_indices, x_indices):
            counter = Counter(
                nn_index[
                    y_grid : y_grid + self.rebinning_mult_factor,
                    x_grid : x_grid + self.rebinning_mult_factor,
                ].flatten()
            )
            pixel_index = np.array(list(counter.keys()))
            weights = np.array(list(counter.values())) / np.sum(list(counter.values()))
            y_idx = int(y_grid / self.rebinning_mult_factor)
            x_idx = int(x_grid / self.rebinning_mult_factor)
            mapping_matrix[pixel_index, y_idx, x_idx] = weights
        return super()._get_sparse_mapping_matrix(mapping_matrix)
