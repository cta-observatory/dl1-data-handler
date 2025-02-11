"""
This module defines the ``ImageMapper`` classes that hold the basic functionality for mapping raw 1D vectors into 2D mapped images.
"""

import numpy as np
from scipy import spatial
from scipy.sparse import csr_matrix
from collections import Counter, namedtuple

from ctapipe.instrument.camera import PixelShape
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import Bool, Int

__all__ = [
    "ImageMapper",
    "AxialMapper",
    "BicubicMapper",
    "BilinearMapper",
    "NearestNeighborMapper",
    "OversamplingMapper",
    "RebinMapper",
    "ShiftingMapper",
    "SquareMapper",
]

class ImageMapper(TelescopeComponent):
    """
    Base component for mapping raw 1D vectors into 2D mapped images.

    This class handles the transformation of raw telescope image or waveform data
    into a format suitable for further analysis. It supports various telescope
    types and applies necessary scaling and offset adjustments to the image data.


    Attributes
    ----------
    geometry : ctapipe.instrument.CameraGeometry
        The geometry of the camera, including pixel positions and camera type.
    camera_type : str
        The type of the camera, derived from the geometry.
    image_shape : int
        The shape of the 2D image, based on the camera type.
    n_pixels : int
        The number of pixels in the camera.
    pix_x : numpy.ndarray
        The x-coordinates of the pixels rounded to three decimal places.
    pix_y : numpy.ndarray
        The y-coordinates of the pixels rounded to three decimal places.
    x_ticks : list
        Unique x-coordinates of the pixels.
    y_ticks : list
        Unique y-coordinates of the pixels.
    internal_pad : int
        Padding used to ensure that the camera pixels aren't affected at the edges.
    rebinning_mult_factor : int
        Multiplication factor used for rebinning.
    index_matrix : numpy.ndarray or None
        Matrix used for indexing, initialized to None.

    Methods
    -------
    map_image(raw_vector)
        Transform the raw 1D vector data into the 2D mapped image.
    """

    # Constants for the ImageMapper classes
    Constants = namedtuple("Constants", ["decimal_precision", "tick_interval_limit"])
    constants = Constants(3, 0.002) 

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

        # Camera types
        self.geometry = geometry
        self.camera_type = self.geometry.name
        self.n_pixels = self.geometry.n_pixels
        # Rotate the pixel positions by the pixel to align
        self.geometry.rotate(self.geometry.pix_rotation)

        self.pix_x = np.around(
            self.geometry.pix_x.value, decimals=self.constants.decimal_precision
        )
        self.pix_y = np.around(
            self.geometry.pix_y.value, decimals=self.constants.decimal_precision
        )

        self.x_ticks = np.unique(self.pix_x).tolist()
        self.y_ticks = np.unique(self.pix_y).tolist()

        # Additional smooth the ticks for 'DigiCam', 'RealLSTCam' and 'CHEC' cameras
        if self.camera_type in ["DigiCam", "RealLSTCam"]:
            self.pix_y, self.y_ticks = self._smooth_ticks(self.pix_y, self.y_ticks)
        if self.camera_type == "CHEC":
            self.pix_x, self.x_ticks = self._smooth_ticks(self.pix_x, self.x_ticks)
            self.pix_y, self.y_ticks = self._smooth_ticks(self.pix_y, self.y_ticks)

        # At the edges of the cameras some mapping methods run into issues.
        # Therefore, we are using a default padding to ensure that the camera pixels aren't affected.
        # The default padding is removed after the conversion is finished.
        self.internal_pad = 0
        # Retrieve default shape of the image from the oversampling method.
        _, output_grid = self._get_grids_for_oversampling()    
        # This value can be overwritten by the subclass
        self.image_shape = int(len(output_grid) ** 0.5)

        # Only needed for rebinnig
        self.rebinning_mult_factor = 1

        # Set the indexed matrix to None
        self.index_matrix = None

    def map_image(self, raw_vector: np.array) -> np.array:
        """
        Map the raw pixel data to a 2D image.

        Parameters
        ----------
        raw_vector : numpy.ndarray
            A numpy array of values for each pixel, in order of pixel index.

        Returns
        -------
        numpy.ndarray
            A numpy array of shape [img_width, img_length, N_channels].
        """
        # Reshape each channel and stack the result
        images = np.concatenate(
            [
                (raw_vector[:, channel].T @ self.mapping_table).reshape(
                    self.image_shape, self.image_shape, 1
                )
                for channel in range(raw_vector.shape[1])
            ],
            axis=-1,
        )
        return images

    def _get_virtual_pixels(self, x_ticks, y_ticks, pix_x, pix_y):
        """Get the virtual pixels outside of the camera."""
        gridpoints = np.array(np.meshgrid(x_ticks, y_ticks)).T.reshape(-1, 2)
        gridpoints = [tuple(l) for l in gridpoints.tolist()]
        virtual_pixels = set(gridpoints) - set(zip(pix_x, pix_y))
        virtual_pixels = np.array(list(virtual_pixels))
        return virtual_pixels

    def _create_virtual_hex_pixels(
        self, first_ticks, second_ticks, first_pos, second_pos
    ):
        """Create virtual hexagonal pixels outside of the camera."""
        dist_first = np.around(
            abs(first_ticks[0] - first_ticks[1]), decimals=self.constants.decimal_precision
        )
        dist_second = np.around(
            abs(second_ticks[0] - second_ticks[1]), decimals=self.constants.decimal_precision
        )

        tick_diff = len(first_ticks) * 2 - len(second_ticks)
        tick_diff_each_side = tick_diff // 2
        # Extend second_ticks
        for _ in range(tick_diff_each_side + self.internal_pad * 2):
            second_ticks = (
                [
                    np.around(
                        second_ticks[0] - dist_second,
                        decimals=self.constants.decimal_precision,
                    )
                ]
                + second_ticks
                + [
                    np.around(
                        second_ticks[-1] + dist_second,
                        decimals=self.constants.decimal_precision,
                    )
                ]
            )
        # Extend first_ticks
        for _ in range(self.internal_pad):
            first_ticks = (
                [
                    np.around(
                        first_ticks[0] - dist_first,
                        decimals=self.constants.decimal_precision,
                    )
                ]
                + first_ticks
                + [
                    np.around(
                        first_ticks[-1] + dist_first,
                        decimals=self.constants.decimal_precision,
                    )
                ]
            )
        # Adjust for odd tick_diff
        if tick_diff % 2 != 0:
            second_ticks.insert(
                0,
                np.around(
                    second_ticks[0] - dist_second, decimals=self.constants.decimal_precision
                ),
            )

        # Create the virtual pixels outside of the camera
        virtual_pixels = []
        for i in np.arange(2):
            vp1 = self._get_virtual_pixels(
                first_ticks[i::2], second_ticks[0::2], first_pos, second_pos
            )
            vp2 = self._get_virtual_pixels(
                first_ticks[i::2], second_ticks[1::2], first_pos, second_pos
            )
            (
                virtual_pixels.append(vp1)
                if vp1.shape[0] < vp2.shape[0]
                else virtual_pixels.append(vp2)
            )
        virtual_pixels = np.concatenate(virtual_pixels)
        first_pos = np.concatenate((first_pos, virtual_pixels[:, 0]))
        second_pos = np.concatenate((second_pos, virtual_pixels[:, 1]))

        return first_pos, second_pos, dist_first, dist_second

    def _generate_nearestneighbor_table(self, input_grid, output_grid, pixel_weight):
        """Generate a nearest neighbor table for mapping."""
        # Finding the nearest point in the hexagonal input grid
        # for each point in the square utü grid
        tree = spatial.cKDTree(input_grid)
        nn_index = np.reshape(
            tree.query(output_grid)[1], (self.internal_shape, self.internal_shape)
        )

        mapping_matrix = np.zeros(
            (input_grid.shape[0], self.internal_shape, self.internal_shape),
            dtype=np.float32,
        )
        for y_grid in np.arange(self.internal_shape):
            for x_grid in np.arange(self.internal_shape):
                mapping_matrix[nn_index[y_grid][x_grid]][y_grid][x_grid] = pixel_weight
        return self._get_sparse_mapping_matrix(mapping_matrix)

    def _get_sparse_mapping_matrix(self, mapping_matrix, normalize=False):
        """Get a sparse mapping matrix from the given mapping matrix."""
        # Cutting the mapping table after n_pixels, since the virtual pixels have intensity zero.
        mapping_matrix = mapping_matrix[: self.n_pixels]
        # Normalization (approximation) of the mapping table
        if normalize:
            norm_factor = np.sum(mapping_matrix) / float(self.n_pixels)
            mapping_matrix /= norm_factor
        # Slice the mapping table to the correct shape
        mapping_matrix = mapping_matrix[
            :,
            self.internal_pad : self.internal_shape - self.internal_pad,
            self.internal_pad : self.internal_shape - self.internal_pad,
        ]
        # Applying a flip to all mapping tables so that the image indexing starts from the top left corner
        mapping_matrix = np.flip(mapping_matrix, axis=1)
        # Reshape and convert to sparse matrix
        sparse_mapping_matrix = csr_matrix(
            mapping_matrix.reshape(
                mapping_matrix.shape[0], self.image_shape * self.image_shape
            ),
            dtype=np.float32,
        )
        return sparse_mapping_matrix

    def _get_weights(self, points, target):
        """
        Calculate barycentric weights for multiple triangles and target points.

        Parameters
        ----------
        points : numpy.ndarray
            A numpy array of shape (i, 3, 2) for three points (one triangle).
        target : numpy.ndarray
            A numpy array of shape (i, 2) for one target 2D point.

        Returns
        -------
        numpy.ndarray
            A numpy array of shape (i, 3) containing the three weights.
        """
        x1, y1 = points[:, 0, 0], points[:, 0, 1]
        x2, y2 = points[:, 1, 0], points[:, 1, 1]
        x3, y3 = points[:, 2, 0], points[:, 2, 1]
        xt, yt = target[:, 0], target[:, 1]

        divisor = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        w1 = ((y2 - y3) * (xt - x3) + (x3 - x2) * (yt - y3)) / divisor
        w2 = ((y3 - y1) * (xt - x3) + (x1 - x3) * (yt - y3)) / divisor
        w3 = 1 - w1 - w2

        weights = np.stack((w1, w2, w3), axis=-1)
        return weights.astype(np.float32)


    def _get_grids_for_oversampling(
        self,
    ):
        """Get the grids for oversampling."""

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
        ) = self._create_virtual_hex_pixels(
            first_ticks, second_ticks, first_pos, second_pos
        )

        # Create the output grid
        grid_first = []
        for i in first_ticks:
            grid_first.append(i - dist_first / 4.0)
            grid_first.append(i + dist_first / 4.0)
        grid_second = [second_ticks[0] - dist_second / 2.0]
        for j in second_ticks:
            grid_second.append(j + dist_second / 2.0)

        tick_diff = (len(grid_first) - len(grid_second)) // 2
        # Extend second_ticks
        for _ in range(tick_diff):
            grid_second = (
                [
                    np.around(
                        grid_second[0] - dist_second,
                        decimals=self.constants.decimal_precision,
                    )
                ]
                + grid_second
                + [
                    np.around(
                        grid_second[-1] + dist_second,
                        decimals=self.constants.decimal_precision,
                    )
                ]
            )
        # Adjust for odd tick_diff
        # TODO: Check why MAGICCam, VERITAS, and UNKNOWN-7987PX (AdvCam) do not need this adjustment
        if tick_diff % 2 != 0 and self.camera_type not in ["MAGICCam", "VERITAS", "UNKNOWN-7987PX"]:
            grid_second.insert(
                0,
                np.around(
                    grid_second[0] - dist_second, decimals=self.constants.decimal_precision
                ),
            )

        if len(self.x_ticks) < len(self.y_ticks):
            input_grid = np.column_stack([first_pos, second_pos])
            x_grid, y_grid = np.meshgrid(grid_first, grid_second)
        else:
            input_grid = np.column_stack([second_pos, first_pos])
            x_grid, y_grid = np.meshgrid(grid_second, grid_first)
        output_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])

        return input_grid, output_grid

    def _get_grids_for_interpolation(
        self,
    ):
        """Get the grids for interpolation."""

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
        ) = self._create_virtual_hex_pixels(
            first_ticks, second_ticks, first_pos, second_pos
        )
        # Create the input grid
        input_grid = (
            np.column_stack([first_pos, second_pos])
            if len(self.x_ticks) < len(self.y_ticks)
            else np.column_stack([second_pos, first_pos])
        )
        # Create the square grid
        grid_first = np.linspace(
            np.min(first_pos),
            np.max(first_pos),
            num=self.internal_shape * self.rebinning_mult_factor,
            endpoint=True,
        )
        grid_second = np.linspace(
            np.min(second_pos),
            np.max(second_pos),
            num=self.internal_shape * self.rebinning_mult_factor,
            endpoint=True,
        )
        if len(self.x_ticks) < len(self.y_ticks):
            x_grid, y_grid = np.meshgrid(grid_first, grid_second)
        else:
            x_grid, y_grid = np.meshgrid(grid_second, grid_first)
        output_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        return input_grid, output_grid

    def _smooth_ticks(self, pix_pos, ticks):
        """Smooth the ticks needed for the 'DigiCam' and 'CHEC' cameras."""
        remove_val, change_val = [], []
        for i in range(len(ticks) - 1):
            if abs(ticks[i] - ticks[i + 1]) <= self.constants.tick_interval_limit:
                remove_val.append(ticks[i])
                change_val.append(ticks[i + 1])

        ticks = [tick for tick in ticks if tick not in remove_val]
        pix_pos = [
            change_val[remove_val.index(pos)] if pos in remove_val else pos
            for pos in pix_pos
        ]
        return pix_pos, ticks


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
            abs(first_ticks[0] - first_ticks[1]), decimals=self.constants.decimal_precision
        )
        dist_second = np.around(
            abs(second_ticks[0] - second_ticks[1]), decimals=self.constants.decimal_precision
        )

        # manipulate y ticks with extra ticks
        num_extra_ticks = len(self.y_ticks)
        for i in np.arange(num_extra_ticks):
            second_ticks.append(
                np.around(
                    second_ticks[-1] + dist_second, decimals=self.constants.decimal_precision
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
                    second_ticks[-1] + dist_second, decimals=self.constants.decimal_precision
                )
            )
            second_ticks.insert(
                0,
                np.around(
                    second_ticks[0] - dist_second, decimals=self.constants.decimal_precision
                ),
            )
        # If tick_diff is odd, add one more tick to the beginning
        if tick_diff % 2 != 0:
            second_ticks.insert(
                0,
                np.around(
                    second_ticks[0] - dist_second, decimals=self.constants.decimal_precision
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
                    for l in range(weights.shape[3]):
                        index = (
                            corner_indexes[i][k][j][l]
                            if weights.shape[3] == 3
                            else corner_indexes[k][j][i][l]
                        )
                        mapping_matrix[index][k][j] = (
                            weights[i][k][j][l] / 4
                            if weights.shape[3] == 3
                            else weights[k][j][i][l] / 4
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
