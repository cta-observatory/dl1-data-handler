"""
This module defines the base ``ImageMapper`` classes that hold the basic functionality for mapping raw 1D vectors into 2D mapped images.
"""

from abc import ABC, abstractmethod
from collections import namedtuple

import astropy.units as u
import numpy as np
from ctapipe.core import Component
from scipy import spatial
from scipy.sparse import csr_matrix

__all__ = ["ImageMapper"]


class ImageMapper(ABC, Component):
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
    cam_neighbor_array : numpy.ndarray or None
        Matrix used for indexing, initialized to None.

    Methods
    -------
    map_image(raw_vector)
        Transform the raw 1D vector data into the 2D mapped image.
    """

    # Constants for the ImageMapper classes
    Constants = namedtuple("Constants", ["decimal_precision", "tick_interval_limit"])
    constants = Constants(3, 0.002)

    @abstractmethod
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

        # Filter out non-traitlet kwargs before passing to Component
        # This allows compatibility with ctapipe's reader which may pass extra kwargs
        component_kwargs = {
            key: value
            for key, value in kwargs.items()
            if self.class_own_traits().get(key) is not None
        }

        super().__init__(
            config=config,
            parent=parent,
            **component_kwargs,
        )
        # Camera types
        self.geometry = geometry
        self.camera_type = self.geometry.name
        self.n_pixels = self.geometry.n_pixels
        # Rotate the pixel positions by the pixel to align
        if self.camera_type == "AdvCamSiPM":
            self.geometry.pix_rotation = 8.213 * u.deg
        self.geometry.rotate(self.geometry.pix_rotation)

        self.pix_x = np.around(
            self.geometry.pix_x.to_value("m"), decimals=self.constants.decimal_precision
        )
        self.pix_y = np.around(
            self.geometry.pix_y.to_value("m"), decimals=self.constants.decimal_precision
        )

        self.x_ticks = np.unique(self.pix_x).tolist()
        self.y_ticks = np.unique(self.pix_y).tolist()

        # Additional smooth the ticks for 'DigiCam', 'RealLSTCam', 'CHEC' and 'VERITAS' cameras
        if self.camera_type in ["DigiCam", "RealLSTCam"]:
            self.pix_y, self.y_ticks = self._smooth_ticks(self.pix_y, self.y_ticks)
        if self.camera_type in ["CHEC", "AdvCamSiPM"]:
            self.pix_x, self.x_ticks = self._smooth_ticks(self.pix_x, self.x_ticks)
            self.pix_y, self.y_ticks = self._smooth_ticks(self.pix_y, self.y_ticks)
        if self.camera_type == "VERITAS":
            self.pix_x, self.x_ticks = self._smooth_ticks(self.pix_x, self.x_ticks)

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
        self.cam_neighbor_array = None

    def map_image(self, raw_vectors: np.ndarray) -> np.ndarray:
        """
        Map raw pixel vectors to 2D image representations.

        This method supports both a single image and a batch of images. The input
        pixel values are projected onto the image plane using the precomputed
        mapping table. The operation is vectorized over images and channels and
        uses sparse matrix multiplication for efficient processing.

        Parameters
        ----------
        raw_vectors : numpy.ndarray
            Input pixel values.

            For a single image:
                Shape ``(n_pixels, n_channels)``

            For a batch of images:
                Shape ``(n_images, n_pixels, n_channels)``

            Each pixel value should be ordered according to the camera pixel
            indexing scheme used to generate the mapping table.

        Returns
        -------
        numpy.ndarray
            The mapped image(s).

            For a single input image:
                Shape ``(image_shape, image_shape, n_channels)``

            For a batch input:
                Shape ``(n_images, image_shape, image_shape, n_channels)``

        Notes
        -----
        The mapping is performed by multiplying the input pixel vectors with
        ``self.mapping_table``. For batched inputs, all images and channels are
        combined into a single matrix multiplication to improve performance.
        """
        single_image = raw_vectors.ndim == 2
        if single_image:
            raw_vectors = raw_vectors[None, ...]
        B, P, C = raw_vectors.shape
        # (B*C, P)
        x = raw_vectors.transpose(0, 2, 1).reshape(B * C, P)
        # sparse multiplication
        y = x @ self.mapping_table  # (B*C, H*W)
        # (B, C, H, W)
        y = y.reshape(B, C, self.image_shape, self.image_shape)
        # (B, H, W, C) s
        images = y.transpose(0, 2, 3, 1)
        return images[0] if single_image else images

    def _get_virtual_pixels(self, x_ticks, y_ticks, pix_x, pix_y):
        """Get the virtual pixels outside of the camera."""
        gridpoints = np.array(np.meshgrid(x_ticks, y_ticks)).T.reshape(-1, 2)
        gridpoints = [tuple(point) for point in gridpoints.tolist()]
        virtual_pixels = set(gridpoints) - set(zip(pix_x, pix_y))
        virtual_pixels = np.array(list(virtual_pixels))
        return virtual_pixels

    def _create_virtual_hex_pixels(
        self, first_ticks, second_ticks, first_pos, second_pos
    ):
        """Create virtual hexagonal pixels outside of the camera."""
        dist_first = np.around(
            abs(first_ticks[0] - first_ticks[1]),
            decimals=self.constants.decimal_precision,
        )
        dist_second = np.around(
            abs(second_ticks[0] - second_ticks[1]),
            decimals=self.constants.decimal_precision,
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
                    second_ticks[0] - dist_second,
                    decimals=self.constants.decimal_precision,
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
        if tick_diff % 2 != 0 and self.camera_type not in [
            "MAGICCam",
            "VERITAS",
            "AdvCamSiPM",
        ]:
            grid_second.insert(
                0,
                np.around(
                    grid_second[0] - dist_second,
                    decimals=self.constants.decimal_precision,
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
