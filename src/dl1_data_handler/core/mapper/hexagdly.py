"""
This module defines the ``ImageMapper`` subclass ``HexagdlyMapper`` that hold the basic functionality for the hexagonal grid addressing method used by the ``hexagdly`` package.
"""

import numpy as np
from scipy.sparse import csr_matrix

from ctapipe.instrument.camera import PixelShape
from dl1_data_handler.core import ImageMapper

__all__ = ["HexagdlyMapper"]

class _HexGridTransform:
    """Map a 1-D hexagonal camera image to the 2-D grid ``keras_hexagdly``
    (and upstream HexagDLy) expect.

    Implementation detail of :class:`HexagdlyMapper`, not itself a mapper --
    not part of the module's public ``__all__``.

    ``keras_hexagdly`` works on a 2-D array where the hexagonal neighbour
    structure is encoded by an offset-column layout. This class finds the
    row/column of every camera pixel on that grid from its ``ctapipe``
    ``CameraGeometry``.

    The offset-column addressing needs one family of neighbour pairs to be
    horizontal. That alignment comes from ``ImageMapper.__init__``, which
    rotates the geometry by its ``pix_rotation`` before
    :class:`HexagdlyMapper` builds this transform, so ``geometry`` is
    expected to be aligned already. The resulting mapping is verified
    against the camera's own neighbour graph
    (:attr:`neighbor_mismatch_count`); zero mismatches means every pixel's
    grid neighbours are exactly its physical camera neighbours, so a
    geometry whose ``pix_rotation`` doesn't match its pixel layout fails
    loudly rather than mapping silently wrong.

    Only hexagonal-pixel cameras are supported; :class:`HexagdlyMapper`
    rejects other pixel shapes before this class is used.

    Parameters
    ----------
    geometry : ctapipe.instrument.CameraGeometry
        Camera geometry to map. Must have hexagonal pixels.

    Attributes
    ----------
    H, W : int
        Height and width of the 2-D grid the camera maps onto.
    row_idx, col_idx : numpy.ndarray
        Grid row/column of each camera pixel, in pixel-index order.
    neighbor_mismatch_count : int
        Number of camera pixels whose grid-neighbours don't exactly match
        their physical camera neighbours. Should be 0; a positive value
        means the addressing produced for this geometry is not trustworthy.
    """

    def __init__(self, geometry):
        x = geometry.pix_x.to_value("m").astype(np.float64)
        y = geometry.pix_y.to_value("m").astype(np.float64)

        vecs = self._neighbor_vectors(x, y, geometry.neighbors)
        dist = np.linalg.norm(vecs, axis=1)
        horiz = np.abs(vecs[:, 1]) < np.median(dist) * 1e-3
        if not np.any(horiz):
            raise ValueError(
                "no horizontal neighbour vectors -- _HexGridTransform expects a "
                "hexagonal-pixel geometry already aligned by its pix_rotation"
            )
        if not np.any(~horiz):
            raise ValueError(
                "no non-horizontal neighbour vectors -- _HexGridTransform expects a "
                "hexagonal-pixel geometry already aligned by its pix_rotation"
            )
        horizontal_pitch = np.median(np.abs(vecs[horiz, 0]))
        vertical_pitch = np.median(np.abs(vecs[~horiz, 1]))

        # Candidate search over grid origin (y0) and chirality (sign of the
        # axial q/r pair). Camera geometries are not guaranteed to share a
        # common pixel-index handedness -- e.g. DigiCam's raw pixel order is
        # point-inverted (both q and r negated) relative to LSTCam/MAGICCam/
        # NectarCam/FlashCam, which all happen to agree with each other.
        # Only (sq, sr) = (1, 1) and (-1, -1) are geometrically valid: negating
        # only one of q/r is a reflection (mirrors the lattice, changing which
        # physical pixels are diagonal vs vertical neighbours -- see the (1,-1)/
        # (-1,1) rows always producing a much larger, wrong H below) rather
        # than a rotation, and is never the fix for a pure handedness flip.
        best = None
        for y0 in (0.0, y.max(), y.min()):
            r0 = np.rint((y - y0) / vertical_pitch).astype(np.int64)
            q0 = np.rint((x - x.min()) / horizontal_pitch - r0 / 2).astype(np.int64)
            for sign in (1, -1):
                row, col = self._offset_from_axial(sign * q0, sign * r0)
                row, col = row - row.min(), col - col.min()
                mc = self._mismatch(geometry, row, col)
                if best is None or mc < best[0]:
                    best = (mc, row, col)
                if mc == 0:
                    break
            if best[0] == 0:
                break

        self.neighbor_mismatch_count, self.row_idx, self.col_idx = best
        self.H = int(self.row_idx.max()) + 1
        self.W = int(self.col_idx.max()) + 1
        if len(set(zip(self.row_idx, self.col_idx))) != geometry.n_pixels:
            raise RuntimeError(
                f"_HexGridTransform produced colliding grid cells for camera "
                f"'{geometry.name}' -- no candidate grid origin gave every pixel "
                "a unique cell. This geometry's hex addressing is not reliable."
            )

    @staticmethod
    def _neighbor_vectors(x, y, neighbors):
        v = []
        for i, nb in enumerate(neighbors):
            for j in nb:
                if i < j:
                    v.append((x[j] - x[i], y[j] - y[i]))
        return np.asarray(v)

    @staticmethod
    def _offset_from_axial(q, r):
        row = r + np.floor_divide(q - (q & 1), 2)
        return row.astype(np.int64), q.astype(np.int64)

    @staticmethod
    def _hex_neighbors_on_grid(grid, row, col):
        diag = (
            [(-1, -1), (-1, 1), (0, -1), (0, 1)]
            if col % 2 == 0
            else [(0, -1), (0, 1), (1, -1), (1, 1)]
        )
        out = []
        for dr, dc in [(-1, 0), (1, 0), *diag]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] >= 0:
                out.append(int(grid[nr, nc]))
        return sorted(out)

    @classmethod
    def _mismatch(cls, geometry, row, col):
        grid = np.full((int(row.max()) + 1, int(col.max()) + 1), -1, int)
        for i, (r, c) in enumerate(zip(row, col)):
            if grid[r, c] >= 0:
                return geometry.n_pixels  # collision -> reject this origin
            grid[r, c] = i
        m = 0
        for i, cam_nb in enumerate(geometry.neighbors):
            if cls._hex_neighbors_on_grid(grid, int(row[i]), int(col[i])) != sorted(
                map(int, cam_nb)
            ):
                m += 1
        return m


class HexagdlyMapper(ImageMapper):
    """Map a hexagonal camera image onto a ``keras_hexagdly``-addressed grid.

    ``HexagdlyMapper`` places each pixel value at its exact cell on the
    offset-column grid that ``keras_hexagdly.layers.Conv2d``/``MaxPool2d``
    address (see :class:`_HexGridTransform`), and verifies that every camera
    pixel's hexagonal neighbours land where those layers expect --
    construction fails if any neighbour relation is broken. No interpolation
    is involved: every output cell is either exactly one camera pixel's value
    or empty padding.

    The other mappers target a plain square convolution instead, either by
    resampling onto a square grid (``BilinearMapper``, ``BicubicMapper``,
    ``NearestNeighborMapper``, ``RebinMapper``, ``OversamplingMapper``) or by
    re-addressing pixels without changing their values (``AxialMapper``,
    ``ShiftingMapper``). In those layouts a square kernel's neighbourhood
    does not coincide with the camera's hexagonal one, which is what
    ``keras_hexagdly``'s parity-aware sub-kernels exist to handle.

    The mapping is built as a one-hot ``mapping_table`` (weight 1.0 at each
    pixel's grid cell), so the inherited ``ImageMapper.map_image`` is reused
    unmodified, following the same pattern as e.g. ``SquareMapper``. Because
    ``DLDataReader`` assumes a square ``(image_shape, image_shape, channels)``
    input shape, the grid is padded to a square of size
    ``max(_HexGridTransform.H, _HexGridTransform.W)``; the extra rows/columns
    are all-zero padding cells, same as the padding every other mapper
    already introduces at camera edges.

    Note the row axis follows ``_HexGridTransform``'s own convention (row 0
    at the bottom, matching ``origin="lower"`` display), which is flipped
    relative to the top-left-origin convention the other mappers produce via
    ``ImageMapper._get_sparse_mapping_matrix``. This has no effect on
    training (it's a fixed permutation applied consistently at map time and
    inference time); it only means a raw ``CameraDisplay``-style comparison
    against e.g. ``BilinearMapper`` output would show it upside down.
    """

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
        geometry : ctapipe.instrument.CameraGeometry
            Geometry of the camera to map. Must have hexagonal pixels.
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent : ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``parent``.
        """
        super().__init__(
            geometry=geometry,
            config=config,
            parent=parent,
            **kwargs,
        )

        if self.geometry.pix_type != PixelShape.HEXAGON:
            raise ValueError(
                f"HexagdlyMapper is only available for hexagonal pixel cameras. "
                f"Pixel type of the selected camera is '{self.geometry.pix_type}'."
            )

        # Build the hex-grid addressing for this camera geometry.
        grid = _HexGridTransform(self.geometry)
        if grid.neighbor_mismatch_count > 0:
            raise ValueError(
                f"_HexGridTransform produced {grid.neighbor_mismatch_count} "
                f"neighbour mismatch(es) for camera '{self.camera_type}' -- the "
                "resulting hex addressing is not reliable for this geometry."
            )
        self.grid_transform = grid

        # Pad the (H, W) grid to a square, since DLDataReader assumes a square
        # image_shape. row_idx/col_idx are already < H <= image_shape and
        # < W <= image_shape, so no adjustment to their values is needed.
        self.image_shape = max(grid.H, grid.W)

        # One-hot mapping table: each pixel maps to exactly one grid cell,
        # weight 1.0. This lets the inherited `ImageMapper.map_image` be
        # reused unmodified.
        flat_index = (
            grid.row_idx.astype(np.int64) * self.image_shape
            + grid.col_idx.astype(np.int64)
        )
        self.mapping_table = csr_matrix(
            (
                np.ones(self.n_pixels, dtype=np.float32),
                (np.arange(self.n_pixels), flat_index),
            ),
            shape=(self.n_pixels, self.image_shape * self.image_shape),
            dtype=np.float32,
        )
