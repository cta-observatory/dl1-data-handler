"""
This module defines the ``ImageMapper`` subclass ``HexagonalPatchMapper`` that hold the basic functionality for the trigger patches of hexagonal pixel data from the AdvCam.
"""

import numpy as np
import tables
from importlib.resources import files

from ctapipe.instrument.camera import PixelShape
from ctapipe.core.traits import CaselessStrEnum
from dl1_data_handler.core import ImageMapper

__all__ = ["HexagonalPatchMapper"]

class HexagonalPatchMapper(ImageMapper):
    """
    HexagonalPatchMapper retrieves the information needed to work with unmapped
    hexagonal cameras: the pixel neighbor array for indexed convolutions and,
    for the Advanced SiPM camera (AdvCam), the trigger geometry (flowers, FEBs and
    the three trigger sectors).

    The trigger geometry is read from ``resources/triggergeometry_AdvCam_<version>.h5``.
    Each sector is made of complete flowers and complete FEBs, so the flower and FEB
    content of each sector is derived from the pixel mapping of the sector.
    """

    trigger_geometry_version = CaselessStrEnum(
        ["v1"],
        default_value="v1",
        help=(
            "Set the version of the trigger geometry file, only available for the Advanced SiPM camera (AdvCam). "
            "``v1``: 3 overlapping trigger sectors of 2989 pixels (427 flowers, 61 FEBs) for prod2 AdvCamSiPM."
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
                f"HexagonalPatchMapper is only available for hexagonal pixel cameras. Pixel type of the selected camera is '{geometry.pix_type}'."
            )

        # Sector information is only available for the AdvCam
        self.sector_mappings = None

        if geometry.name == "AdvCamSiPM":
            path = files("dl1_data_handler.resources").joinpath(
                f"triggergeometry_AdvCam_{self.trigger_geometry_version}.h5"
            )
            with tables.open_file(path, mode="r") as f:
                # Complete camera: pixels, flowers (1141) and FEBs (163)
                self.cam_neighbor_array = f.root.neighbors.camera_neighbors[:]
                self.fl_neighbor_array_l1 = f.root.neighbors.flower_neighbors_l1[:]
                self.supfl_neighbor_array_l1 = f.root.neighbors.superflower_neighbors_l1[:]
                self.fl_neighbor_array_tdscan = f.root.neighbors.flower_neighbors_tdscan[:]
                self.feb_indices = f.root.modules.indices[:]
                self.feb_neighbors = f.root.neighbors.feb_neighbors[:]
                # Sectors: boolean pixel masks, pixel order of each sector and pixel neighbors
                self.sectors_bool = f.root.sectors.mask[:]
                self.sector_mappings = f.root.sectors.mapping[:]
                self.sect0_neighbors = f.root.sectors.sect0_neighbors[:]

            # The superflowers on the camera border have less than 7 flowers (padded with -1)
            self.supfl_neighbor_l1_mask = self.supfl_neighbor_array_l1 >= 0

            # Flower and FEB order of each sector. The sector pixel mapping keeps the
            # pixels of each flower (7) and of each FEB (49) together, so the first pixel
            # of each group gives the flower or FEB index.
            n_sectors = self.sector_mappings.shape[0]
            self.sector_flower_order = self.sector_mappings.reshape(n_sectors, -1, 7)[:, :, 0] // 7
            self.sector_feb_order = self.sector_mappings.reshape(n_sectors, -1, 49)[:, :, 0] // 49

            # Neighbor arrays in the local order of a sector (valid for the three sectors).
            self.sect0_flower_neighbors = _get_sector_neighbors(
                self.fl_neighbor_array_tdscan, self.sector_flower_order[0]
            )
            self.sect0_feb_neighbors = _get_sector_neighbors(
                self.feb_neighbors, self.sector_feb_order[0]
            )
        # Retrieve the camera neighbor array to perform convolutions with cameras different from AdvCamSiPM.
        else:
            self.log.debug(f"Computing neighbor array for {geometry.name} ...")
            neighbor_matrix = geometry.neighbor_matrix
            num_pixels = neighbor_matrix.shape[0]
            self.cam_neighbor_array = np.full((num_pixels, 7), -1, dtype=int)
            for i in range(num_pixels):
                neighbors = [i] + np.where(neighbor_matrix[i])[0].tolist()
                self.cam_neighbor_array[i, : len(neighbors)] = neighbors

    def get_sector(self, raw_vector, sector_index, level="pixel"):
        """
        Select and reorder the elements of one sector.

        Parameters
        ----------
        raw_vector : numpy.ndarray
            Array whose first axis is pixels (7987), flowers (1141) or FEBs (163).
        sector_index : int
            Index of the sector.
        level : str
            ``pixel``, ``flower`` or ``feb``.

        Returns
        -------
        numpy.ndarray
            The elements of the sector in the standard sector order.
        """
        order = {
            "pixel": self.sector_mappings,
            "flower": self.sector_flower_order,
            "feb": self.sector_feb_order,
        }[level][sector_index]
        return raw_vector[order]

def _get_sector_neighbors(global_neighbors, order):
    """
    Translate a camera neighbor array (global indices, -1 padded) into the local
    indices of a sector. Neighbors outside the sector become -1 and every neighbor
    keeps its position in the row.
    """
    position = np.full(global_neighbors.shape[0], -1)
    position[order] = np.arange(len(order))
    rows = global_neighbors[order]
    return np.where(rows >= 0, position[np.clip(rows, 0, None)], -1)
