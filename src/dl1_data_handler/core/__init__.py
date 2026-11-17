"""Convenience module exposing all ``ImageMapper`` classes."""

from .mapper.image import ImageMapper
from .mapper.axial import AxialMapper
from .mapper.bicubic import BicubicMapper
from .mapper.bilinear import BilinearMapper
from .mapper.hexagdly import HexagdlyMapper
from .mapper.hexagonalpatch import HexagonalPatchMapper
from .mapper.nearestneighbor import NearestNeighborMapper
from .mapper.oversampling import OversamplingMapper
from .mapper.rebin import RebinMapper
from .mapper.shifting import ShiftingMapper
from .mapper.square import SquareMapper

__all__ = [
    "ImageMapper",
    "AxialMapper",
    "BicubicMapper",
    "BilinearMapper",
    "HexagdlyMapper",
    "HexagonalPatchMapper",
    "NearestNeighborMapper",
    "OversamplingMapper",
    "RebinMapper",
    "ShiftingMapper",
    "SquareMapper",
]
