"""Tests for image_mapper module."""
import pytest
import numpy as np
from ctapipe.instrument import CameraGeometry

from dl1_data_handler.image_mapper import (
    BilinearMapper,
    BicubicMapper,
    NearestNeighborMapper,
    RebinMapper,
    AxialMapper,
    OversamplingMapper,
    ShiftingMapper,
    SquareMapper,
)


@pytest.fixture
def lstcam_geometry():
    """Fixture to provide LSTCam geometry."""
    return CameraGeometry.from_name("LSTCam")


@pytest.fixture
def sample_image(lstcam_geometry):
    """Fixture to provide a sample image for testing."""
    return np.random.rand(lstcam_geometry.n_pixels, 1).astype(np.float32)


class TestInterpolationImageShape:
    """Test that interpolation_image_shape parameter works correctly (issue #171)."""

    @pytest.mark.parametrize(
        "mapper_class",
        [BilinearMapper, BicubicMapper, NearestNeighborMapper, RebinMapper],
    )
    def test_interpolation_image_shape_kwarg(self, lstcam_geometry, mapper_class):
        """Test that interpolation_image_shape can be set via kwarg.
        
        This is a regression test for issue #171 where passing
        interpolation_image_shape directly to mapper constructors
        was silently ignored.
        """
        # Request a custom interpolation grid size
        custom_size = 55
        mapper = mapper_class(
            geometry=lstcam_geometry, interpolation_image_shape=custom_size
        )

        # Verify the trait is set correctly
        assert (
            mapper.interpolation_image_shape == custom_size
        ), f"{mapper_class.__name__}: interpolation_image_shape trait not set correctly"

        # Verify the image_shape is updated
        assert (
            mapper.image_shape == custom_size
        ), f"{mapper_class.__name__}: image_shape not updated to custom size"

        # Verify the mapping table has the correct shape
        expected_mapping_cols = custom_size * custom_size
        assert (
            mapper.mapping_table.shape[1] == expected_mapping_cols
        ), f"{mapper_class.__name__}: mapping_table shape incorrect"

    @pytest.mark.parametrize(
        "mapper_class",
        [BilinearMapper, BicubicMapper, NearestNeighborMapper, RebinMapper],
    )
    def test_interpolation_image_shape_output(
        self, lstcam_geometry, sample_image, mapper_class
    ):
        """Test that the output image has the correct shape when interpolation_image_shape is set."""
        custom_size = 138
        mapper = mapper_class(
            geometry=lstcam_geometry, interpolation_image_shape=custom_size
        )

        # Map the image
        mapped_image = mapper.map_image(sample_image)

        # Verify output shape
        expected_shape = (custom_size, custom_size, 1)
        assert (
            mapped_image.shape == expected_shape
        ), f"{mapper_class.__name__}: output shape incorrect. Expected {expected_shape}, got {mapped_image.shape}"

    @pytest.mark.parametrize(
        "mapper_class",
        [BilinearMapper, BicubicMapper, NearestNeighborMapper, RebinMapper],
    )
    def test_default_image_shape(self, lstcam_geometry, mapper_class):
        """Test that mappers use default image_shape when interpolation_image_shape is not set."""
        mapper = mapper_class(geometry=lstcam_geometry)

        # Default for LSTCam should be 110
        default_size = 110
        assert (
            mapper.image_shape == default_size
        ), f"{mapper_class.__name__}: default image_shape incorrect"
        assert (
            mapper.interpolation_image_shape is None
        ), f"{mapper_class.__name__}: interpolation_image_shape should be None by default"


class TestMapperBasicFunctionality:
    """Test basic functionality of all mapper classes."""

    @pytest.mark.parametrize(
        "mapper_class",
        [
            BilinearMapper,
            BicubicMapper,
            NearestNeighborMapper,
            RebinMapper,
            AxialMapper,
            OversamplingMapper,
            ShiftingMapper,
        ],
    )
    def test_hexagonal_mapper_instantiation(self, lstcam_geometry, mapper_class):
        """Test that hexagonal mappers can be instantiated."""
        mapper = mapper_class(geometry=lstcam_geometry)
        assert mapper is not None
        assert mapper.mapping_table is not None

    def test_square_mapper_instantiation(self):
        """Test that SquareMapper can be instantiated with square pixel camera."""
        # SCTCam has square pixels
        square_geometry = CameraGeometry.from_name("SCTCam")
        mapper = SquareMapper(geometry=square_geometry)
        assert mapper is not None
        assert mapper.mapping_table is not None
        
        # Test output shape
        sample_square_image = np.random.rand(square_geometry.n_pixels, 1).astype(np.float32)
        mapped_image = mapper.map_image(sample_square_image)
        
        # Output should be square image with 1 channel
        assert len(mapped_image.shape) == 3
        assert mapped_image.shape[0] == mapped_image.shape[1]
        assert mapped_image.shape[2] == 1
        assert mapped_image.shape[0] == mapper.image_shape


    @pytest.mark.parametrize(
        "mapper_class",
        [
            BilinearMapper,
            BicubicMapper,
            NearestNeighborMapper,
            RebinMapper,
            AxialMapper,
            OversamplingMapper,
            ShiftingMapper,
        ],
    )
    def test_mapper_output_shape(self, lstcam_geometry, sample_image, mapper_class):
        """Test that mappers produce correctly shaped output."""
        mapper = mapper_class(geometry=lstcam_geometry)
        mapped_image = mapper.map_image(sample_image)

        # Output should be square image with 1 channel
        assert len(mapped_image.shape) == 3
        assert mapped_image.shape[0] == mapped_image.shape[1]
        assert mapped_image.shape[2] == 1

    @pytest.mark.parametrize(
        "mapper_class",
        [
            BilinearMapper,
            BicubicMapper,
            NearestNeighborMapper,
            RebinMapper,
            AxialMapper,
            OversamplingMapper,
            ShiftingMapper,
        ],
    )
    def test_mapper_multichannel(self, lstcam_geometry, mapper_class):
        """Test that mappers work with multi-channel input."""
        # Create a 2-channel image
        multichannel_image = np.random.rand(lstcam_geometry.n_pixels, 2).astype(
            np.float32
        )
        mapper = mapper_class(geometry=lstcam_geometry)
        mapped_image = mapper.map_image(multichannel_image)

        # Output should preserve the number of channels
        assert mapped_image.shape[2] == 2


class TestAxialMapperSpecific:
    """Test AxialMapper specific functionality."""

    def test_set_index_matrix_false(self, lstcam_geometry):
        """Test AxialMapper with set_index_matrix=False (default)."""
        mapper = AxialMapper(geometry=lstcam_geometry, set_index_matrix=False)
        assert mapper.index_matrix is None

    def test_set_index_matrix_true(self, lstcam_geometry):
        """Test AxialMapper with set_index_matrix=True."""
        mapper = AxialMapper(geometry=lstcam_geometry, set_index_matrix=True)
        assert mapper.index_matrix is not None
        # Index matrix should have the same shape as the output image
        assert mapper.index_matrix.shape == (mapper.image_shape, mapper.image_shape)
