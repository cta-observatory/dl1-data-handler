"""Tests for image_mapper module."""
import pytest
import numpy as np
from ctapipe.instrument import CameraGeometry

from dl1_data_handler.image_mapper import (
    BilinearMapper,
    BicubicMapper,
    HexagdlyMapper,
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
        
        Note: RebinMapper uses a small size (10) and increased max_memory_gb
        to avoid excessive memory requirements during testing.
        """
        # Request a custom interpolation grid size
        # Use smaller size for RebinMapper due to memory requirements
        custom_size = 10 if mapper_class == RebinMapper else 55
        
        # RebinMapper needs max_memory_gb set higher to allow the allocation
        kwargs = {"interpolation_image_shape": custom_size}
        if mapper_class == RebinMapper:
            kwargs["max_memory_gb"] = 100
        
        mapper = mapper_class(geometry=lstcam_geometry, **kwargs)

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
        """Test that the output image has the correct shape when interpolation_image_shape is set.
        
        Note: RebinMapper uses a small size (10) and increased max_memory_gb
        to avoid excessive memory requirements during testing.
        """
        # Use smaller size for RebinMapper due to memory requirements
        custom_size = 10 if mapper_class == RebinMapper else 138
        
        # RebinMapper needs max_memory_gb set higher to allow the allocation
        kwargs = {"interpolation_image_shape": custom_size}
        if mapper_class == RebinMapper:
            kwargs["max_memory_gb"] = 100
        
        mapper = mapper_class(geometry=lstcam_geometry, **kwargs)

        # Map the image
        mapped_image = mapper.map_image(sample_image)

        # Verify output shape
        expected_shape = (custom_size, custom_size, 1)
        assert (
            mapped_image.shape == expected_shape
        ), f"{mapper_class.__name__}: output shape incorrect. Expected {expected_shape}, got {mapped_image.shape}"

    @pytest.mark.parametrize(
        "mapper_class",
        [BilinearMapper, BicubicMapper, NearestNeighborMapper],
    )
    def test_default_image_shape(self, lstcam_geometry, mapper_class):
        """Test that mappers use default image_shape when interpolation_image_shape is not set.
        
        Note: RebinMapper is excluded from this test because its default size (110)
        exceeds the default memory limit (10 GB), requiring ~67 GB.
        """
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
            AxialMapper,
            HexagdlyMapper,
            OversamplingMapper,
            ShiftingMapper,
        ],
    )
    def test_hexagonal_mapper_instantiation(self, lstcam_geometry, mapper_class):
        """Test that hexagonal mappers can be instantiated.
        
        Note: RebinMapper is excluded from this test because its default size (110)
        exceeds the default memory limit (10 GB). See test_rebinmapper_small_size_works
        for RebinMapper instantiation test with appropriate parameters.
        """
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
            AxialMapper,
            HexagdlyMapper,
            OversamplingMapper,
            ShiftingMapper,
        ],
    )
    def test_mapper_output_shape(self, lstcam_geometry, sample_image, mapper_class):
        """Test that mappers produce correctly shaped output.
        
        Note: RebinMapper is excluded from this test because its default size (110)
        exceeds the default memory limit (10 GB). See test_rebinmapper_small_size_works
        for RebinMapper output shape test with appropriate parameters.
        """
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
            AxialMapper,
            HexagdlyMapper,
            OversamplingMapper,
            ShiftingMapper,
        ],
    )
    def test_mapper_multichannel(self, lstcam_geometry, mapper_class):
        """Test that mappers work with multi-channel input.
        
        Note: RebinMapper is excluded from this test because its default size (110)
        exceeds the default memory limit (10 GB). See test_rebinmapper_small_size_works
        for RebinMapper multichannel test with appropriate parameters.
        """
        # Create a 2-channel image
        multichannel_image = np.random.rand(lstcam_geometry.n_pixels, 2).astype(
            np.float32
        )
        mapper = mapper_class(geometry=lstcam_geometry)
        mapped_image = mapper.map_image(multichannel_image)

        # Output should preserve the number of channels
        assert mapped_image.shape[2] == 2

class TestMapperBatchFunctionality:
    """Test batched image mapping functionality."""

    @pytest.mark.parametrize(
        "mapper_class",
        [
            BilinearMapper,
            BicubicMapper,
            NearestNeighborMapper,
            AxialMapper,
            HexagdlyMapper,
            OversamplingMapper,
            ShiftingMapper,
        ],
    )
    def test_mapper_batch_images(self, lstcam_geometry, mapper_class):
        """Test that mappers support mapping multiple images at once.

        The batch interface should produce the same result as mapping each
        image individually, while returning an additional batch dimension.

        Note: RebinMapper is excluded because its default configuration requires
        special memory handling.
        """
        rng = np.random.default_rng(42)

        n_images = 10
        n_channels = 2

        mapper = mapper_class(geometry=lstcam_geometry)

        # Create batch of images
        batch_images = rng.random(
            (
                n_images,
                lstcam_geometry.n_pixels,
                n_channels,
            ),
            dtype=np.float32,
        )

        # Map batch
        mapped_batch = mapper.map_image(batch_images)

        # Check batch output shape
        expected_shape = (
            n_images,
            mapper.image_shape,
            mapper.image_shape,
            n_channels,
        )

        assert (
            mapped_batch.shape == expected_shape
        ), (
            f"{mapper_class.__name__}: batch output shape incorrect. "
            f"Expected {expected_shape}, got {mapped_batch.shape}"
        )

        # Compare with individual mapping
        mapped_individual = np.stack(
            [
                mapper.map_image(batch_images[i])
                for i in range(n_images)
            ],
            axis=0,
        )

        np.testing.assert_allclose(
            mapped_batch,
            mapped_individual,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{mapper_class.__name__}: batch mapping differs from individual mapping",
        )


class TestRebinMapperMemoryValidation:
    """Test RebinMapper memory validation and functionality."""

    def test_rebinmapper_default_size_exceeds_limit(self, lstcam_geometry):
        """Test that RebinMapper default size exceeds the default memory limit.
        
        RebinMapper's default behavior requires ~67 GB for LSTCam, which exceeds
        the 10 GB default safety limit. This is expected behavior.
        """
        # Default size (110) should raise ValueError due to memory requirements
        with pytest.raises(ValueError, match="would require approximately.*GB of memory"):
            RebinMapper(geometry=lstcam_geometry)

    def test_rebinmapper_large_size_raises_error(self, lstcam_geometry):
        """Test that RebinMapper raises ValueError for large interpolation_image_shape."""
        # Large size should raise ValueError with even more memory requirements
        with pytest.raises(ValueError, match="would require approximately.*GB of memory"):
            RebinMapper(geometry=lstcam_geometry, interpolation_image_shape=200)

    def test_rebinmapper_error_message_helpful(self, lstcam_geometry):
        """Test that RebinMapper error message suggests alternatives."""
        try:
            RebinMapper(geometry=lstcam_geometry, interpolation_image_shape=200)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            # Check that error message contains helpful information
            assert "BilinearMapper" in error_msg or "BicubicMapper" in error_msg
            assert "memory-efficient" in error_msg
            assert "interpolation_image_shape" in error_msg or "image_shape" in error_msg
            assert "GB of memory" in error_msg

    def test_rebinmapper_disable_memory_check(self, lstcam_geometry):
        """Test that RebinMapper memory check can be disabled with max_memory_gb=None."""
        # Small size that would normally pass, but we're testing the None behavior
        # Note: We still use a small size to avoid actually allocating huge memory
        mapper = RebinMapper(
            geometry=lstcam_geometry,
            interpolation_image_shape=10,
            max_memory_gb=None
        )
        assert mapper is not None
        assert mapper.mapping_table is not None

    def test_rebinmapper_custom_memory_limit(self, lstcam_geometry):
        """Test that RebinMapper respects custom max_memory_gb values."""
        # Size that requires ~0.13 GB should pass with 1 GB limit
        mapper = RebinMapper(
            geometry=lstcam_geometry,
            interpolation_image_shape=10,
            max_memory_gb=1
        )
        assert mapper is not None
        
        # Size that requires ~0.13 GB should fail with 0.01 GB limit
        with pytest.raises(ValueError, match="would require approximately.*GB of memory"):
            RebinMapper(
                geometry=lstcam_geometry,
                interpolation_image_shape=10,
                max_memory_gb=0.01
            )

    def test_rebinmapper_small_size_works(self, lstcam_geometry, sample_image):
        """Test that RebinMapper works with small interpolation_image_shape and increased limit."""
        # Small size with increased memory limit should work
        mapper = RebinMapper(
            geometry=lstcam_geometry,
            interpolation_image_shape=10,
            max_memory_gb=100
        )
        assert mapper is not None
        assert mapper.mapping_table is not None
        
        # Test that it can actually map an image
        mapped_image = mapper.map_image(sample_image)
        
        # Output should be square image with 1 channel
        assert len(mapped_image.shape) == 3
        assert mapped_image.shape[0] == mapped_image.shape[1]
        assert mapped_image.shape[2] == 1
        assert mapped_image.shape[0] == 10


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


class TestHexagdlyMapperSpecific:
    """Test HexagdlyMapper specific functionality.

    Unlike the interpolation-based mappers, HexagdlyMapper places each pixel
    at an *exact* grid cell (no interpolation), verified against the
    camera's own neighbour graph. These tests cover that verification and
    the exact placement, on top of the generic contract tests above.
    """

    @pytest.mark.parametrize(
        "camera_name",
        ["LSTCam", "MAGICCam", "NectarCam", "FlashCam", "DigiCam", "VERITAS"],
    )
    def test_zero_neighbor_mismatches(self, camera_name):
        """The hex-grid addressing must exactly reproduce each camera's
        physical neighbour graph -- zero mismatches, not an approximation.

        DigiCam specifically regression-tests the chirality search in
        _HexGridTransform: its raw pixel index order is point-inverted
        (both axial q and r negated) relative to LSTCam, MAGICCam, NectarCam
        and FlashCam, which all happen to share one handedness. Without
        searching over both, DigiCam mismatched on every single pixel
        (1296/1296).

        VERITAS covers a geometry stored in mm rather than m.
        """
        geometry = CameraGeometry.from_name(camera_name)
        mapper = HexagdlyMapper(geometry=geometry)
        assert mapper.grid_transform.neighbor_mismatch_count == 0

    def test_square_pixel_camera_rejected(self):
        """HexagdlyMapper only supports hexagonal-pixel cameras."""
        square_geometry = CameraGeometry.from_name("SCTCam")
        with pytest.raises(ValueError, match="hexagonal pixel cameras"):
            HexagdlyMapper(geometry=square_geometry)

    def test_image_shape_is_square_padded(self, lstcam_geometry):
        """DLDataReader assumes a square image_shape; the (possibly
        non-square) hex grid must be padded to image_shape = max(H, W)."""
        mapper = HexagdlyMapper(geometry=lstcam_geometry)
        grid = mapper.grid_transform
        assert mapper.image_shape == max(grid.H, grid.W)

    def test_chirality_search_is_generic_not_camera_specific(self, lstcam_geometry):
        """A mirror image of a camera that already works (LSTCam) must map
        with zero mismatches too, so the addressing can't depend on one
        camera's handedness or be keyed off its name.

        This alone doesn't force the sign flip in _HexGridTransform's
        chirality search -- the mirrored lattice is matched by the first
        candidate. That path is exercised by DigiCam in
        test_zero_neighbor_mismatches, whose pixel layout needs the flipped
        sign.
        """
        # A mirror image is rotated the opposite way, so pix_rotation has to be
        # mirrored along with the pixels for the geometry to stay consistent --
        # ImageMapper aligns the lattice from pix_rotation before mapping.
        mirrored = CameraGeometry(
            name="LSTCam_mirrored_for_test",
            pix_id=lstcam_geometry.pix_id,
            pix_x=-lstcam_geometry.pix_x,
            pix_y=lstcam_geometry.pix_y,
            pix_area=lstcam_geometry.pix_area,
            pix_type=lstcam_geometry.pix_type,
            pix_rotation=-lstcam_geometry.pix_rotation,
        )
        # Mirroring must not change the neighbour topology itself -- only the
        # handedness -- otherwise this wouldn't isolate the chirality issue.
        assert all(
            set(map(int, mirrored.neighbors[i])) == set(map(int, lstcam_geometry.neighbors[i]))
            for i in range(lstcam_geometry.n_pixels)
        )
        mapper = HexagdlyMapper(geometry=mirrored)
        assert mapper.grid_transform.neighbor_mismatch_count == 0

    def test_exact_pixel_placement_roundtrip(self, lstcam_geometry):
        """Every real camera pixel's value must land at exactly its
        row_idx/col_idx grid cell, with no interpolation/blending."""
        mapper = HexagdlyMapper(geometry=lstcam_geometry)
        grid = mapper.grid_transform

        image = np.arange(1, lstcam_geometry.n_pixels + 1, dtype=np.float32).reshape(
            -1, 1
        )
        mapped = mapper.map_image(image)

        for pixel_idx in range(lstcam_geometry.n_pixels):
            row, col = int(grid.row_idx[pixel_idx]), int(grid.col_idx[pixel_idx])
            assert mapped[row, col, 0] == pixel_idx + 1

        # Every non-pixel cell is empty padding.
        n_nonzero = np.count_nonzero(mapped[..., 0])
        assert n_nonzero == lstcam_geometry.n_pixels

    def test_origin_and_chirality_search_runs_once_not_per_image(
        self, lstcam_geometry, monkeypatch
    ):
        """The grid-origin/chirality candidate search must run only once, at
        mapper construction time -- never per map_image() call. If it ran per
        image, mapping a large dataset would pay the search cost on every
        single event instead of once per camera type.
        """
        from dl1_data_handler import image_mapper as image_mapper_module

        call_count = {"n": 0}
        original_mismatch = image_mapper_module._HexGridTransform._mismatch

        def counting_mismatch(cls, geometry, row, col):
            call_count["n"] += 1
            return original_mismatch(geometry, row, col)

        monkeypatch.setattr(
            image_mapper_module._HexGridTransform,
            "_mismatch",
            classmethod(counting_mismatch),
        )

        mapper = HexagdlyMapper(geometry=lstcam_geometry)
        n_calls_after_construction = call_count["n"]
        assert n_calls_after_construction > 0  # the search did run at construction

        rng = np.random.default_rng(0)
        for _ in range(20):
            image = rng.random((lstcam_geometry.n_pixels, 1)).astype(np.float32)
            mapper.map_image(image)

        assert call_count["n"] == n_calls_after_construction
