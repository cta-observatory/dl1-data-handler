import numpy as np
import pytest
from traitlets import TraitError
from traitlets.config.loader import Config

from dl1_data_handler.io import (
    DLImageReader,
    get_unmapped_image,
)


@pytest.fixture
def dl1_image_reader(dl1_gamma_file):
    config = Config({"DLImageReader": {"allowed_tels": [4]}})
    return DLImageReader(input_url_signal=[dl1_gamma_file], config=config)


def test_dl1_image_reading(dl1_image_reader):
    """check reading from pixel-wise image data files"""

    # Test basic properties
    assert dl1_image_reader.get_n_events() == 1  # nosec
    assert dl1_image_reader.tel_type == "LST_LST_LSTCam"  # nosec
    # Test the generation of a mono batch
    mono_batch = dl1_image_reader.generate_mono_batch([0])
    assert mono_batch["tel_id"] == 4  # nosec
    assert mono_batch["features"].shape == (1, 110, 110, 2)  # nosec
    # Check that the columns that are kept have no NaN values
    for col in dl1_image_reader.example_ids_keep_columns:
        assert not np.isnan(mono_batch[col][0])  # nosec
    # Check that the transformation is also present and has no NaN values
    assert mono_batch["log_true_energy"][0] == np.log10(mono_batch["true_energy"][0])  # nosec
    assert mono_batch["impact_radius"][0] == np.sqrt(
        mono_batch["true_core_y"][0] ** 2 + mono_batch["true_core_x"][0] ** 2
    )  # nosec


def test_dl1_hillas_parameter_extraction(dl1_image_reader):
    """Test DL1 reader extracts hillas parameters correctly and handles missing keys."""

    hillas_names_1 = dl1_image_reader.dl1b_parameter_colnames

    hillas_names_2 = [
        "obs_id",
        "event_id",
        "tel_id",
        "hillas_intensity",
        "hillas_skewness",
        "hillas_kurtosis",
        "NO_NAME",  # Include a non-existent name
    ]

    batch = dl1_image_reader.generate_mono_batch([0])

    # Test with full set of valid names
    hillas = dl1_image_reader.get_parameters(batch, hillas_names_1)
    present_count = sum(name in hillas for name in hillas_names_1)

    assert present_count == len(hillas_names_1), (
        f"Missing parameters: {set(hillas_names_1) - hillas.keys()}"
    )  # nosec

    # Test with one invalid parameter name included
    hillas_partial = dl1_image_reader.get_parameters(batch, hillas_names_2)
    present_count_partial = sum(name in hillas_partial for name in hillas_names_2)

    assert present_count_partial < len(hillas_names_2), (
        "Unexpected match for invalid parameter"
    )  # nosec

    assert "NO_NAME" not in hillas_partial, "'NO_NAME' should not be in the result"  # nosec

    hillas_all = dl1_image_reader.get_parameters(batch)

    assert list(hillas_all.keys()) == dl1_image_reader.dl1b_parameter_colnames  # nosec


def test_get_unmapped_image_log():
    """check reading log_image from unmapped image data"""
    dl1_event = {
        "image": np.array([10.0, 100.0, 0.0, -5.0], dtype=np.float32),
        "image_mask": np.array([1, 1, 0, 0], dtype=np.int32),
        "peak_time": np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
    }
    transforms = {
        "image_scale": 1.0,
        "image_offset": 0.0,
        "peak_time_scale": 1.0,
        "peak_time_offset": 0.0,
    }
    channels = ["image", "log_image"]

    unmapped = get_unmapped_image(dl1_event, channels, transforms)

    image = unmapped[:, 0]
    log_image = unmapped[:, 1]

    valid = image > 0
    np.testing.assert_allclose(log_image[valid], np.log10(image[valid]))
    # For invalid values, the original values should remain unchanged
    np.testing.assert_allclose(log_image[~valid], image[~valid])


def test_empty_input_url_signal():
    """Test that a TraitError is raised if input_url_signal is empty."""
    with pytest.raises(
        TraitError, match="The 'input_url_signal' list cannot be empty."
    ):
        DLImageReader(input_url_signal=[])


def test_stereo_signal_background_sharing_event_ids(dl1_gamma_file, tmp_path):
    """Signal and background events sharing (obs_id, event_id) stay separate in stereo mode."""
    import shutil

    background_file = tmp_path / "background.dl1.h5"
    shutil.copy(dl1_gamma_file, background_file)
    config = Config(
        {
            "DLImageReader": {
                "mode": "stereo",
                "min_telescopes": 1,
                "allowed_tels": [4],
            }
        }
    )
    signal_only = DLImageReader(input_url_signal=[dl1_gamma_file], config=config)
    reader = DLImageReader(
        input_url_signal=[dl1_gamma_file],
        input_url_background=[background_file],
        config=config,
    )
    # Every event of the copied file must be counted as its own example
    n_events = signal_only.get_n_events()
    assert reader.get_n_events() == 2 * n_events  # nosec
    assert reader.n_signal_events == n_events  # nosec
    assert reader.n_bkg_events == n_events  # nosec
    # Each stereo example must contain exactly one event (one file, one class)
    for idx in range(reader.get_n_events()):
        batch = reader.generate_stereo_batch([idx])
        assert len(np.unique(batch["file_index"])) == 1  # nosec
        assert len(np.unique(batch["true_shower_primary_class"])) == 1  # nosec
