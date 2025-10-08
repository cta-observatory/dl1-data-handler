import pytest
from traitlets.config.loader import Config
import numpy as np

from dl1_data_handler.reader import DLImageReader, DLWaveformReader


@pytest.fixture
def dl1_image_reader(dl1_gamma_file):
    config = Config({"DLImageReader": {"allowed_tels": [4]}})
    return DLImageReader(input_url_signal=[dl1_gamma_file], config=config)


def test_dl1_image_reading(dl1_image_reader):
    """check reading from pixel-wise image data files"""

    # Test basic properties
    assert dl1_image_reader._get_n_events() == 1  # nosec
    assert dl1_image_reader.tel_type == "LST_LST_LSTCam"  # nosec
    # Test the generation of a mono batch
    mono_batch = dl1_image_reader.generate_mono_batch([0])
    assert mono_batch["tel_id"] == 4  # nosec
    assert mono_batch["features"].shape == (1, 110, 110, 2)  # nosec
    # Check that the columns that are kept have no NaN values
    for col in dl1_image_reader.example_ids_keep_columns:
        assert not np.isnan(mono_batch[col][0])  # nosec
    # Check that the transformation is also present and has no NaN values
    assert mono_batch["log_true_energy"][0] == np.log10(
        mono_batch["true_energy"][0]
    )  # nosec
    assert mono_batch["impact_radius"][0] == np.sqrt(
        mono_batch["true_core_y"][0] ** 2 + mono_batch["true_core_x"][0] ** 2
    )  # nosec


def test_r1_waveform_reading(r1_tmp_path, r1_gamma_file):
    """check reading from pixel-wise waveform data files"""
    # Create a configuration suitable for the test
    config = Config(
        {
            "DLWaveformReader": {
                "allowed_tels": [4],
                "sequence_length": 20,
            },
        }
    )
    # Create an image reader and test basic properties
    r1_reader = DLWaveformReader(input_url_signal=[r1_gamma_file], config=config)
    assert r1_reader._get_n_events() == 1  # nosec
    assert r1_reader.tel_type == "LST_LST_LSTCam"  # nosec
    # Test the generation of a mono batch
    mono_batch = r1_reader.generate_mono_batch([0])
    assert mono_batch["tel_id"] == 4  # nosec
    assert mono_batch["features"].shape == (1, 110, 110, 20)  # nosec


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

    assert present_count == len(
        hillas_names_1
    ), f"Missing parameters: {set(hillas_names_1) - hillas.keys()}"  # nosec

    # Test with one invalid parameter name included
    hillas_partial = dl1_image_reader.get_parameters(batch, hillas_names_2)
    present_count_partial = sum(name in hillas_partial for name in hillas_names_2)

    assert present_count_partial < len(
        hillas_names_2
    ), "Unexpected match for invalid parameter"  # nosec

    assert (
        "NO_NAME" not in hillas_partial
    ), "'NO_NAME' should not be in the result"  # nosec

    hillas_all = dl1_image_reader.get_parameters(batch)

    assert list(hillas_all.keys()) == dl1_image_reader.dl1b_parameter_colnames  # nosec
