import pytest
from traitlets.config.loader import Config

from dl1_data_handler.reader import DLImageReader, DLWaveformReader

def test_dl1_image_reading(dl1_tmp_path, dl1_gamma_file):
    """check reading from pixel-wise image data files"""
    # Create a configuration suitable for the test
    config = Config(
        {
            "DLImageReader": {
                "allowed_tels": [4],
            },
        }
    )
    # Create an image reader and test basic properties
    dl1_reader = DLImageReader(input_url_signal=[dl1_gamma_file], config=config)
    assert dl1_reader._get_n_events() == 1
    assert dl1_reader.tel_type == "LST_LST_LSTCam"
    # Test the generation of a mono batch
    mono_batch = dl1_reader.generate_mono_batch([0])
    assert mono_batch["tel_id"] == 4
    assert mono_batch["features"].shape == (1,  110, 110, 2)

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
    assert r1_reader._get_n_events() == 1
    assert r1_reader.tel_type == "LST_LST_LSTCam"
    # Test the generation of a mono batch
    mono_batch = r1_reader.generate_mono_batch([0])
    assert mono_batch["tel_id"] == 4
    assert mono_batch["features"].shape == (1,  110, 110, 20)

def test_dl1_hillas_parameter_extraction(dl1_gamma_file):
    """Test DL1 reader extracts hillas parameters correctly and handles missing keys."""

    hillas_names_1 = [
    "obs_id", "event_id", "tel_id", "hillas_intensity", "hillas_skewness", "hillas_kurtosis",
    "hillas_fov_lon", "hillas_fov_lat", "hillas_r", "hillas_phi", "hillas_length",
    "hillas_length_uncertainty", "hillas_width", "hillas_width_uncertainty", "hillas_psi",
    "timing_intercept", "timing_deviation", "timing_slope",
    "leakage_pixels_width_1", "leakage_pixels_width_2",
    "leakage_intensity_width_1", "leakage_intensity_width_2",
    "concentration_cog", "concentration_core", "concentration_pixel",
    "morphology_n_pixels", "morphology_n_islands", "morphology_n_small_islands",
    "morphology_n_medium_islands", "morphology_n_large_islands",
    "intensity_max", "intensity_min", "intensity_mean", "intensity_std",
    "intensity_skewness", "intensity_kurtosis", "peak_time_max", "peak_time_min",
    "peak_time_mean", "peak_time_std", "peak_time_skewness", "peak_time_kurtosis", "core_psi"
    ]

    hillas_names_2 = [
        "obs_id", "event_id", "tel_id", "hillas_intensity",
        "hillas_skewness", "hillas_kurtosis", "NO_NAME"  # Include a non-existent name
    ]

    config = Config({"DLImageReader": {"allowed_tels": [4]}})
    reader = DLImageReader(input_url_signal=[dl1_gamma_file], config=config)

    batch = reader.generate_mono_batch([0])

    # Test with full set of valid names
    hillas = reader.get_parameters_dict(batch, hillas_names_1)
    present_count = sum(name in hillas for name in hillas_names_1)
    assert present_count == len(
        hillas_names_1
    ), f"Missing parameters: {set(hillas_names_1) - hillas.keys()}"

    # Test with one invalid parameter name included
    hillas_partial = reader.get_parameters_dict(batch, hillas_names_2)
    present_count_partial = sum(name in hillas_partial for name in hillas_names_2)
    assert present_count_partial < len(
        hillas_names_2
    ), "Unexpected match for invalid parameter"
    assert "NO_NAME" not in hillas_partial, "'NO_NAME' should not be in the result"
