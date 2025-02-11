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