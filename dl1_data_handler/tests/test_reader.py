import pytest
from traitlets.config.loader import Config

from dl1_data_handler.reader import DLImageReader

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