from traitlets.config.loader import Config

from dl1_data_handler.io import DLWaveformReader


def test_r1_waveform_reading(r1_gamma_file):
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
    assert r1_reader.get_n_events() == 1  # nosec
    assert r1_reader.tel_type == "LST_LST_LSTCam"  # nosec
    # Test the generation of a mono batch
    mono_batch = r1_reader.generate_mono_batch([0])
    assert mono_batch["tel_id"] == 4  # nosec
    assert mono_batch["features"].shape == (1, 110, 110, 20)  # nosec
