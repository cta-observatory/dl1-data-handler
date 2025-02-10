"""
common pytest fixtures for tests in dl1-data-handler.
Credits to ctapipe for the original code.
"""

import pytest

from ctapipe.core import run_tool
from ctapipe.utils import get_dataset_path
from ctapipe.utils.filelock import FileLock

@pytest.fixture(scope="session")
def prod5_gamma_simtel_path():
    return get_dataset_path("gamma_prod5.simtel.zst")

@pytest.fixture(scope="session")
def dl1_tmp_path(tmp_path_factory):
    """Temporary directory for global dl1 test data"""
    return tmp_path_factory.mktemp("dl1_")

@pytest.fixture(scope="session")
def dl1_gamma_file(dl1_tmp_path, prod5_gamma_simtel_path):
    """
    DL1 file containing both images and parameters from a gamma simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    output = dl1_tmp_path / "gamma.dl1.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_gamma_simtel_path}",
            f"--output={output}",
            "--write-images",
            "--DataWriter.Contact.name=αℓℓ the äüöß",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output