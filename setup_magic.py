from setuptools import setup, find_packages
from os import path
from dl1_data_handler.version import *

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dl1_data_handler",
    version=get_version_pypi() + "_magic",
    author="DL1DH Team",
    author_email="d.nieto@ucm.es",
    description="dl1 HDF5 data writer + reader + processor",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="http://github.com/cta-observatory/dl1-data-handler",
    license="MIT",
    packages=["dl1_data_handler"],
    install_requires=[
        "numpy>1.16",
        "astropy>=4.0.5,<5",
        "ctapipe==0.9.1",
        "pandas",
        "pytest-cov",
        "tables",
    ],
    entry_points={
        "console_scripts": [
            "dl1dh-generate_runlist=dl1_data_handler.generate_runlist:main",
            "dl1dh-write_data=dl1_data_handler.write_data:main",
        ]
    },
    dependency_links=[],
    zip_safe=True,
)
