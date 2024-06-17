from setuptools import setup, find_packages
from os import path

def getVersionFromFile():
    file = open(".github/versionBackup.txt").readlines()
    for line in file:
        for word in line.split():
            return word

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()
    
setup(
    name="dl1_data_handler",
    version=getVersionFromFile(),
    author="DL1DH Team",
    author_email="d.nieto@ucm.es",
    description="dl1 HDF5 data reader + processor",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="http://github.com/cta-observatory/dl1-data-handler",
    license="MIT",
    packages=["dl1_data_handler"],
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.11",
        "astropy",
        "ctapipe==0.21.1",
        "traitlets>=5.0",
        "jupyter",
        "pandas",
        "pytest-cov",
        "tables>=3.8",
    ],
    dependency_links=[],
    zip_safe=True,
)
