from setuptools import setup
from setuptools import find_packages

setup(name='dl1_data_handler',
      version='0.7.0',
      description='dl1 HDF5 data writer + reader + processor',
      url='http://github.com/cta-observatory/dl1-data-handler',
      license='MIT',
      packages=['dl1_data_handler'],
      dependencies=[
         numpy>=1.15.0,
         tables>=3.4.4],
      dependency_links=['https://github.com/cta-observatory/ctapipe/tarball/v0.6.2#egg=ctapipe-0.6.2'],
      zip_safe=True)
