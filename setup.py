from setuptools import setup
from setuptools import find_packages

setup(name='dl1_data_handler',
      version='0.7.2',
      description='dl1 HDF5 data writer + reader + processor',
      url='http://github.com/cta-observatory/dl1-data-handler',
      license='MIT',
      packages=['dl1_data_handler'],
      install_requires=[
          'numpy>=1.15.0',
          'tables>=3.4.4',
          'pyhessio @ https://api.github.com/repos/cta-observatory/pyhessio/tarball/v2.1.1',
          'ctapipe @ https://api.github.com/repos/cta-observatory/ctapipe/tarball/v0.6.2',
          'ctapipe-extra @ https://api.github.com/repos/cta-observatory/ctapipe-extra/tarball/v0.2.16',
          'pytest-cov'],
      dependency_links=[],
      zip_safe=True)
