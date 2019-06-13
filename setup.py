from setuptools import setup
from setuptools import find_packages

setup(name='dl1_data_handler',
      version='0.7.5',
      description='dl1 HDF5 data writer + reader + processor',
      url='http://github.com/cta-observatory/dl1-data-handler',
      license='MIT',
      packages=['dl1_data_handler'],
      install_requires=[
          'numpy>=1.15.0',
          'tables>=3.4.4',
          'pyhessio @ https://api.github.com/repos/cta-observatory/pyhessio/tarball/v2.1.1',
          'ctapipe @ https://api.github.com/repos/cta-observatory/ctapipe/tarball/v0.6.2',
          'eventio>=0.16.1',
          'ctapipe-extra @ https://api.github.com/repos/cta-observatory/ctapipe-extra/tarball/v0.2.17',
          'pytest-cov',
          ],
      dependency_links=[],
      zip_safe=True)


setup(
    name='ctapipe_io_dl1dh',
    packages=['ctapipe_io_dl1dh'],
    version='0.1',
    description='ctapipe plugin for reading DL1DH files',
    long_description_content_type='text/markdown',
    install_requires=[
        'tables>=3.4.4',
        'ctapipe',
        'dl1_data_handler',
    ],
    tests_require=['pytest'],
    setup_requires=['pytest_runner'],
    license='MIT',
)