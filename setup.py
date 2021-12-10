#!/usr/bin/env python
from setuptools import setup
from setuptools import find_packages

setup(name='dl1_data_handler',
      version='0.10.0',
      description='dl1 HDF5 data writer + reader + processor',
      url='http://github.com/cta-observatory/dl1-data-handler',
      license='MIT',
      packages=['dl1_data_handler'],
      install_requires=[
          'ctapipe==0.10.5',
          'pytest-cov',
          'pyirf',
          ],
      entry_points = {
        'console_scripts': ['dl1dh-generate_runlist=dl1_data_handler.generate_runlist:main',
                           'dl1dh-write_data=dl1_data_handler.write_data:main'],
      },
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