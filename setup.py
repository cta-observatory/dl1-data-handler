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
          'astropy',
          'numpy>=1.15.0',
          'scipy',
          'jupyter',
          'tables>=3.4.4',
          'pytest-cov',
          'pyirf',
          ],
      dependency_links=[],
      zip_safe=True)
