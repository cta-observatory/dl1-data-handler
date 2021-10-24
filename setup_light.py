#!/usr/bin/env python
from setuptools import setup
from setuptools import find_packages

setup(name='dl1_data_handler_light',
        version='0.9.0',
      description='light version of the dl1 HDF5 data writer + reader + processor',
      long_description='light install version of the dl1_data_handler. Visit http://github.com/cta-observatory/dl1-data-handler for a complete description',
      url='http://github.com/cta-observatory/dl1-data-handler',
      license='MIT',
      packages=['dl1_data_handler'],
      install_requires=[
          'astropy',
          'numpy>=1.15.0',
          'scipy',
          'jupyter',
          'tables>=3.4.4',
          'uproot==3.12.0',
          ],
      dependency_links=[],
      zip_safe=True)

