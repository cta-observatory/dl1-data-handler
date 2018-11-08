from setuptools import setup
from setuptools import find_packages

setup(name='dl1-data-handler',
      version='0.6.0',
      description='dl1 HDF5 data writer + reader + processor',
      url='http://github.com/cta-observatory/dl1-data-handler',
      license='MIT',
      packages=['dl1-data-handler'],
      dependencies=[],
      dependency_links=['git+http://github.com/cta-observatory/ctapipe'],
      zip_safe=True)
