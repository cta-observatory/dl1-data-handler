from setuptools import setup
from setuptools import find_packages

setup(name='dl1_data_handler',
      version='0.7.0',
      description='dl1 HDF5 data writer + reader + processor',
      url='http://github.com/cta-observatory/dl1-data-handler',
      license='MIT',
      packages=['dl1_data_handler'],
      install_requires=[
          'numpy>=1.15.0', 
          'tables>=3.4.4',
          'ctapipe @ https://api.github.com/repos/cta-observatory/ctapipe/tarball/0.6.2',
          'ctapipe-extra @ https://api.github.com/repos/cta-observatory/ctapipe-extra/tarball/0.2.16'
          ],
      dependency_links=['git+https://github.com/cta-observatory/ctapipe.git#egg=ctapipe-0.6.2',
          'git+https://github.com/cta-observatory/ctapipe-extra.git#egg=ctapipe-extra-0.2.16'],
      zip_safe=True)
