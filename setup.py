from setuptools import setup
from setuptools import find_packages

setup(name='image_extractor',
        version='0.2',
        description='simtel to HDF5 data dumper',
        url='http://github.com/',
        author='Bryan Kim',
        author_email='bsk2133@columbia.edu',
        license='MIT',
        packages=['image_extractor'],
        dependencies = [],
        dependency_links = ['git+http://github.com/cta-observatory/ctapipe'],
        zip_safe=False)
