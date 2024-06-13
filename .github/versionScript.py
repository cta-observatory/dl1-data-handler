import sys
import os

#sys.path.append("..")

#from ..dl1_data_handler.dl1_data_handler import get_version_pypi
import dl1_data_handler
def get_version():
    return get_version_pypi()


if __name__ == "__main__":
    print(get_version())
