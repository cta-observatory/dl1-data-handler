import sys
sys.path.append('//home/runner/work/dl1-data-handler/dl1-data-handler')

#from ..dl1_data_handler.dl1_data_handler import get_version_pypi
import version
def get_version():
    return get_version_pypi()


if __name__ == "__main__":
    print(get_version())
