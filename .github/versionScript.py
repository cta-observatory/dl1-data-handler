import sys
import pprint

pprint.pprint(sys.path)

 
# append the path of the
# parent directory
sys.path.append("..")

from dl1_data_handler import get_version_pypi

def get_version():
    return get_version_pypi()


if __name__ == "__main__":
    print(get_version())
