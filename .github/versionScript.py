import sys
from .. import dl1_data_handler 

def get_version():
    return get_version_pypi()


if __name__ == "__main__":
    print(get_version())
