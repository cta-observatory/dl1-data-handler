import sys

from dl1-data-handler.dl1-data-handler.dl1_data_handler import version
def get_version():
    return get_version_pypi()


if __name__ == "__main__":
    print(get_version())
