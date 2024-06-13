import sys
import os

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))
print(parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Use a function from my_module
from dl1_data_handler.version import get_version_pypi

def get_version():
    return get_version_pypi()


if __name__ == "__main__":
    print(get_version())
