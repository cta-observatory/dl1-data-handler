from dl1_data_handler.dl1_data_handler import version

def get_version():
    return version.get_version_pypi()


if __name__ == "__main__":
    print(get_version())
