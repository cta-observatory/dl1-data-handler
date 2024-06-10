from dl1_data_handler.version import get_version  
def example():
    return get_version(pep440=False)


if __name__ == "__main__":
    print(example())
