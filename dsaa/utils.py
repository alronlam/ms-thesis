import os


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
