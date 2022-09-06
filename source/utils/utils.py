from .constants import *


def load_types():
    with open(TYPE_FILE_PATH) as type_file:
        types = [line.strip() for line in type_file.readlines()]
    return types
