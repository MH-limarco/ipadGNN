from os import path
DEFAULT_DIR = "row_data"

def parse_setting(map, name):
    setting = map[name]
    return setting["func"], setting["transform"]

def parse_data_dir(func_name, name):
    return path.join(DEFAULT_DIR, func_name, name)