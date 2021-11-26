import os
import pandas as pd
import sys
from biksLog import get_logger

log = get_logger()

def parse_keys(keys):
    key_str = ""
    
    for key in keys:
        key_str += f"\n    - {key}"
    
    return key_str

def parse_input_key(keys):
    key_str = ""
    
    for key in keys:
        key_str += f" {key}"
    
    return key_str

def create_file(path, iterations, key_str, dataset, keys):
    file_content = ("The experiment just ran with the following parameters:\n"
                    f"  - Iterations: {iterations}\n"
                    f"  - Keys: {parse_keys(keys)}\n"
                    f"  - Dataset: {dataset}\n\n"
                    "The keys are as follows:\n"
                    f"{key_str}\n\n"
                    "If you wish to run this experiment again, run the following string:\n"
                    f"  - python main.py --iterations {iterations} --keys {parse_input_key(keys)} --dataset {dataset}")
    
    save_path = os.path.join(path, "info.txt")
    
    f = open(save_path, "w")
    f.write(file_content)
    f.close()

def create_introduce_file(iterations, folder_name, parsed_keys, dataset, root_name, key_str):
    path = os.path.join(root_name, folder_name)
    dataframes = []
    keys = []
    
    keys = [k.get_key() for k in parsed_keys]
    
    create_file(path, iterations, key_str, dataset, keys)
