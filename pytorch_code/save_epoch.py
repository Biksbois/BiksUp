import csv
from datetime import datetime
import os
from pickle import NONE
from biksLog import get_logger
import sys
import pandas as pd
# from biksLog import get_logger
from functools import reduce
from icecream import ic
import math
from aggregate import avg_and_std

from collections import defaultdict

from metadata import parse_keylist

log = get_logger()

from global_items import AVG_FOLDER, CSV_FOLDER, STD_FOLDER, NEW_CSV, KEY_COL_NAME

def get_foldername():
    now = datetime.now()
    return now.strftime("%d_%m_%Y-%H_%M_%S")

def try_create_folder(folder):
    try:
        os.makedirs(folder)
        log.info(f"Successfully created folder: {folder}")
    except Exception as e:
        log.exception(f"{e} - Unable to create folder")
        sys.exit()

def create_if_not_exists(folder):
    exists = os.path.exists(folder)
    
    if not exists:
        try_create_folder(folder)
    else:
        log.info(f"Path already exists: {folder}")

def create_csv_folder(folder_name):
    folder_path = os.path.join(CSV_FOLDER, folder_name)
    create_if_not_exists(CSV_FOLDER)
    create_if_not_exists(folder_path)

def save_df(col_name, key, val, save_path, csv_name):
    key_str = f"_{key}"
    save_path = os.path.join(save_path, csv_name + ".csv")
    if os.path.exists(save_path):
        df = pd.read_csv(save_path) # TODO: Read in another way
        if key_str in df[KEY_COL_NAME].tolist():
            key_index = df[KEY_COL_NAME].tolist().index(key_str)
            if not col_name in df.columns:
                df[col_name] = [-1.0] * len(df[KEY_COL_NAME])
            df[col_name][key_index] = val
        else:
            if not col_name in df.columns:
                df[col_name] = [-1.0] * len(df[KEY_COL_NAME])
            new_row = dict((k,-1) for k in df.columns)
            new_row[KEY_COL_NAME] = key_str
            new_row[col_name] = float(val)
            df = df.append(new_row, ignore_index=True)
    else:
        data = {
            KEY_COL_NAME:[key_str],
            col_name:[float(val)]
        }
        df = pd.DataFrame(data)
    
    df.to_csv(save_path, index=False) # TODO: save another way another way

def save_avg_and_std(avg, std, key, dataset, foldername, score_name):
    save_path = os.path.join(NEW_CSV, foldername)
    create_if_not_exists(save_path)
    save_df(dataset, key, avg, save_path, f"avg_{score_name}")
    save_df(dataset, key, std, save_path, f"std_{score_name}")

def save_avg_and_std_lists(lst, key, dataset, foldername, score_name):
    avg, std = avg_and_std(lst)
    save_avg_and_std(avg, std, key, dataset, foldername, score_name)
