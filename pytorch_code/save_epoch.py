import csv
from datetime import datetime
import os
from pickle import NONE
from biksLog import get_logger
import sys
import pandas as pd
from biksLog import get_logger
from functools import reduce
from icecream import ic
import math

log = get_logger()

from global_items import AVG_FOLDER, CSV_FOLDER, STD_FOLDER

log = get_logger()

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

def generate_csv_name(key, iter):
    return 'k_' + key + '_i_' + str(iter) + '.csv'

def save_epoch(folder_name, csv_name, duration, epoch, mrr, hit, loss, loss_evolution, iter):
    header = ['duration', 'epoch', 'mrr', 'hit', 'loss', 'loss_evolution']
    
    zipped_data = zip(duration, epoch, mrr, hit, loss, loss_evolution)
    data = [z for z in zipped_data]
    
    csv_name = generate_csv_name(csv_name, iter)
    
    create_csv_folder(folder_name)
    
    csv_path = os.path.join(CSV_FOLDER, folder_name, csv_name)
    
    try:
        with open(csv_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
        log.info(f"Successfully wrote the file {csv_path}")
    except Exception as e:
        log.exception(f"Unable to write file. {e}")
        sys.exit()


def get_one_file(root_path, folder_name, i, p, iter):
    csv_path = os.path.join(root_path, str(i), folder_name, generate_csv_name(p, iter))
    try:
        temp_df = pd.read_csv(csv_path)
        temp_df.drop('loss_evolution', axis=1, inplace=True)
    except Exception as e:
        log.exception(f"Unable to open file {csv_path} - {e}")
        sys.exit()
    return temp_df

def save_avg_csv(pd_sum, avg_path, p, iter):
    # iterations = str(iter + 1)
    if not os.path.exists(avg_path):
        os.makedirs(avg_path)
    avg_path = os.path.join(avg_path, generate_csv_name(p, iter))
    try:
        pd_sum.to_csv(avg_path, index=False)
    except Exception as e:
        log.exception(f"Unable to save file {avg_path} - {e}")

def save_std_csv(pd_sum, std_path, key, iter):
    if not os.path.exists(std_path):
        os.makedirs(std_path)
    std_path = os.path.join(std_path, generate_csv_name(key, iter))
    try:
        pd_sum.to_csv(std_path, index=False)
    except Exception as e:
        log.exception(f"Unable to save file {std_path} - {e}")

def calc_std(pd_list, pd_sum):
    result_list = []
    n = len(pd_list[-1])
    
    for i in range(len(pd_list[-1])):
        result_list.append([p[i] for p in pd_list])
    
    for i in range(len(result_list)):
        result_list[i] = [(r - pd_sum[i])**2 for r in result_list[i]]
        result_list[i] = [sum(result_list[i])]
        result_list[i] = [math.sqrt(r / n) for r in result_list[i]][0]
    
    return result_list

def save_std(pd_sum, save_path, key, iter, pd_list):
    std_pd = pd_sum.copy()
    
    columns = ['mrr', 'hit', 'loss']
    
    for col in columns:
        std_pd[col] = calc_std([c[col] for c in pd_list], pd_sum[col])
    
    save_std_csv(std_pd, save_path, key, iter)

def save_average(iter, folder_name, parsed_keys, root_path=CSV_FOLDER, save_root=AVG_FOLDER, second_root=STD_FOLDER, dataset=NONE):
    for p in parsed_keys:
        pd_list = []
        for i in range(int(iter)):
            pd_list.append(get_one_file(root_path, folder_name, i, p.get_key(), iter))
        
        if len(pd_list)==1:
            pd_sum = pd_list[0]
        elif len(pd_list) == 0:
            log.exception(f"When saving for key {p.get_key()}, no files are extracted.")
            break
        else:
            pd_sum = reduce(lambda a, b: a.add(b, fill_value=0), pd_list)
            pd_sum = pd_sum / len(pd_list)
        
        save_path = os.path.join(save_root, folder_name)
        save_avg_csv(pd_sum, save_path, p.get_key(), iter)
        if second_root == None:
            save_std(pd_sum, save_path, p.get_key(), iter, pd_list)
        else:
            save_path = os.path.join(second_root, folder_name)
            save_std(pd_sum, save_path, p.get_key(), iter, pd_list)

def combine_files(iterations, folder_name, parsed_keys, dataset, root_name):
    path = os.path.join(root_name, folder_name)
    
    for file in os.listdir(path):
        pass


if __name__ == '__main__':
    folder_name = get_foldername()
    csv_name = '111'
    
    duration = [-1, -1]
    epoch = [1,1]
    mrr = [0,0]
    hit = [2,2]
    loss = [3,3]
    loss_evolution = [[8,8,8,8], [9,9,9,9]]
    
    save_epoch(folder_name, csv_name, duration, epoch, mrr, hit, loss, loss_evolution)


