import csv
from datetime import datetime
from genericpath import exists
import os
from biksLog import get_logger
import sys
import pandas as pd
from biksLog import get_logger
from functools import reduce

log = get_logger()

from global_items import AVG_FOLDER, CSV_FOLDER

log = get_logger()

def get_foldername():
    now = datetime.now()
    return now.strftime("%d_%m_%Y-%H_%M_%S")

def create_csv_folder(folder_name):
    folder_path = os.path.join(CSV_FOLDER, folder_name)
    exists = os.path.exists(folder_name)

    try:
        if not exists:
            os.makedirs(folder_path)
            log.info(f"Successfully created folder: {folder_path}")
    except Exception as e:
        log.exception(f"{e} - Unable to create folder")
        sys.exit()

def save_epoch(folder_name, csv_name, duration, epoch, mrr, hit, loss, loss_evolution):
    header = ['duration', 'epoch', 'mrr', 'hit', 'loss', 'loss_evolution']
    
    zipped_data = zip(duration, epoch, mrr, hit, loss, loss_evolution)
    data = [z for z in zipped_data]
    
    if not csv_name.endswith('.csv'):
        csv_name += '.csv'
    
    create_csv_folder(folder_name)
    csv_path = os.path.join(CSV_FOLDER ,folder_name, csv_name)
    
    try:
        with open(csv_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write multiple rows
            writer.writerows(data)
        log.info(f"Successfully wrote the file {csv_path}")
    except Exception as e:
        log.exception(f"Unable to write file. {e}")
        sys.exit()


def get_one_file(root_path, folder_name, i, p):
    csv_path = os.path.join(root_path, str(i), folder_name, str(p) + '.csv')
    try:
        temp_df = pd.read_csv(csv_path)
        temp_df.drop('loss_evolution', axis=1, inplace=True)
    except Exception as e:
        log.exception(f"Unable to open file {csv_path} - {e}")
        sys.exit()
    return temp_df

def save_avg_csv(pd_sum, avg_path, p):
    if not os.path.exists(avg_path):
        os.makedirs(avg_path)
    avg_path = os.path.join(avg_path, p + '.csv')
    try:
        pd_sum.to_csv(avg_path, index=False)
    except Exception as e:
        log.exception(f"Unable to save file {avg_path} - {e}")

def save_average(iter, folder_name, parsed_keys, root_path=CSV_FOLDER, save_root=AVG_FOLDER):
    for p in parsed_keys:
        pd_list = []
        for i in range(int(iter)):
            pd_list.append(get_one_file(root_path, folder_name, i, p.get_key()))
        
        pd_sum = reduce(lambda a, b: a.add(b, fill_value=0), pd_list)
        pd_sum = pd_sum / len(pd_list)
        
        save_path = os.path.join(save_root, folder_name)
        save_avg_csv(pd_sum, save_path, p.get_key())

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

if __name__ == '__main__':
    data = {'mrr':[1,2,3],
            'hit':[4,5,6]
    }
    
    df_one = pd.DataFrame(data)
    df_two = pd.DataFrame(data)

    # print((df_one + df_two) / 2)
    print(reduce(lambda a, b: a.add(b, fill_value=0), [df_one, df_two]))
    
    