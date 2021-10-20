import csv
from datetime import datetime
from genericpath import exists
import os
from biksLog import get_logger
import sys

from global_items import CSV_FOLDER

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
