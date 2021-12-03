from utils import Data, split_validation
from save_epoch import save_avg_and_std_lists
from model import *

import pickle
import numpy as np
import time


def train_model(model, train_data, test_data, cur_key, opt):
    best_hit = [0,0]
    best_mrr = [0,0]
    best_epoch = [0, 0]
    best_loss = [0,0]
    best_loss_list = [[0], [0]]
    
    bad_counter = 0
    
    epoch_loss_list = []
    inner_test_loss = []
    
    epoch_time_list = []
    epoch_count = 0
    for epoch in range(opt.epoch):
        epoch_time_start = time.time()
        hit, mrr, loss_list, total_loss, test_loss = train_test(model, train_data, test_data, cur_key)
        epoch_loss_list.append(np.mean(loss_list))
        inner_test_loss.append(np.mean(test_loss))
        
        flag = 0
        if hit >= best_hit[0]:
            best_hit[0] = hit
            best_epoch[0] = epoch
            best_mrr[0] = mrr
            best_loss[0] = total_loss
            best_loss_list[0] = loss_list
            flag = 1
        if mrr >= best_mrr[1]:
            best_mrr[1] = mrr
            best_hit[1] = hit
            best_epoch[1] = epoch
            best_loss[1] = total_loss
            best_loss_list[1] = loss_list
            flag = 1
        bad_counter += 1 - flag
        epoch_time_end = time.time()
        
        epoch_minute = (epoch_time_end - epoch_time_start) / 60
        epoch_time_list.append(epoch_minute)
        epoch_count += 1
        
        if bad_counter >= opt.patience:
            break
    
    return best_hit, best_mrr, best_epoch, best_loss, best_loss_list, epoch_loss_list, inner_test_loss, epoch_time_list, epoch_count

def load_dataset(opt, dataset):
    train_data = pickle.load(open('../datasets/' + dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + dataset + '/test.txt', 'rb'))
        valid_data = []
    
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    if dataset == 'diginetica':
        n_node = 43098
    elif 'yoochoose' in dataset:
        n_node = 37484
    else:
        n_node = 310
    
    return n_node, train_data, valid_data, test_data

def save_all_data(dataset, folder_name, cur_key, **kwargs):
    for name, lst in kwargs.items():    
        save_avg_and_std_lists(lst, cur_key.get_key(), dataset, folder_name, name)

