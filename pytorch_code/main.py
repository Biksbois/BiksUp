#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: The BiksBois™ ©
"""

import argparse
import pickle
import time

from torch.nn.functional import fold
from metadata import get_data_dict
from utils import build_graph, Data, split_validation
from model import *
from metadata import metadataObj, get_metadata_list, data_dict
from pretty_printer import print_hyperparameters, introduce_biksup, check_if_valid, print_best_results
from parameter_class import parameterObj, get_parameters
from biksLog import get_logger
import sys
import os
from icecream import ic
from global_items import AVG_FOLDER, NEW_CSV
from save_epoch import get_foldername, save_avg_and_std_lists
from rich.progress import Progress
from aggregate import avg_and_std
from intro_file import create_introduce_file
import pandas as pd



log = get_logger()
parser = argparse.ArgumentParser()
parameters = get_parameters()
for p in parameters:
    p.add_new_argument(parser)
opt = parser.parse_args()

def main():
    check_if_valid(opt)
    parsed_keys = get_metadata_list(opt.keys, opt.runall, opt.runlast)
    key_str = introduce_biksup(parameters, parsed_keys, data_dict, opt, get_data_dict())
    folder_name = get_foldername()

    datasets = []
    if opt.exp_graph == '1':
        for fnames in os.listdir(os.getcwd()+'/datasets/'):
            if '1_' in fnames:
                datasets.append(fnames)
    else:
        datasets = opt.dataset

    for dataset in datasets:
        with Progress(auto_refresh=False) as progress:
            progress_list = []
            
            for k in parsed_keys:
                temp_process = progress.add_task(f"[cyan]Processing key {k.get_key()}", total=int(opt.iterations))
                progress_list.append(temp_process)
            

            for i in range(len(parsed_keys)):
                hit_list = []
                mrr_list = []
                time_list = []
                epoch_count_list = []
                epoch_time_list = []
                outer_loss_list = []
                outer_test_loss = []
                
                
                for iter in range(int(opt.iterations)):
                    cur_key = parsed_keys[i]
                
                    train_data = pickle.load(open('../datasets/' + dataset + '/train.txt', 'rb'))
                    if opt.validation:
                        train_data, valid_data = split_validation(train_data, opt.valid_portion)
                        test_data = valid_data
                    else:
                        test_data = pickle.load(open('../datasets/' + dataset + '/test.txt', 'rb'))
                    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
                    # g = build_graph(all_train_seq)
                    train_data = Data(train_data, shuffle=True)
                    test_data = Data(test_data, shuffle=False)
                    # del all_train_seq, g
                    if dataset == 'diginetica':
                        n_node = 43098
                    elif 'yoochoose' in dataset:
                        n_node = 37484
                    else:
                        n_node = 310

                    model = trans_to_cuda(SessionGraph(opt, n_node, cur_key))

                    
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
                    
                    start = time.time()
                    
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
                    
                    end = time.time()
                    
                    progress.update(progress_list[i], advance=1)
                    progress.refresh()
                    minutes = (end - start)/60
                    
                    avg_test_loss, std_test_loss = avg_and_std(inner_test_loss)
                    avg_loss, std_loss = avg_and_std(epoch_loss_list)
                    avg_val, std_val = avg_and_std(epoch_time_list)
                    
                    df1 = pd.DataFrame(inner_test_loss)
                    df2 = pd.DataFrame(epoch_loss_list)
                    df1.to_csv(f"test_{cur_key.get_key()}.csv")
                    df2.to_csv(f"other_{cur_key.get_key()}.csv")
                    
                    hit_list.append(best_hit[0])
                    mrr_list.append(best_mrr[1])
                    time_list.append(minutes)
                    epoch_count_list.append(epoch_count)
                    epoch_time_list.append(avg_val)
                    outer_loss_list.append(avg_loss)
                    outer_test_loss.append(avg_test_loss)

                    print_best_results(best_hit, best_mrr, best_epoch, cur_key.get_key(), iter)

                save_avg_and_std_lists(hit_list, cur_key.get_key(), dataset, folder_name, 'hit')
                save_avg_and_std_lists(mrr_list, cur_key.get_key(), dataset, folder_name, 'mrr')
                save_avg_and_std_lists(time_list, cur_key.get_key(), dataset, folder_name, 'totaltime')
                save_avg_and_std_lists(epoch_count_list, cur_key.get_key(), dataset, folder_name, 'epochcount')
                save_avg_and_std_lists(epoch_time_list, cur_key.get_key(), dataset, folder_name, 'epochtime')
                save_avg_and_std_lists(outer_loss_list, cur_key.get_key(), dataset, folder_name, 'epochloss')
                save_avg_and_std_lists(outer_test_loss, cur_key.get_key(), dataset, folder_name, 'epochtestloss')
    
    f = os.path.join(NEW_CSV, folder_name)
    for file in os.listdir(f):
        p = os.path.join(f, file)
        df = pd.read_csv(p)
        df = df.transpose()
        df.to_csv(p)
    create_introduce_file(opt.iterations, folder_name, parsed_keys, dataset, NEW_CSV, key_str)

if __name__ == '__main__':
    main()
