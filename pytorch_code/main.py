#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
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
from global_items import AVG_FOLDER
from save_epoch import get_foldername, save_epoch, save_average, combine_files
from rich.progress import Progress
import statistics


log = get_logger()
parser = argparse.ArgumentParser()
parameters = get_parameters()
for p in parameters:
    p.add_new_argument(parser)
opt = parser.parse_args()

def average(lst):
    return sum(lst) / len(lst)

def std(lst):
    return statistics.stdev(lst)

def avg_and_std(lst):
    return average(lst), std(lst)

def main():
    check_if_valid(opt)
    parsed_keys = get_metadata_list(opt.keys, opt.runall, opt.runlast)
    key_str = introduce_biksup(parameters, parsed_keys, data_dict, opt, get_data_dict())
    folder_name = get_foldername()
    
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
            
            
            for iter in range(int(opt.iterations)):
                cur_key = parsed_keys[i]
            
                train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
                if opt.validation:
                    train_data, valid_data = split_validation(train_data, opt.valid_portion)
                    test_data = valid_data
                else:
                    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
                # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
                # g = build_graph(all_train_seq)
                train_data = Data(train_data, shuffle=True)
                test_data = Data(test_data, shuffle=False)
                # del all_train_seq, g
                if opt.dataset == 'diginetica':
                    n_node = 43098
                elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
                    n_node = 37484
                else:
                    n_node = 310

                model = trans_to_cuda(SessionGraph(opt, n_node, cur_key))

                start = time.time()
                
                best_hit = [0,0]
                best_mrr = [0,0]
                best_epoch = [0, 0]
                best_loss = [0,0]
                best_loss_list = [[0], [0]]
                
                bad_counter = 0
                
                epoch_time_list = []
                epoch_count = 0
                
                for epoch in range(opt.epoch):
                    epoch_time_start = time.time()
                    hit, mrr, loss_list, total_loss = train_test(model, train_data, test_data, cur_key)
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
                
                progress.update(progress_list[i], advance=1)
                progress.refresh()
                end = time.time()
                minutes = (end - start)/60
                
                avg_val, std_val = avg_and_std(epoch_time_list)
                
                hit_list.append(best_hit[0])
                mrr_list.append(best_mrr[1])
                time_list.append(minutes)
                epoch_count_list.append(epoch_count)
                epoch_time_list.append(avg_val)
                
                row_name = ""
                col_name = ""
                
                print_best_results(best_hit, best_mrr, best_epoch, cur_key.get_key(), iter)
                
                temp_folder_name = os.path.join(str(iter), folder_name)
                save_epoch(temp_folder_name, cur_key.get_key(), [minutes, minutes], best_epoch, best_mrr, best_hit, best_loss, best_loss_list, opt.iterations)
        save_average(opt.iterations, folder_name, parsed_keys)
        combine_files(opt.iterations, folder_name, parsed_keys, opt.dataset, AVG_FOLDER, key_str)
        
        hit_avg, hit_std = avg_and_std(hit_list)
        mrr_avg, mrr_std = avg_and_std(mrr_list)
        time_avg, time_std = avg_and_std(time_list)
        epoch_count_avg, epoch_count_std = avg_and_std(epoch_count_list)
        epoch_time_avg, epoch_time_std = avg_and_std(epoch_time_list)

# data, key_one, key_two, key_three, key_four, key_five
# yoo, 86.48, 89.59, 114.82, 93.55, 61.54
# yoo, 86.48, 89.59, 114.82, 93.55, 61.54
# dic, 86.48, 89.59, 114.82, 93.55, 61.54


if __name__ == '__main__':
    main()
