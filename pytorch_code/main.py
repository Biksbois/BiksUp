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
from global_items import TRUE_LIST
from utils import build_graph, Data, split_validation
from model import *
from metadata import metadataObj, get_metadata_list, data_dict
from pretty_printer import introduce_biksup, check_if_valid, print_best_results
from parameter_class import get_parameters
from biksLog import get_logger
import sys
import os
from icecream import ic
from global_items import NEW_CSV
from save_epoch import get_foldername
from rich.progress import Progress
from aggregate import avg_and_std, transpose_files
from intro_file import create_introduce_file
import big_o
from train_method import train_model, save_all_data, load_dataset


def ensure_valid_big_o_config(log, parsed_keys, opt):
    if len(parsed_keys) != 1:
        log.exception("You can only have one key when running big_o")
        sys.exit()
    if len(opt.dataset) != 1:
        log.exception("You can only have one dataset when running big_o")
        sys.exit()
    if opt.iterations != 1:
        log.warning("--iterations parameter is obselete when running big_o. The input has been ignored.")

def big_o_run(dataset):
    train_data = dataset[0]
    test_data = dataset[1]
    
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    
    n_node = 310 #TODO: HERE
    cur_key = metadataObj('0110')
    model = trans_to_cuda(SessionGraph(opt, n_node, cur_key))
    _, _, _, _, _, _, _, _, _ = train_model(model, train_data, test_data, cur_key, opt)

def get_n_sessions(n):
    train_data = pickle.load(open('../datasets/' + "sample" + '/train.txt', 'rb')) #TODO: HERE
    test_data = pickle.load(open('../datasets/' + "sample" + '/test.txt', 'rb')) #TODO: HERE
    
    train_data = (train_data[0][:n], train_data[1][:n])
    test_n = math.ceil(n*0.3)
    test_data = (test_data[0][:test_n], test_data[1][:test_n])
    
    print(f"Train: {len(train_data[0])} - Test: {len(test_data[0])}")
    
    return train_data, test_data

def big_o_main():
    best, others = big_o.big_o(big_o_run, get_n_sessions, n_repeats=2, min_n=1, max_n=1200) #TODO: HERE
    log.debug(best)
    # log.debug(others)

def main(opt):
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

                    n_node, train_data, _, test_data = load_dataset(opt, dataset)

                    model = trans_to_cuda(SessionGraph(opt, n_node, cur_key))

                    start = time.time()
                    
                    best_hit, best_mrr, best_epoch, _, _, epoch_loss_list, inner_test_loss, epoch_time_list, epoch_count = train_model(model, train_data, test_data, cur_key, opt)
                    
                    end = time.time()
                    
                    progress.update(progress_list[i], advance=1)
                    progress.refresh()
                    minutes = (end - start)/60
                    
                    avg_test_loss, _ = avg_and_std(inner_test_loss)
                    avg_loss, _ = avg_and_std(epoch_loss_list)
                    avg_val, _ = avg_and_std(epoch_time_list)
                    
                    hit_list.append(best_hit[0])
                    mrr_list.append(best_mrr[1])
                    time_list.append(minutes)
                    epoch_count_list.append(epoch_count)
                    epoch_time_list.append(avg_val)
                    outer_loss_list.append(avg_loss)
                    outer_test_loss.append(avg_test_loss)
                    
                    print_best_results(best_hit, best_mrr, best_epoch, cur_key.get_key(), iter)
                
                save_all_data(dataset, folder_name, cur_key, 
                            hit=hit_list, 
                            mrr=mrr_list, 
                            totaltime=time_list,
                            epochcount=epoch_count_list, 
                            epochtime=epoch_time_list, 
                            epochloss=outer_loss_list, 
                            epochtestloss=outer_test_loss
                )
    transpose_files(folder_name)
    create_introduce_file(opt.iterations, folder_name, parsed_keys, dataset, NEW_CSV, key_str)

if __name__ == '__main__':
    log = get_logger()
    parser = argparse.ArgumentParser()
    parameters = get_parameters()
    
    for p in parameters:
        p.add_new_argument(parser)
    
    opt = parser.parse_args()
    
    if opt.big_o in TRUE_LIST:
        parsed_keys = get_metadata_list(opt.keys, opt.runall, opt.runlast)
        ensure_valid_big_o_config(log, parsed_keys, opt)
        best, others = big_o_main()
    else:
        main(opt)
    
