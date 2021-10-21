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
from utils import build_graph, Data, split_validation
from model import *
from metadata import metadataObj, get_metadata_list, data_dict
from pretty_printer import print_hyperparameters, introduce_biksup, check_if_valid
from parameter_class import parameterObj, get_parameters
from biksLog import get_logger
import sys
import os
from icecream import ic
from save_epoch import get_foldername, save_epoch, save_average


log = get_logger()
parser = argparse.ArgumentParser()
parameters = get_parameters()
for p in parameters:
    p.add_new_argument(parser)
opt = parser.parse_args()

def main():
    check_if_valid(opt)
    parsed_keys = get_metadata_list(opt.keys, opt.runall, opt.runlast)
    introduce_biksup(parameters, parsed_keys, data_dict, opt)
    print(opt.nonhybrid)
    folder_name = get_foldername()
    
    for iter in range(int(opt.iterations)):
        for i in range(len(parsed_keys)):
            cur_key = parsed_keys[i]
            # if cur_key.test_case('key_05'):
            #     print("    - perform key_05")
        
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

            model = trans_to_cuda(SessionGraph(opt, n_node))

            start = time.time()
            
            best_hit = [0,0]
            best_mrr = [0,0]
            best_epoch = [0, 0]
            best_loss = [0,0]
            best_loss_list = [[0], [0]]
            
            bad_counter = 0
            for epoch in range(opt.epoch):
                print('-------------------------------------------------------')
                print('epoch: ', epoch)
                hit, mrr, loss_list, total_loss = train_test(model, train_data, test_data, cur_key)
                flag = 0
                if hit >= best_hit[0]:
                    best_hit[0] = hit
                    best_epoch[0] = epoch
                    best_loss[0] = total_loss
                    best_loss_list[0] = loss_list
                    flag = 1
                if mrr >= best_mrr[1]:
                    best_mrr[1] = mrr
                    best_epoch[1] = epoch
                    best_loss[1] = total_loss
                    best_loss_list[1] = loss_list
                    flag = 1
                print('Best Result:')
                print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d'% (best_hit[0], best_mrr[0], best_epoch[0]))
                print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d'% (best_hit[1], best_mrr[1], best_epoch[1]))
                bad_counter += 1 - flag
                if bad_counter >= opt.patience:
                    break
            print('-------------------------------------------------------')
            end = time.time()
            print("Run time: %f s" % (end - start))
            minutes = (end - start)/60
            temp_folder_name = os.path.join(str(iter), folder_name)
            save_epoch(temp_folder_name, cur_key.get_key(), [minutes, minutes], best_epoch, best_mrr, best_hit, best_loss, best_loss_list)
    save_average(iter, folder_name, parsed_keys)


if __name__ == '__main__':
    main()
