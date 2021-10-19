#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *
from metadata import metadataObj, get_metadata_list, data_dict
from pretty_printer import print_hyperparameters, introduce_biksup, check_if_valid
from parameter_class import parameterObj, get_parameters
from biksLog import get_logger
import sys
from icecream import ic


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
    
    for i in range(len(parsed_keys)):
        print(f" - For key number {i}, this iteration wants to:")

        cur_key = parsed_keys[i]
        
        if cur_key.test_case('key_01'):
            print("    - perform key_01")
        
        if cur_key.test_case('key_02'):
            print("    - perform key_02")
        
        if cur_key.test_case('key_03'):
            print("    - perform key_03")
        
        if cur_key.test_case('key_04'):
            print("    - perform key_04")
        
        if cur_key.test_case('key_05'):
            print("    - perform key_05")
        
        if cur_key.test_case('key_06'):
            print("    - perform key_06")
        
        if cur_key.test_case('key_07'):
            print("    - perform key_07")
        
        if cur_key.test_case('key_08'):
            print("    - perform key_08")
    
    # train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    # if opt.validation:
    #     train_data, valid_data = split_validation(train_data, opt.valid_portion)
    #     test_data = valid_data
    # else:
    #     test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    # # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # # g = build_graph(all_train_seq)
    # train_data = Data(train_data, shuffle=True)
    # test_data = Data(test_data, shuffle=False)
    # # del all_train_seq, g
    # if opt.dataset == 'diginetica':
    #     n_node = 43098
    # elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    #     n_node = 37484
    # else:
    #     n_node = 310

    # model = trans_to_cuda(SessionGraph(opt, n_node))

    # start = time.time()
    # best_result = [0, 0]
    # best_epoch = [0, 0]
    # bad_counter = 0
    # for epoch in range(opt.epoch):
    #     print('-------------------------------------------------------')
    #     print('epoch: ', epoch)
    #     hit, mrr = train_test(model, train_data, test_data)
    #     flag = 0
    #     if hit >= best_result[0]:
    #         best_result[0] = hit
    #         best_epoch[0] = epoch
    #         flag = 1
    #     if mrr >= best_result[1]:
    #         best_result[1] = mrr
    #         best_epoch[1] = epoch
    #         flag = 1
    #     print('Best Result:')
    #     print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
    #     bad_counter += 1 - flag
    #     if bad_counter >= opt.patience:
    #         break
    # print('-------------------------------------------------------')
    # end = time.time()
    # print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
