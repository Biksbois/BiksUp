#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from icecream import ic
import sys


class GNN(Module):
    def __init__(self, hidden_size, cur_key, step=1):
        super(GNN, self).__init__()
        torch.manual_seed(0)
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        
        self.gate_size_mult = cur_key.count_active_weights()
        self.gate_size = self.gate_size_mult * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))
        # remember to change self.hidden_size*3 to self.hidden_size*2
        self.linear_noGRU = nn.Linear(self.hidden_size, self.hidden_size*3, bias=True)
        self.linear_1 = nn.Linear(self.hidden_size*3, self.hidden_size*2, bias=True)
        self.linear_2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden, cur_key):
        if not cur_key.use_GRU():
            # inputs = torch.cat([inputs, hidden], 2)
            inputs = self.linear_noGRU(hidden)
            inputs = F.relu(inputs)
            inputs = self.linear_1(inputs)
            inputs = F.relu(inputs)
            inputs = self.linear_2(inputs)
            inputs = F.relu(inputs)
            return inputs 
        else:
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            if self.gate_size_mult == 3:
                gi = F.linear(inputs, self.w_ih, self.b_ih)
                gh = F.linear(hidden, self.w_hh, self.b_hh)
                i_r, i_i, i_n = gi.chunk(3,2)
                h_r, h_i, h_n = gh.chunk(3,2)
            elif self.gate_size_mult == 2:
                gi = F.linear(inputs, self.w_ih, self.b_ih)
                gh = F.linear(hidden, self.w_hh, self.b_hh)
                if cur_key.use_reset_GRU_weights() and cur_key.use_update_GRU_weights():
                    i_r, i_i = gi.chunk(2,2)
                    h_r, h_i = gh.chunk(2,2)
                    ones = torch.ones(self.input_size, 1 * self.hidden_size)
                    ones = trans_to_cuda(ones)
                    i_n = torch.matmul(inputs, ones)
                    h_n = hidden
                elif cur_key.use_reset_GRU_weights() and cur_key.use_newgate_GRU_weights():
                    i_r, i_n = gi.chunk(2,2)
                    h_r, h_n = gh.chunk(2,2)
                    ones = torch.ones(self.input_size, 1 * self.hidden_size)
                    ones = trans_to_cuda(ones)
                    i_i = torch.matmul(inputs, ones)
                    h_i = hidden
                elif cur_key.use_newgate_GRU_weights() and cur_key.use_update_GRU_weights():
                    i_i, i_n = gi.chunk(2,2)
                    h_i, h_n = gh.chunk(2,2)
                    ones = torch.ones(self.input_size, 1 * self.hidden_size)
                    ones = trans_to_cuda(ones)
                    i_r = torch.matmul(inputs, ones)
                    h_r = hidden
            elif self.gate_size_mult == 1:
                ones = torch.ones(self.input_size, self.gate_size)
                ones = trans_to_cuda(ones)
                gi = torch.matmul(inputs, ones)
                
                if cur_key.use_reset_GRU_weights():
                    i_r = F.linear(inputs, self.w_ih, self.b_ih)
                    h_r = F.linear(hidden, self.w_hh, self.b_hh)
                    
                    i_i = gi
                    h_i = hidden
                    
                    i_n = gi
                    h_n = hidden
                elif cur_key.use_update_GRU_weights():
                    i_r = gi
                    h_r = hidden
                    
                    i_i = F.linear(inputs, self.w_ih, self.b_ih)
                    h_i = F.linear(hidden, self.w_hh, self.b_hh)
                    
                    i_n = gi
                    h_n = hidden
                elif cur_key.use_newgate_GRU_weights():
                    i_r = gi
                    h_r = hidden
                    
                    i_i = gi
                    h_i = hidden
                    
                    i_n = F.linear(inputs, self.w_ih, self.b_ih)
                    h_n = F.linear(hidden, self.w_hh, self.b_hh)
            else:
                ones = torch.ones(self.input_size, self.hidden_size * 3)
                ones = trans_to_cuda(ones)
                gi = torch.matmul(inputs, ones)
                i_r, i_i, i_n = gi.chunk(3, 2)
                h_r, h_i, h_n = hidden, hidden, hidden
            
            if cur_key.use_reset_sigmoid():
                resetgate = torch.sigmoid(i_r + h_r)
            else:
                resetgate = i_r + h_r
            
            if cur_key.use_input_sigmoid():
                inputgate = torch.sigmoid(i_i + h_i)
            else:
                inputgate = i_i + h_i
            
            if cur_key.use_newgate_tahn():
                newgate = torch.tanh(i_n + resetgate * h_n)
            else:
                newgate = i_n + resetgate * h_n
            
            hy = newgate + inputgate * (hidden - newgate)
            return hy

    def forward(self, A, hidden, cur_key):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden, cur_key)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node, cur_key):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, cur_key, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, cur_key):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size

        if not cur_key.use_nonhybrid():
            if cur_key.use_attention():
                q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
                q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
                alpha = self.linear_three(torch.sigmoid(q1 + q2))
                a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
            elif cur_key.use_weighted_attention():
                alpha = 1/torch.sum(mask,1)
                alpha = alpha[:, None, None]
                
                alpha = trans_to_cuda(alpha)
                a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
            else:
                a = torch.sum(hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
            
            if cur_key.use_local():
                a = self.linear_transform(torch.cat([a, ht], 1))

        elif cur_key.use_local():
            a = ht
        
        b = self.embedding.weight[1:]
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A, cur_key):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden, cur_key)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, cur_key):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(np.array(A)).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A, cur_key)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask, cur_key)


def train_test(model, train_data, test_data, cur_key):
    model.scheduler.step()
    model.train()
    total_loss = 0.0
    loss_list = []
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data, cur_key)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
        loss_list.append(loss.item())
    test_loss = []
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data, cur_key)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets_cpu = targets
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        test_loss.append(loss.item())
        for score, target, mask in zip(sub_scores, targets_cpu, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    
    return hit, mrr, loss_list, total_loss, test_loss
