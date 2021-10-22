import torch
from torch.nn import Parameter, Module

def print_parameter():
    for i in range(5):
        # print(Parameter(torch.Tensor(10,10)))
        print(torch.Tensor(2,2) * 100)