import torch
import test_print as tp
import numpy as np
import random

def print_parameter():
    for i in range(1):
        # print(Parameter(torch.Tensor(10,10)))
        print(torch.Tensor(2,2) * 100)

if __name__ == '__main__':
    print_parameter()
    # print(torch.randint(0,10,(10,)))
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print(torch.randint(0,10,(10,)))

    # print(np.random.randint(10, size=10))
    # np.random.seed(0)
    # print(np.random.randint(10, size=10))
    
    print("--------------------")
    print_parameter()