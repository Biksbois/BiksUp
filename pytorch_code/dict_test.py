import argparse
import pickle
import time
import sys
import os

if __name__ == '__main__':
    datasets = []
    for fnames in os.listdir(os.getcwd()+'/datasets/'):
        if '1_' in fnames:
            datasets.append(fnames)
    print(datasets)
