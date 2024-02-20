import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import wfdb
import pywt
import argparse

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('Device: ', device)

class Config():
    def __init__(self):
        # self.data_path = '../../data'
        self.data_path = '../../all_data'
        self.learning_rate = 1e-5
        self.epoch = 100
        self.dropout = 0
        self.batch_size = 32

        self.window_number = 30
        self.sub_len = 50
        self.beat_data = 200

        parser = argparse.ArgumentParser()
        parser.add_argument("--variable", type=str)
        args = parser.parse_args()
        if len(sys.argv) > 1:
            variable = args.variable
        else:
            variable = ''
        test_set = list(map(int, variable.split(',')))

        self.save_num = f'_DenoiseNormalTest{test_set[0]}'

        print("learning_rate: ", self.learning_rate)
        print("epoch: ", self.epoch)
        print("batch_size: ", self.batch_size)
        print("save_num: ", self.save_num)

config = Config()