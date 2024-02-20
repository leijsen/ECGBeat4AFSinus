import os
import sys

import numpy
import numpy as np
import torch
import time
import random
import h5py
import argparse

from My_util import *
from torch.utils.data import Dataset
from tqdm import tqdm
from mne.filter import filter_data, notch_filter
# from scipy.signal import butter, filtfilt

class Ecgdata(Dataset):
    def __init__(self, data_path):
        self.write_list = []
        train_data, test_data, train_labels, test_labels, test_list = self.load_data(data_path)
        self.train_data = train_data
        self.valid_data = test_data
        self.train_labels = train_labels
        self.valid_labels = test_labels
        self.test_list = test_list


    def __len__(self):
        return len(self.train_data) + len(self.valid_data)
    def __getitem__(self, index):
        return self.train_data[index], self.train_labels[index]


    def load_data(self, data_path):
        save_num = config.save_num
        folder_path = f"/data/lj/PAF_net1d{save_num}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        global key_value
        csv_file_path = f'/data/lj/people_info.csv'
        df = pd.read_csv(csv_file_path)
        key_value = df.set_index('Key').to_dict()['Value']

        IDlist = []
        for file in os.listdir(data_path):
            file = file.split('.')[0]
            ID = file.split('_')[1]
            IDlist.append(ID)
        IDlist = list(set(IDlist))

        parser = argparse.ArgumentParser()
        parser.add_argument('--variable', type=str)
        args = parser.parse_args()

        if len(sys.argv) > 1:
            variable = args.variable
        else:
            variable = ""
        test_set = list(map(int, variable.split(',')))
        print(test_set)

        train_list = []
        test_list = []
        for file_name in os.listdir(data_path):
            file_num = int(file_name.split('_')[1])
            if file_num in test_set:
                test_list.append(file_name.split('.')[0])
            else:
                train_list.append(file_name.split('.')[0])
        train_list = list(set(train_list))
        test_list = list(set(test_list)) # Strings like data_0_1 are stored

        train_data, test_data, train_labels, test_labels = [], [], [], []

        print('Training data is being read')
        processor = tqdm(range(len(train_list)), file=sys.stderr)
        for name in train_list:
            result = self.get_signal(name, data_path, False)
            _x = []
            _y = []
            if result is not None:
                _x, _y = result
            if len(_x) == 0:
                continue
            else:
                _x = np.array(_x)
                _y = np.array(_y)
                train_data.extend(_x)
                train_labels.extend(_y)
            processor.update(1)

        train_data = np.expand_dims(train_data, 1)
        train_data, train_labels = np.array(train_data, dtype=np.float64), np.array(train_labels, dtype=np.int64)

        print('Reading test data')
        processor = tqdm(range(len(test_list)), file=sys.stderr)

        with h5py.File(f"{folder_path}/list_data{test_set[0]}.h5", 'w') as hf:
            for name in test_list:
                result = self.get_signal(name, data_path, True)
                _x = []
                _y = []
                if result is not None:
                    _x, _y = result
                if len(_x) == 0:
                    continue
                else:
                    self.write_list.append(name)
                    _x = np.array(_x)
                    _y = np.array(_y)
                    test_data.extend(_x)
                    test_labels.extend(_y)
                    hf_data = np.concatenate((_x, np.zeros((_x.shape[0], 2))), axis=1)
                    hf.create_dataset(f"{name}", data=hf_data)

                processor.update(1)

        test_data = np.expand_dims(test_data, 1)
        test_data, test_labels = np.array(test_data, dtype=np.float64), np.array(test_labels, dtype=np.int64)

        return train_data, test_data, train_labels, test_labels, test_list


    def get_signal(self, name, data_path, fullTest=False):
        sig, fields = wfdb.rdsamp(os.path.join(data_path, name))
        record = wfdb.rdrecord(os.path.join(data_path, name))
        sig = sig.flatten()

        name_key = int(name.split('_')[1])
        label = key_value[name_key]

        annotation = wfdb.rdann(os.path.join(data_path, name), 'atr')
        R_location = annotation.sample
        R_class = annotation.symbol

        all_segments = []
        all_labels = []


        start = 10
        end = len(R_location) - 5
        i = start
        j = end
        p_signal = record.p_signal[:, 0]
        p_signal = p_signal.flatten()

        # Filtering operations
        sfreq = 200
        low_freq = 0.5
        high_freq = 50.0
        p_signal = filter_data(p_signal, sfreq, low_freq, high_freq, method='iir', verbose=False)


        """
        A beat corresponds to a tag
        """
        while i < j:
            if R_class[i] == 'N':
                left = R_location[i] - config.beat_data // 2
                right = R_location[i] + config.beat_data // 2
                sub_sig = p_signal[left:right]
                all_segments.append(sub_sig)
                if label == 0:
                    all_labels.append(0)
                else:
                    all_labels.append(1)
            i += 1

        _x = all_segments
        _y = all_labels

        return _x, _y

    def z_score(self):
        pass



if __name__ == '__main__':
    train_data = Ecgdata(config.data_path)
    print(train_data.train_data, train_data.train_labels)
    print(train_data.train_data.shape, train_data.train_labels.shape)


