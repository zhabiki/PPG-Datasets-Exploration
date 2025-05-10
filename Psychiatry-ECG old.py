import torch
import pandas as pd
import os
import numpy as np
import scipy.io
import re

def extract_number(filename):
    match = re.search(r'\((\d+)\)', filename)  # Ищем число в имени файла
    if match:
        return int(match.group(1))  # Возвращаем найденное число как целое число
    return float('inf')  # Если числа нет, ставим "бесконечность", чтобы такие файлы были в конце

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.path = 'DATASETS/Psychiatry ECG/'
        self.classes = os.listdir(self.path)

        self.samples = {c: sorted(os.listdir(self.path + c), key=extract_number) for c in self.classes}
        self.pinned_disorder = None

    def set_disorder(self, disorder):
        assert disorder in list(self.samples.keys())

        self.pinned_disorder = disorder

    def __getitem__(self, index):
        state, sample = index
        if state in ['Bipolar', 'Schizophrenia']:
            k = 13
        elif state == 'Depression':
            k = 11

        sample = [i for i in range(sample*k, sample+k)]
        full_data = scipy.io.loadmat(self.path + state + '/' + self.samples[state][sample[0]])['sinyal']
        for i in range(1, len(sample)):
            signal = scipy.io.loadmat(self.path + state + '/' + self.samples[state][sample[i]])['sinyal']
            full_data = np.append(full_data, signal, axis=0)

        return full_data, 1
    
    def __len__(self):
        if self.pinned_disorder in ['Bipolar' or 'Schizophrenia']:
            return len(os.listdir(self.path + self.pinned_disorder))//13
        elif self.pinned_disorder == 'Depression':
            return len(os.listdir(self.path + self.pinned_disorder))//11
