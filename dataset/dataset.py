import sys

import pandas as pd
import re
import torch
from bpemb import BPEmb
from sklearn.utils import shuffle
from torch.utils.data import Dataset

sys.path.append('..')
import config


class StatusDataset(Dataset):
    def __init__(self, path=config.path_to_data, mode='train'):
        self.path_to_data = path
        self.mode = mode
        print(f"Loading {self.mode} data...")
        self.data = self.read_data()
        self.preprocess_data()
        self.bpemb_ru = BPEmb(lang="ru", dim=300, vs=50000)
        self.placeholder = torch.zeros(config.max_seq_length, dtype=torch.long)

    def read_data(self):

        if self.mode == 'train':
            data = pd.read_csv(self.path_to_data)
            data = data[data.isTest == 0][['text_orig', 'label']]
        elif self.mode == 'val':
            data = pd.read_csv(self.path_to_data)
            data = data[data.isTest == 1][['text_orig', 'label']]
        elif self.mode == 'test':
            data = pd.read_parquet(self.path_to_data)
        return data

    def preprocess_data(self):
        self.data['text_orig'] = self.data['text_orig'].map(self.remove_urls)
        self.data = shuffle(self.data)
        self.data.reset_index(drop=True, inplace=True)

    def remove_urls(self, v_text):
        v_text = re.sub(r"(/[\w\-?=$&:;#@/]+)", '', v_text, flags=re.MULTILINE)
        v_text = re.sub(r'(https?:[/.]*)', '', v_text, flags=re.MULTILINE)
        return v_text

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        text = self.data['text_orig'][idx]
        label = self.data['label'][idx]
        ids_tokens = self.bpemb_ru.encode_ids(text)
        placeholder = self.placeholder.clone()
        placeholder[:len(ids_tokens)] = torch.tensor(ids_tokens)[:config.max_seq_length]
        return placeholder, torch.tensor(label)
