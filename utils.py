#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: LauTrueYes
# @Date  : 2020/12/27
from tqdm import tqdm
import torch
import time
from datetime import timedelta
from torch.utils.data import TensorDataset, DataLoader

PAD, CLS = '[PAD]', '[CLS]'

def load_dataset(file_path, config):
    """
    返回结果4个list：ids, label, ids_len, mask
    :param file_path:
    :param seq_len:
    :return:
    """
    contents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content, original_label = line.split('\t')
            # label = config.class2id[original_label]
            label = int(original_label)
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            pad_size = config.pad_size
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids = token_ids + ([0] * (pad_size-len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, mask, int(label)))
    return contents

def build_dataset(config):
    """
    返回值train,dev,test
    4个list：ids, label, ids_len, mask
    :param config:
    :return:
    """
    train = load_dataset(config.train_path, config)
    dev = load_dataset(config.dev_path, config)
    test = load_dataset(config.test_path, config)
    return train, dev, test



def build_data_loader(dataset, config):
    token_ids = [i[0] for i in dataset]
    mask= [i[1] for i in dataset]
    label_ids = [i[2]for i in dataset]
    iter_set = TensorDataset(torch.LongTensor(token_ids).to(config.device),
                             torch.LongTensor(mask).to(config.device),
                             torch.LongTensor(label_ids).to(config.device))
    iter = DataLoader(iter_set, batch_size=config.batch_size, shuffle=False)
    return iter

def get_time_dif(start_time):
    """
    获取已使用的时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
