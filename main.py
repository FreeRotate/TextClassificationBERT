#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: LauTrueYes
# @Date  : 2020/12/27
import time
import torch
import numpy as np
import utils
import argparse
from train import train
from test import test
from predict import predict
from importlib import import_module

parser = argparse.ArgumentParser(description='TextClassification')
parser.add_argument('--model', type=str, default='BERT', help='BERT')  #在defaule中修改所需的模型
args = parser.parse_args()


if __name__ == '__main__':
    dataset = './data/NewsTitle'    #数据集地址
    model_name = args.model
    lib = import_module('models.'+model_name)
    config = lib.Config(dataset)
    model = lib.Model(config).to(config.device)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True   #保证每次运行结果一样

    start_time = time.time()
    print('加载数据集')
    train_data, dev_data, test_data = utils.build_dataset(config)
    train_loader = utils.build_data_loader(train_data, config)
    dev_loader = utils.build_data_loader(dev_data, config)
    test_loader = utils.build_data_loader(test_data, config)

    time_dif = utils.get_time_dif(start_time)
    print("模型开始之前，准备数据时间：",time_dif)

    train(config, model, train_loader, dev_loader)
    predict(config, model, test_loader)
    test(config, model, test_loader)
