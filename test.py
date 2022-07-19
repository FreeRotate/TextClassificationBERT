#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: LauTrueYes
# @Date  : 2020/12/27
import torch
from train import evaluate

def test(config, model, test_loader):
    """
    模型测试
    :param config:
    :param model:
    :param test_loader:
    :return:
    """
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test_loss, test_acc, test_f1, test_report, test_confusion = evaluate(config, model, test_loader)
    msg = "Dev Loss:{}--------Dev Acc:{}--------Dev F1:{}"
    print(msg.format(test_loss, test_acc, test_f1))
    print("Dev Report")
    print(test_report)
    print("Dev Confusion")
    print(test_confusion)