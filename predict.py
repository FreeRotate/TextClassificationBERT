#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict.py
# @Author: LauTrueYes
# @Date  : 2020/12/27

import torch

def predict(config, model, test_iter):
    """

    :param config:
    :param model:
    :param test_iter:
    :return:
    """
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    predict_labels = []
    with torch.no_grad():   #不需要梯度
        for i, (input_ids, attention_mask, labels) in enumerate(test_iter):
            input_ids, attention_mask, labels = input_ids.to(config.device), attention_mask.to(config.device), labels.to(config.device)
            input = (input_ids, attention_mask, None)
            label_predict = model(input)

            predict = [config.id2class[i] for i in label_predict.tolist()]
            predict_labels.append(predict)
    with open(config.predict_path, 'a', encoding='utf-8') as p:
        with open(config.test_path, 'r', encoding='utf-8') as t:
            i, j = 0, 0
            for line in t:
                line = line.strip()
                if not line:
                    continue
                content, label = line.split('\t')
                predict_label = predict_labels[i][j]
                predict_data = str(content) + '\t' + predict_label + '\n'
                j += 1
                if j == config.batch_size:
                    i += 1
                    j = 0
                p.write(predict_data)
