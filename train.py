#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: LauTrueYes
# @Date  : 2020/12/27
import torch
import numpy as np
from sklearn import metrics
from torch.optim import AdamW

def train(config, model, train_loader, dev_loader):

    dev_best_f1 = float('-inf')
    avg_loss = []
    param_optimizer = list(model.named_parameters())    #拿到所有model中的参数
    no_decay = ['bias','LayerNorm.bias', 'LayerNorm.weight']    #不需要衰减的参数
    optimizer_grouped_parameters = [
        {'params':[p for n,p in param_optimizer if not any( nd in n for nd in no_decay) ], 'weight_decay':0.01 },
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]

    optimizer = AdamW(params = optimizer_grouped_parameters, lr = config.learning_rate)

    for epoch in range(config.num_epochs):
        train_right, train_total = 0, 0
        model.train()
        model.to(config.device)
        print('Epoch:{}/{}'.format(epoch+1, config.num_epochs))
        for batch_idx,(input_ids, attention_mask, label_ids) in enumerate(train_loader):
            input_ids, attention_mask, label_ids = input_ids.to(config.device), attention_mask.to(config.device), label_ids.to(config.device)
            input = (input_ids, attention_mask, label_ids)
            loss, predicts = model(input)

            avg_loss.append(loss.data.item())

            batch_right = (predicts == label_ids).sum().item()
            train_right += batch_right
            train_total += len(predicts)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print("Epoch:{}--------Iter:{}--------train_loss:{:.3f}--------train_acc:{:.3f}".format(epoch + 1,
                                                                                                        batch_idx + 1,
                                                                                                        np.array(avg_loss).mean(),
                                                                                                        train_right/train_total))
        dev_loss, dev_acc, dev_f1, dev_report, dev_confusion = evaluate(config, model, dev_loader)
        msg = "Dev Loss:{}--------Dev Acc:{}--------Dev F1:{}"
        print(msg.format(dev_loss, dev_acc, dev_f1))
        print("Dev Report")
        print(dev_report)
        print("Dev Confusion")
        print(dev_confusion)

        if dev_best_f1 < dev_f1:
            dev_best_f1 = dev_f1
            torch.save(model.state_dict(), config.save_path)
            print("***************************** Save Model *****************************")


def evaluate(config, model, dev_loader):

    loss_all = np.array([], dtype=float)
    predict_all = np.array([], dtype=int)
    label_all = np.array([], dtype=int)
    with torch.no_grad():   #不需要梯度
        model.eval()  # 开启评估模式
        for i, (input_ids, attention_mask, label_ids) in enumerate(dev_loader):
            input_ids, attention_mask, label_ids = input_ids.to(config.device), attention_mask.to(config.device), label_ids.to(config.device)
            input = (input_ids, attention_mask, label_ids)
            loss, label_predict = model(input)

            loss_all = np.append(loss_all, loss.data.item())
            predict_all = np.append(predict_all, label_predict.data.cpu().numpy())
            label_all = np.append(label_all, label_ids.data.cpu().numpy())
        acc = metrics.accuracy_score(label_all, predict_all)
        f1 = metrics.f1_score(label_all, predict_all, average='macro')
        report = metrics.classification_report(label_all, predict_all, target_names=config.class_list, digits=3)
        confusion = metrics.confusion_matrix(label_all, predict_all)

        return loss.mean(), acc, f1, report, confusion

