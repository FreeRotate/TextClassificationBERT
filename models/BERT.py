#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : BERT.py
# @Author: LauTrueYes
# @Date  : 2020/12/27
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertTokenizer

class Config(object):
    """
    配置参数
    """
    def __init__(self, dataset):
        self.model_name = 'BERT'   #模型名称

        self.train_path = dataset + '/train.txt'   #训练集
        self.test_path = dataset + '/test.txt' #测试集
        self.dev_path = dataset + '/dev.txt'   #验证集
        self.predict_path = dataset + '/' + self.model_name +'_predict.txt'   #预测数据

        self.class_list = [x.strip() for x in open(dataset + '/class.txt', encoding='utf-8').readlines()]   #类别
        self.class2id = {cls:id for id, cls in enumerate(self.class_list)}
        self.id2class = {j:i for i, j in self.class2id.items()}
        self.save_path = dataset + '/saved_data/' + self.model_name + '.ckpt'   #模型训练结果
        self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu') #设备配置

        self.require_improvement = 1000 #若超过1000batch效果还没有提升，提前结束训练

        self.num_classes = len(self.class_list) #类别数量
        self.num_epochs = 1 #轮次数
        self.batch_size = 64   #batch_size，一次传入128个pad_size
        self.pad_size = 60   #每句话处理长度（短填，长切）
        self.learning_rate = 1e-5 #学习率
        self.bert_path = './pretrained/bert-base-chinese'    #bert预训练位置
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  #bert切词器
        self.hidden_size = 768  #bert隐藏层个数，在bert_config.json中有设定，不能随意改
        self.hidden_dropout_prob = 0.1


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()   #继承父类方法
        self.num_classes = config.num_classes
        self.bert = BertModel.from_pretrained(config.bert_path) #加载预训练模型
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        for param in self.bert.parameters():    #加载bert所有参数
            param.requires_grad = True  #需要梯度,需要微调，一般都设定为True
        # 以上为原生BERT

        self.classifier = nn.Linear(config.hidden_size, config.num_classes)


    def forward(self, input):
        #x是输入数据其中有：[ids, seq_len, mask]
        input_ids, attention_mask, labels = input[0], input[1], input[2]
        outputs = self.bert(input_ids=input_ids)   #shape[batch_size, hidden_size]
        sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        #不需要encoded_layers，只需要pooled_output返回得到pooled
        logits = self.classifier(sequence_output)   #shape[batch_size, num_classes]
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        if loss is not None:
            return loss, logits.argmax(dim=-1)
        else:
            return logits.argmax(dim=-1)
