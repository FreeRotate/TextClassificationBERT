#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : BERT.py
# @Author: LauTrueYes
# @Date  : 2020/12/27

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel


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




