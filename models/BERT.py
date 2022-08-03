#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : BERT.py
# @Author: LauTrueYes
# @Date  : 2022/8/3 9:29
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertTokenizer

class Config(object):
    """
    配置参数
    """
    def __init__(self, dataset):
        self.model_name = 'BERT'   #模型名称

        self.train_path = dataset + '/train.json'   #训练集
        self.dev_path = dataset + '/dev.json'   #验证集
        self.test_path = dataset + '/test.json' #测试集
        self.predict_path = dataset + '/saved_data/' + 'predict.json'    #预测结果
        self.value_path = dataset + '/saved_data/' + 'value.csv'        #评价效果
        self.save_path = dataset + '/saved_data/' + 'model.ckpl'

        self.label_list = [x.strip() for x in open(dataset + '/class.txt', encoding='utf-8').readlines()]   #类别
        self.num_labels = len(self.label_list) #类别数量
        self.label2id = {cls:id for id, cls in enumerate(self.label_list)}
        self.id2label = {j:i for i, j in self.label2id.items()}
        self.save_path = dataset + '/saved_data/' + self.model_name + '.ckpt'   #模型训练结果
        self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu') #设备配置

        self.epochs = 10 #轮次数
        self.batch_size = 64   #batch_size，一次传入128个pad_size
        self.lr = 1e-5 #学习率
        self.bert_path = './pretrained/bert-base-chinese'    #bert预训练位置
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  #bert切词器
        self.hidden_size = 768  #bert隐藏层个数，在bert_config.json中有设定，不能随意改
        self.hidden_dropout_prob = 0.1

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(config.bert_path, output_hidden_states=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.hidden_size = config.hidden_size
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, bert_inputs, label_ids, label_mask, use_crf=False):
        ids_lens = bert_inputs[3]
        batch_size, seq_len = ids_lens.shape
        mask = ids_lens.gt(0)

        input_ids = bert_inputs[0]
        attention_mask = bert_inputs[1].type_as(mask)
        token_type_ids = bert_inputs[2]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_enc_out = outputs[0]

        bert_mask = attention_mask.type_as(mask)

        bert_chunks = last_enc_out[bert_mask].split(ids_lens[mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
        bert_embed = bert_out.new_zeros(batch_size, seq_len, self.hidden_size)
        # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
        bert_embed = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)
        bert_embed = self.dropout(bert_embed)

        label_predict = self.classifier(bert_embed)

        if use_crf:
            loss = self.crf(label_predict, label_ids, label_mask)
            loss = -1 * loss
        else:
            active_logits = label_predict.view(-1, self.num_labels)
            active_labels = torch.where(label_mask.view(-1), label_ids.view(-1), self.loss_fct.ignore_index)
            loss = self.loss_fct(active_logits, active_labels)

        output = label_predict.data.argmax(dim=-1)
        return loss, output
