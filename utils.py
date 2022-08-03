#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: LauTrueYes
# @Date  : 2022/8/3 10:34
import json
import torch

def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            item = json.loads(line)
            dataset.append(item)
    return dataset

class DataLoader(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for index in range(len(self.dataset)):
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch):
            yield batch
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def bert_word2id(sentence_list, config):
    #接收一个batch的数据
    ids_list = []
    lens_list = []
    segment_list = []
    mask_list = []
    tokenizer = config.tokenizer
    max_ids_lens = 0
    max_lens_lens = 0

    for sentence in sentence_list:
        ids = []
        lens = []
        for word in sentence:
            id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            for i in id:
                ids.append(i)
            lens.append(len(id))
        if len(ids) > max_ids_lens:
            max_ids_lens = len(ids)
        if len(lens) > max_lens_lens:
            max_lens_lens = len(lens)

        ids_list.append(ids)    #单词后的编码
        lens_list.append(lens)  #原始单词被切分编码后长度

    mask_list += [[1]*len(ids) for ids in ids_list]
    for index, item in enumerate(ids_list):
        pad_size = max_ids_lens - len(item)
        ids_list[index] += [0 for i in range(pad_size)]
        mask_list[index] += [0 for i in range(pad_size)]


    for index, item in enumerate(lens_list):
        pad_size = max_lens_lens - len(item)
        lens_list[index] += [0 for i in range(pad_size)]

    segment_list += [[0] * len(ids) for ids in ids_list]

    return ids_list, segment_list, mask_list, lens_list


def batch_variable(batch_data, config):
    sentence_list = []
    labels_list = []

    for index, item in enumerate(batch_data):
        sentence = item['text']
        labels = ['O'] * len(sentence)

        for label_name, tag in item['label'].items():
            for sub_name, sub_index in tag.items():
                for start_index, end_index in sub_index:
                    assert ''.join(sentence[start_index:end_index + 1]) == sub_name
                    if start_index == end_index:
                        labels[start_index] = 'B-' + label_name
                    else:
                        labels[start_index] = 'B-' + label_name
                        labels[start_index + 1:end_index + 1] = ['I-' + label_name] * (len(sub_name) - 1)
        sentence_list.append([word for word in sentence])
        labels_list.append(labels)

    batch_size = len(batch_data)
    max_seq_len = max(len(insts['text']) for insts in batch_data)
    label_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    for index, labels in enumerate(labels_list):
        label_ids[index, :len(labels)] = torch.tensor([config.label2id[label] for label in labels])
        label_mask[index, :len(labels)].fill_(1)

    ids_list, segment_list, mask_list, lens_list = bert_word2id(sentence_list, config)
    id_list = torch.LongTensor(ids_list)
    mask_list = torch.LongTensor(mask_list)
    segment_list = torch.LongTensor(segment_list)
    lens_list = torch.LongTensor(lens_list)
    bert_inputs = [id_list.to(config.device), mask_list.to(config.device), segment_list.to(config.device),  lens_list.to(config.device)]

    return bert_inputs, label_ids.to(config.device), label_mask.to(config.device)
