#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : GRU.py
# @Author: LauTrueYes
# @Date  : 2022/8/3 9:06
import torch
import torch.nn as nn
from torchcrf import CRF

class Model(nn.Module):
    def __init__(self, vocab_len, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.embed = nn.Embedding(num_embeddings=vocab_len, embedding_dim=config.embed_dim)
        self.gru = nn.GRU(input_size=config.embed_dim, hidden_size=config.hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, word_ids, label_ids=None, label_mask=None, use_crf=True):
        word_embed = self.embed(word_ids)
        sequence_output, _ = self.gru(word_embed)
        logits = self.classifier(sequence_output)
        if label_ids != None:
            if use_crf:
                loss = self.crf(logits, label_ids, label_mask)
                loss = -1 * loss
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(label_mask.view(-1), label_ids.view(-1), self.loss_fct.ignore_index)
                loss = self.loss_fct(active_logits, active_labels)

        else:
            loss = None

        return loss, logits.argmax(dim=-1)
