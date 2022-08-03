#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: LauTrueYes
# @Date  : 2022/8/3 13:17
import torch
from train import evaluate

def test(model, test_loader, config):
    model.load_state_dict(torch.load(config.save_path), False)
    model.eval()
    test_loss, test_acc, test_f1, test_report = evaluate(model, test_loader, config, vocab)
    msg = "Test Loss:{}--------Test Acc:{}--------Test F1:{}"
    print(msg.format(test_loss, test_acc, test_f1))
    print("Test Report")
    print(test_report)