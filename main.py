#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: LauTrueYes
# @Date  : 2020/12/27
import argparse
from utils import load_dataset, DataLoader
from train import train
from importlib import import_module

parser = argparse.ArgumentParser(description='NER')
parser.add_argument('--model', type=str, default='BERT', help='BERT')  #在defaule中修改所需的模型
args = parser.parse_args()

if __name__ == '__main__':
    dataset = './data/CLUENER/'
    model_name = args.model
    lib = import_module('models.' + model_name)
    config = lib.Config(dataset)
    model = lib.Model(config).to(config.device)

    train_CL = load_dataset(config.train_path)
    dev_CL = load_dataset(config.dev_path)
    test_CL = load_dataset(config.test_path)

    train_loader = DataLoader(train_CL, config.batch_size)
    dev_loader = DataLoader(dev_CL, config.batch_size)
    test_loader = DataLoader(test_CL, config.batch_size)

    train(model=model, train_loader=train_loader, dev_loader=dev_loader, config=config)

