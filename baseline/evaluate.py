#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: jby
@Date: 2020-05-20 10:56:49
@LastEditTime: 2020-07-13 20:41:44
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ptg/evaluate.py
'''
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from dataset import collate_fn
import config


def evaluate(model, val_data, epoch):
    print('validating')

    val_loss = []
    with torch.no_grad():
        DEVICE = torch.device("cuda" if config.is_cuda else "cpu")
        val_dataloader = DataLoader(dataset=val_data,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    pin_memory=True, drop_last=True,
                                    collate_fn=collate_fn)
        batch_nums = len(val_dataloader)
        for i, data in enumerate(tqdm(val_dataloader)):
            x, y, x_len, y_len, oov, len_oovs = data
            if config.is_cuda:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)
            loss = model(x, x_len, y, len_oovs, epoch, batch_nums=batch_nums)
            val_loss.append(loss.item())
    return np.mean(val_loss)

