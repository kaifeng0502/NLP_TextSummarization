#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: jby
@Date: 2020-07-13 11:00:51
@LastEditTime: 2020-07-15 10:50:04
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /JD_project_2/baseline/model/main.py
'''

import config
import torch
from dataset import MyDataset
from train import train

DEVICE = torch.device('cuda') if config.is_cuda else torch.device('cpu')
dataset = MyDataset(config.data_path,
                    max_src_len=config.max_src_len,
                    max_tgt_len=config.max_tgt_len,
                    truncate_src=config.truncate_src,
                    truncate_tgt=config.truncate_tgt)
val_dataset = MyDataset(config.val_data_path,
                        max_src_len=config.max_src_len,
                        max_tgt_len=config.max_tgt_len,
                        truncate_src=config.truncate_src,
                        truncate_tgt=config.truncate_tgt)

vocab = dataset.build_vocab(embed_file=config.embed_file)
# embedding_weights = torch.from_numpy(vocab.embeddings)
train(dataset, val_dataset, vocab, start_epoch=0)
