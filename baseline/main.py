#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: jby
@Date: 2020-05-20 10:58:59
@LastEditTime: 2020-07-13 14:36:58
@LastEditors: lpx
@Description: In User Settings Edit
@FilePath: /baseline/main.py
'''
import config
import torch
from dataset import MyDataset
from train import train

DEVICE = torch.device('cuda') if config.is_cuda else torch.device('cpu')
dataset = MyDataset(config.data_path, max_src_len=config.max_src_len, max_tgt_len=config.max_tgt_len,
                    truncate_src=config.truncate_src, truncate_tgt=config.truncate_tgt)
val_dataset=MyDataset(config.val_data_path, max_src_len=config.max_src_len, max_tgt_len=config.max_tgt_len,
                    truncate_src=config.truncate_src, truncate_tgt=config.truncate_tgt)

vocab = dataset.build_vocab(embed_file=config.embed_file)
# embedding_weights = torch.from_numpy(vocab.embeddings)    
train(dataset, val_dataset, vocab, start_epoch=0)
