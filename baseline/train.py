#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: jby
@Date: 2020-07-13 12:31:25
@LastEditTime: 2020-07-13 14:36:52
@LastEditors: lpx
@Description: In User Settings Edit
@FilePath: /baseline/train.py
'''

import torch
from model import Seq2seq
import os
import numpy as np
from dataset import TextDataset
from torch import optim
from torch.utils.data import DataLoader
import pickle
from torch.nn.utils import clip_grad_norm_
from dataset import collate_fn
from tqdm import tqdm
from evaluate import evaluate
import config
from tensorboardX import SummaryWriter


def train(dataset, val_dataset, v, start_epoch=0):
    """Train the model, evaluate it and store it.

    Args:
        dataset (dataset.MyDataset): The training dataset.
        val_dataset (dataset.MyDataset): The evaluation dataset.
        v (vocab.Vocab): The vocabulary built from the training dataset.
        start_epoch (int, optional): The starting epoch number. Defaults to 0.
    """
    print('loading model')
    DEVICE = torch.device("cuda" if config.is_cuda else "cpu")

    model = Seq2seq(v)
    model.load_model()
    model.to(DEVICE)

    # forward
    print("loading data")
    train_data = TextDataset(dataset.pairs, v)
    val_data = TextDataset(val_dataset.pairs, v)

    print("initializing optimizer")

    optimizer = optim.Adagrad(model.parameters(),
                              lr=config.learning_rate,
                              lr_decay=config.lr_decay,
                              initial_accumulator_value=config.initial_accumulator_value)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    val_losses = np.inf
    if (os.path.exists(config.losses_path)):
        with open(config.losses_path, 'rb') as f:
            val_losses = pickle.load(f)

    writer = SummaryWriter()
    with tqdm(total=config.epochs) as epoch_progress:
        for epoch in range(start_epoch, config.epochs):
            batch_losses = []
            with tqdm(total=len(train_dataloader) // config.batch_size) as batch_progress:
                for i, data in enumerate(tqdm(train_dataloader)):
                    model.train()
                    x, y, x_len, y_len, oov, len_oovs = data
                    if config.is_cuda:
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        x_len = x_len.to(DEVICE)
                        len_oovs = len_oovs.to(DEVICE)

                    optimizer.zero_grad()
                    assert not np.any(np.isnan(x.cpu().numpy()))
                    loss = model(x, x_len, y, len_oovs, batch=i)
                    batch_losses.append(loss.item())
                    loss.backward()
                    clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
                    clip_grad_norm_(model.decoder.parameters(), config.max_grad_norm)
                    clip_grad_norm_(model.attention.parameters(), config.max_grad_norm)
                    optimizer.step()
                    if (i % 100) == 0:
                        batch_progress.set_description(f'Epoch {epoch}')
                        batch_progress.set_postfix(Batch=i, Loss=loss.item())
                        batch_progress.update()
                        writer.add_scalar(f'Average loss for epoch {epoch}', np.mean(batch_losses), global_step=i)

            epoch_loss = np.mean(batch_losses)
            epoch_progress.set_description(f'Epoch {epoch}')
            epoch_progress.set_postfix(Loss=epoch_loss)
            epoch_progress.update()

            avg_val_loss = evaluate(model, val_data, epoch)

            print('training loss:{}'.format(epoch_loss),
                  'validation loss:{}'.format(avg_val_loss))
            if (avg_val_loss < val_losses):
                torch.save(model.encoder, config.encoder_save_name)
                torch.save(model.decoder, config.decoder_save_name)
                torch.save(model.attention, config.attention_save_name)
                torch.save(model.reduce_state, config.reduce_state_save_name)
                val_losses = avg_val_loss
            with open(config.losses_path, 'wb') as f:
                pickle.dump(val_losses, f)

    writer.close()
