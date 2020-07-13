#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@Author: lpx, jby
@Date: 2020-07-13 11:07:48
@LastEditTime: 2020-07-13 19:06:39
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /baseline/utils.py
'''

import numpy as np
import config
import time


def timer(module):
    def wrapper(func):
        def cal_time(*args, **kwargs):
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for ', module)
            return res
        return cal_time
    return wrapper


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


def simple_tokenizer(text):
    return text.split()


def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence:
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


def sort_batch_by_len(data_batch):
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    # Sort indices of data in batch by lengths.
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()

    data_batch = {
        name: [_tensor[i] for i in sorted_indices]
        for name, _tensor in res.items()
    }
    return data_batch


def outputids2words(id_list, source_oovs, vocab):
    """
        Maps output ids to words, including mapping in-source OOVs from their temporary ids
        to the original OOV string(applicable in pointer-generator mode).
        Args:
            id_list: list of ids (integers)
            vocab: Vocabulary object
            source_oovs: list of OOV words (strings) in the order corresponding to their temporary source OOV ids 
                         (that have been assigned in pointer-generator mode), or None (in baseline mode)
        Returns:
            words: list of words (strings)
    """
    words = []
    for i in id_list:
        try:
            w = vocab.index2word[i]  # might be [UNK]
        except IndexError:  # w is OOV
            assert source_oovs is not None, "Error: cannot find the ID the in the vocabulary."
            source_oov_idx = i - vocab.size()
            try:
                w = source_oovs[source_oov_idx]
            except ValueError:  # i doesn't correspond to an source oov
                raise ValueError(
                    'Error: model produced word ID %i corresponding to source OOV %i but this example only has %i source OOVs'
                    % (i, source_oov_idx, len(source_oovs)))
        words.append(w)
    return ' '.join(words)


def source2ids(source_words, vocab):
    """Map the source words to their ids and return a list of OOVs in the source.
    Args:
        source_words: list of words (strings)
        vocab: Vocabulary object
    Returns:
        ids:
        A list of word ids (integers); OOVs are represented by their temporary source OOV number. If the vocabulary size is 50k and the source has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
    oovs:
        A list of the OOV words in the source (strings), in the order corresponding to their temporary source OOV numbers.
    """
    ids = []
    oovs = []
    unk_id = vocab["<UNK>"]
    for w in source_words:
        i = vocab[w]
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(
                w
            ) # This is 0 for the first source OOV, 1 for the second source OOV...
            ids.append(
                vocab.size() + oov_num
            ) # This is e.g. 20000 for the first source OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


class Beam(object):
    def __init__(self,
                 tokens,
                 log_probs,
                 decoder_states,
                 attention_weights,
                 max_oovs,
                 encoder_input):
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.attention_weights = attention_weights
        self.max_oovs = max_oovs
        self.encoder_input = encoder_input

    def extend(self,
               token,
               log_prob,
               decoder_states,
               attention_weights,
               max_oovs,
               encoder_input):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    attention_weights=attention_weights,
                    max_oovs=max_oovs,
                    encoder_input=encoder_input)

    def seq_score(self):
        """
        This function calculate the score of the current sequence. 
        """
        score = sum(self.log_probs) / len(self.tokens)
        return score
