#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx
@Date: 2020-07-13 11:00:51
@LastEditTime: 2020-07-13 17:13:03
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /baseline/process.py
'''
import json
import jieba
from data_utils import write_samples, read_samples, partition

samples = set()

with open('data/服饰_50k.json', 'r', encoding='utf8') as file:
    jsf = json.load(file)

for jsobj in jsf.values():
    title = jsobj['title'] + ' '
    kb = dict(jsobj['kb']).items()
    kb_merged = ''
    for key, val in kb:
        kb_merged += key+' '+val+' '

    ocr = ' '.join(list(jieba.cut(jsobj['ocr']))).replace('，', '')
    texts = []
    texts.append(title + kb_merged + ocr)
#     texts.append(title + ocr + kb_merged)
    reference = ' '.join(list(jieba.cut(jsobj['reference'])))
    for text in texts:
        sample = text+'<sep>'+reference
        samples.add(sample)

# replaced = read_samples('output/replaced.txt')
# for line in replaced:
#     samples.add(line.replace('｜', '<sep>'))
    
write_samples(samples, 'output/samples.txt')
partition(samples)
