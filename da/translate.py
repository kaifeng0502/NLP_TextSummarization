#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx
@Date: 2020-07-13 20:16:37
@LastEditTime: 2020-07-13 20:44:55
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /JD_project_2/da/translate.py
'''

from googletrans import Translator
import time
import jieba
from data_utils import read_samples, write_samples
translator = Translator()

res = []
samples = read_samples('output/samples.txt')

count = 0
for line in samples:
    count += 1
    if count % 100 == 0:
        print(count)
        write_samples(res, 'output/translated.txt', 'a')
        res = []
    source = str(line)
    try:
        translation = translator.translate(source, dest='ja')
        time.sleep(1)

        translation = translator.translate(translation.text, dest='zh-cn')
        res.append(' '.join(list(jieba.cut(translation.text))))

    except:
        continue
    time.sleep(1)
