#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx, jby
@Date: 2020-07-13 11:00:51
@LastEditTime: 2020-07-15 17:56:08
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /JD_project_2/baseline/model/predict.py
'''

import config
import torch
from dataset import MyDataset
from model import Seq2seq
from utils import source2ids, outputids2words, Beam, timer, add2heap
import jieba


class Predict():
    @timer(module='initalize predicter')
    def __init__(self):
        self.DEVICE = torch.device("cuda" if config.is_cuda else "cpu")
        dataset = MyDataset(config.data_path,
                            max_src_len=config.max_src_len,
                            max_tgt_len=config.max_tgt_len,
                            truncate_src=config.truncate_src,
                            truncate_tgt=config.truncate_tgt)

        self.vocab = dataset.build_vocab(embed_file=config.embed_file)

        self.model = Seq2seq(self.vocab)
        self.stop_word = list(
            set([
                self.vocab[x.strip()] for x in
                open(config.stop_word_file
                     ).readlines()
            ]))
        self.model.load_model()
        self.model.to(self.DEVICE)

    def greedy_search(self,
                      encoder_input,
                      max_sum_len,
                      max_oovs,
                      x_padding_masks):
        """Function which returns a summary by always picking
           the highest probability option conditioned on the previous word.

        Args:
            encoder_input (Tensor): Input sequence as the source.
            max_sum_len (int): The maximum length a summary can have.
            max_oovs (int): Number of out-of-vocabulary tokens.

        Returns:
            summary (list): The token list of the result summary.
        """

        # run body_sequence through encoder

        encoder_output, encoder_states = self.model.encoder(encoder_input)

        # initialize decoder with encoder forward state

        decoder_states = self.model.reduce_state(encoder_states)

        # initialize with start token
        decoder_input_t = torch.ones(1) * self.vocab.SOS
        decoder_input_t = decoder_input_t.to(self.DEVICE, dtype=torch.int64)

        summary = [self.vocab.SOS]

        while int(decoder_input_t.tolist()[0]) != (self.vocab.EOS) \
                and len(summary) < max_sum_len:
            # as long as decoder input is different from end token: continue
            context_vector, attention_weights = \
                self.model.attention(decoder_states,
                                     encoder_output,
                                     x_padding_masks)
            p_vocab, decoder_states = \
                self.model.decoder(decoder_input_t.unsqueeze(1),
                                   decoder_states,
                                   encoder_output,
                                   context_vector)

            decoder_input_t = torch.argmax(p_vocab, dim=1).to(self.DEVICE)
            decoder_word_idx = int(decoder_input_t.tolist()[0])
            summary.append(decoder_word_idx)

            oov_token = torch.full(decoder_input_t.shape, self.vocab.UNK)
            oov_token = oov_token.long().to(self.DEVICE)
            decoder_input_t = \
                torch.where(decoder_input_t > len(self.vocab) - 1,
                            oov_token,
                            decoder_input_t).to(self.DEVICE)

        return summary

    def best_k(self, beam, k, encoder_output, x_padding_masks):
        """Get best k tokens to extend the current sequence at the current time step.

        Args:
            beam (untils.Beam): The candidate beam to be extended.
            k (int): Beam size.
            encoder_output (Tensor): The lstm output from the encoder.
            x_padding_masks (Tensor):
                The padding masks for the input sequences.

        Returns:
            best_k (list(Beam)): The list of best k candidates.

        """
        # use decoder to generate vocab distribution for the next token
        decoder_input_t = torch.tensor(beam.tokens[-1]).reshape(1, 1)
        decoder_input_t = decoder_input_t.to(self.DEVICE)
        oov_token = torch.full(decoder_input_t.shape,
                               self.vocab.UNK).long().to(self.DEVICE)
        decoder_input_t = torch.where(decoder_input_t > len(self.vocab) - 1,
                                      oov_token,
                                      decoder_input_t)
        context_vector, attention_weights = \
            self.model.attention(beam.decoder_states,
                                 encoder_output,
                                 x_padding_masks)

        p_vocab, decoder_states = self.model.decoder(decoder_input_t,
                                                     beam.decoder_states,
                                                     encoder_output,
                                                     context_vector)

        # modify the result from seq2seq
        log_probs = torch.log(p_vocab.squeeze())

        if len(beam.tokens) == 1:
            forbidden_ids = [
                self.vocab[u"这"],
                self.vocab[u"此"],
                self.vocab[u"采用"],
                self.vocab[u"，"],
                self.vocab[u"。"],
                self.vocab.UNK
            ]
            log_probs[forbidden_ids] = -float('inf')

        topk_probs, topk_idx = torch.topk(log_probs, k)

        # get the best hypothesis
        best_k = [beam.extend(x,
                  log_probs[x],
                  decoder_states,
                  attention_weights,
                  beam.max_oovs,
                  beam.encoder_input) for x in topk_idx.tolist()]

        return best_k

    def beam_search(self,
                    encoder_input,
                    max_sum_len,
                    beam_width,
                    max_oovs,
                    x_padding_masks):
        """Using beam search to generate summary.

        Args:
            encoder_input (Tensor): Input sequence as the source.
            max_sum_len (int): The maximum length a summary can have.
            beam_width (int): Beam size.
            max_oovs (int): Number of out-of-vocabulary tokens.
            x_padding_masks (Tensor):
                The padding masks for the input sequences.

        Returns:
            result (list(Beam)): The list of best k candidates.
        """
        # run body_sequence input through encoder
        encoder_output, encoder_states = self.model.encoder(encoder_input)

        # initialize decoder states with encoder forward states
        decoder_states = self.model.reduce_state(encoder_states)

        # initialize the hypothesis with a class Beam instance.
        attention_weights = torch.zeros(
            (1, encoder_input.shape[1])).to(self.DEVICE)

        init_beam = Beam([self.vocab.SOS],
                         [0],
                         decoder_states,
                         attention_weights,
                         max_oovs,
                         encoder_input)

        # get the beam size and create a list for stroing current candidates
        # and a list for completed hypothesis
        k = beam_width
        curr, completed = [init_beam], []

        # use beam search for max_sum_len (maximum length) steps
        for _ in range(max_sum_len):
            # get k best hypothesis when adding a new token

            topk = []
            for beam in curr:
                if beam.tokens[-1] == self.vocab.EOS:
                    completed.append(beam)
                    k -= 1
                    continue
                for can in self.best_k(beam,
                                       k,
                                       encoder_output,
                                       x_padding_masks):
                    # Using topk as a heap to keep track of top k candidates.
                    # Using object ids to break ties.
                    add2heap(topk, (can.seq_score(), id(can), can), k)

            curr = [items[2] for items in topk]
            # stop when there are enough completed hypothesis
            if len(completed) == k:
                break
        # When there are not engouh completed hypotheses,
        # take whatever when have in current best k as the final candidates.

        completed += curr
        # sort the hypothesis by normalized probability and choose the best one
        result = sorted(completed,
                        key=lambda x: x.seq_score(),
                        reverse=True)[0].tokens
        return result

    @timer(module='doing prediction')
    def predict(self, text, tokenize=True, beam_search=True):
        """Generate summary.

        Args:
            text (str or list): Source.
            tokenize (bool, optional):
                Whether to do tokenize or not. Defaults to True.
            beam_search (bool, optional):
                Whether to use beam search or not.
                Defaults to True (means using greedy search).

        Returns:
            str: The final summary.
        """
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))
        x, oov = source2ids(text, self.vocab)
        x = torch.tensor(x).to(self.DEVICE)
        max_oovs = len(oov)
        oov_token = torch.full(x.shape, self.vocab.UNK).long().to(self.DEVICE)
        x_copy = torch.where(x > len(self.vocab) - 1, oov_token, x)
        x_copy = x_copy.unsqueeze(0)
        x_padding_masks = torch.ne(x_copy, 0).byte().float()
        if beam_search:
            summary = self.beam_search(x_copy,
                                       max_sum_len=config.max_dec_steps,
                                       beam_width=config.beam_size,
                                       max_oovs=max_oovs,
                                       x_padding_masks=x_padding_masks)
        else:
            summary = self.greedy_search(x_copy,
                                         max_sum_len=config.max_dec_steps,
                                         max_oovs=max_oovs,
                                         x_padding_masks=x_padding_masks)
        summary = outputids2words(summary,
                                  oov,
                                  self.vocab)
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()


if __name__ == "__main__":
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))
    prediction = pred.predict("衣香丽 影年 冬装 新款 韩版 宽松 连帽 大 毛领 羽绒服 女中 长款 外套 时尚 温室 玫瑰 水粉 填充物 白鸭绒 适用年龄 25-29周岁 充绒量 100g（含）-150g（不含） 衣领材质 狐狸毛 流行元素 口袋 穿着方式 常规 版型 宽松型 材质 聚对苯二甲酸乙二酯(涤纶) 袖型 插肩袖 图案 纯色 衣长 中长款 衣门襟 拉链 领型 圆领 上市时间 2019年秋季 袖长 长袖 含绒量 90%以上 厚度 加厚 面料 其它 风格 韩版 白 鸭绒  面料 解读  细节 展示  如何 与 一件 羽绒服 天长地久 ?  穿 搭 小心 机  目 基本 信息  织物 平整 细密  绸面 光滑  手感 柔软  轻薄 而 坚牢 耐磨  色泽 鲜艳  易洗 快干   商品 指数  长度 指数 / 中 长  口 颜色 / 温室 玫瑰 水粉 、 榄 仁  搭配 推荐  版型 指数 / 宽松  厚度 指数 / 厚  弹力 指数 / 无弹  忌 拧干  蓬松 保暖 / /  口 配饰 / 无  中性 洗涤剂  口 专柜 价 / 3669  轻柔 手洗  干透 后 轻拍 使 羽绒服 恢复 蓬松 柔软  自然 晾干 轻拍  洗好 后 将 水分 挤出 ( 忌 拧干 )  / / 优雅 纤长  蓬松 柔软  保暖 轻薄  切忌 干洗 和 机洗  ( 漂洗 ) 为宜  可 使 漂洗  忌 烘干  请 在 通风处 自然 烘干  呼吸 保暖  若 使用 洗衣粉  浓度 不要 太 2O  禁止 暴晒 和 慰 烫  天然 纯净  检验 : 2  填充物 1 : 白 鸭绒  中性".split(),  beam_search=True)

    print('hypo: ', prediction)
    print('ref: 选用 蓬松 的 白 鸭绒 作为 填充 ， 使 其 不仅 更加 保暖 ， 而且 穿着 时 不会 有 拖沓 感 ； 再 加上 光滑 的 羽绒 料 ， 穿 起来 更加 耐磨 ； 而且 采用 连帽 大 毛领 设计 ， 能 有效 防止 寒风 侵袭 ， 同时 还 彰显 出 温雅 娴熟 的 气质 。')