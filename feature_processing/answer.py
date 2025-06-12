#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/26 9:40
@desc: 
"""
import json

import clip
import numpy as np
import torch
import yaml
from torch import nn
from transformers import AutoTokenizer

from feature_processing.mcan.mcan_for_finetune import MCANForFinetune
from feature_processing.mcan.net_utils import _transform

with open("configs/mcan.yaml", 'r') as f:
    __C = yaml.safe_load(f)
# print('Loading ckpt {}'.format(path))
# net = MCANForFinetune(__C)
# ckpt = torch.load(path, map_location='cpu')
# net.load_state_dict(ckpt['state_dict'], strict=False)
# net.cuda()
# print('Finish!')
class ExtractModel:
    def __init__(self, encoder) -> None:
        encoder.attnpool = nn.Identity()
        self.backbone = encoder

        self.backbone.cuda().eval()

    @torch.no_grad()
    def __call__(self, img):
        x = self.backbone(img)
        return x

class MCANForVQA:
    def __init__(self, __C):
        self.__C = __C

        # ans_dict_path = "asserts/okvqa/answer_dict_okvqa.json"
        self.ix_to_ans = json.load(open(__C["ANS_DICT_PATH"], 'r', encoding='utf-8'))
        ans_to_ix = {ans: ix for ix, ans in enumerate(self.ix_to_ans)}
        ans_size = len(ans_to_ix)
        print('Answer size: {}'.format(ans_size))

        # load bert tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(__C["BERT_VERSION"])
        self.token_size = self.tokenizer.vocab_size
        print(f'== BertTokenizer loaded, vocab size: {self.token_size}')

        # load visual extract
        self.T = _transform(__C["IMG_RESOLUTION"])
        clip_model, _ = clip.load(__C["CLIP_VERSION"], device='cpu', download_root="models/clip_RN")
        img_encoder = clip_model.visual
        self.net = ExtractModel(img_encoder)

        # load mcan model
        self.mcan = MCANForFinetune(__C, ans_size)
        ckpt = torch.load(__C["MCAN_PATH"], map_location='cpu')
        self.mcan.load_state_dict(ckpt['state_dict'], strict=False)
        self.mcan.cuda()
        print('Finish!')

    def vqa(self, question, image, k=10):
        img_feat = self.get_image_feats(image)
        assert img_feat.shape == (self.__C["IMG_FEAT_GRID"], self.__C["IMG_FEAT_GRID"], self.__C["IMG_FEAT_SIZE"])
        img_feat = img_feat.reshape(-1, self.__C["IMG_FEAT_SIZE"])

        ques_ids = self.get_text_feats(question)
        x = (torch.tensor(img_feat, dtype=torch.float).cuda(), torch.tensor(ques_ids, dtype=torch.long).cuda())
        pred, answer_latents = self.mcan(x, output_answer_latent=True)
        pred_np = pred.sigmoid().cpu().detach().numpy()
        # answer_latents_np = answer_latents.cpu().detach().numpy()
        # for i in range(len(pred_np)):
        ans_np = pred_np[0]
        ans_idx = np.argsort(-ans_np)[:k]
        ans_item = []
        for idx in ans_idx:
            ans_item.append(
                {
                    'answer': self.ix_to_ans[idx],
                    'confidence': float(ans_np[idx])
                }
            )
        return ans_item

    def get_text_feats(self, question):
        ques_ids = self.bert_tokenize(question)
        return ques_ids

    def bert_tokenize(self, text):
        max_token = self.__C['MAX_TOKEN']
        text = text.lower().replace('?', '')
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_token - 2:
            tokens = tokens[:max_token-2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids + [0] * (max_token - len(ids))
        ids = np.array(ids, np.int64)
        return ids

    def get_image_feats(self, img):
        img = self.T(img).unsqueeze(0).cuda()
        clip_feats = self.net(img).cpu().numpy()[0]
        clip_feats = clip_feats.transpose(1, 2, 0)
        return clip_feats


mcan = MCANForVQA(__C)
def generate_pre_answer(question, image):
    return mcan.vqa(question, image)


