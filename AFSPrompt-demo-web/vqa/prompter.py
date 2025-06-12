#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/27 10:20
@desc: 
"""
import json

import yaml

with open("configs/prompt.yaml", "r", encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

def sample_make(ques, capt, cands=None, ans=None):
    line_prefix = cfg["LINE_PREFIX"]
    prompt_text = line_prefix + f'Context: {capt}\n'
    prompt_text += line_prefix + f'Question: {ques}\n'
    if cands is not None:
        cands = cands[:cfg["K_CANDIDATES"]]
        cands_with_conf = [f'{cand["answer"]}({cand["confidence"]:.2f})' for cand in cands]
        cands = ', '.join(cands_with_conf)
        prompt_text += line_prefix + f'Candidates: {cands}\n'
    prompt_text += line_prefix + 'Answer:'
    if ans is not None:
        prompt_text += f' {ans}'
    return prompt_text

def get_context(demos):
    # making context text for one testing input
    prompt_text = cfg["PROMPT_HEAD"]

    example_qids = demos.example_qids

    for key in example_qids:
        ques = demos.get_question(key)
        caption = demos.get_caption(key)
        cands = demos.get_topk_candidates(key)
        gt_ans = demos.get_most_answer(key)
        prompt_text += sample_make(ques, caption, cands=cands, ans=gt_ans)
        prompt_text += '\n\n'
    return prompt_text

class DemosSet:
    def __init__(self, demos, ids=None):
        if ids is None:
            self.example_qids = [i for i in range(cfg["K_CANDIDATES"])]
        else:
            eids  = demos.get("ids", [[]])[0]
            self.example_qids = [eids.index(id) for id in ids]
        self.questions = demos.get('documents')[0]
        self.metadatas = demos.get('metadatas')[0]

    def get_question(self, key):
        return self.questions[key]

    def get_caption(self, key):
        metadata = self.metadatas[key]
        return metadata["caption"]

    def get_topk_candidates(self, key):
        metadata = self.metadatas[key]
        pre_ans_str = metadata["ans"]
        return json.loads(pre_ans_str)

    def get_most_answer(self, key):
        metadata = self.metadatas[key]
        return metadata["gt_ans"]

def get_prompt(inputs, examples, ids=None):
    ques, caption, cands = inputs['question'], inputs['caption'], inputs['answers']
    prompt_query = sample_make(ques, caption, cands)
    demo_set = DemosSet(examples, ids=ids)
    prompt_in_ctx = get_context(demos=demo_set)
    prompt_text = prompt_in_ctx + prompt_query
    return prompt_text