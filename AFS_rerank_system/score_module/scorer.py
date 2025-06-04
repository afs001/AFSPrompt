#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/12/23 14:49
"""
import argparse
import json
import os.path
import time
import yaml

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data.AFSasserts.score_module.features import get_text_score, list_scores_process
from data.AFSasserts.score_module.utils import load_data, load_dict

model_dir = "data/AFSasserts/simmodel/mpnetdot"
model = SentenceTransformer(model_dir).to('cuda')

def encode_text_feature(texts):
    vecs = model.encode(texts)
    return vecs[0,:], vecs[1,:], vecs[2,:]

def load_yaml_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def add_arguments_from_config(parser, config):
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=str, default=value)
    return parser

def main(args):
    context_data = load_data(args.context_path, args.task)
    context_map = {str(t["question_id"]): t for t in context_data}

    if hasattr(args, 'context2_path'):
        context2_data = load_data(args.context2_path)
        context2_map = {t["question_id"]: t for t in context2_data}
        context_map = context_map | context2_map

    val_data = load_data(args.val_path, args.task)
    captions_data = load_dict(args.captions_path)

    if os.path.exists(args.tags_path):
        tags_data = load_dict(args.tags_path)

    if hasattr(args, 'candidates_path'):
        candidates_data = load_dict(args.candidates_path)
    else:
        print("候选示例映射不存在。。。")

    examples_map = load_dict(args.eid_path)

    if os.path.exists(args.score_path):
        examples_scores = load_dict(args.score_path)
    else:
        examples_scores= {}
    for i, val in enumerate(tqdm(val_data)):
        qid = str(val['question_id'])
        iid = str(val["image_id"])

        if qid in examples_scores:
            continue

        e_ids = examples_map[qid][:100]

        question = val["question"]
        caption = captions_data[qid]
        tag = tags_data[qid]
        if hasattr(args, 'candidates_path'):
            cands = candidates_data[qid]

        v_q, v_c, v_t = encode_text_feature([question, caption, tag])

        example_data = []
        for eid in e_ids:
            example = context_map[eid]
            eiid = str(example["image_id"])
            v_eq, v_ec, v_et = encode_text_feature([example["question"], captions_data[eid], tags_data[eid]])

            S_1 = [get_text_score(v_q, v_eq), get_text_score(v_c, v_ec), get_text_score(v_t, v_et)]

            # list_inputs = [
            #     [a["answer"] for a in cands],
            # ]
            # S_2 = list_scores_process(list_inputs)
            # S_1.extend(S_2)
            example_data.append(S_1)
        examples_scores.update({qid: example_data})
        json.dump(examples_scores, open(args.score_path, 'w'))

    json.dump(examples_scores, open(args.score_path, 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score login')
    parser.add_argument("--task", choices=["OKVQA", "FVQA", "AOKVQA"])
    args = parser.parse_args()

    config_path = 'data/AFSasserts/score_module/path_config.yaml'
    config = load_yaml_config(config_path)[args.task]

    parser = add_arguments_from_config(parser, config)
    args = parser.parse_args()
    main(args)