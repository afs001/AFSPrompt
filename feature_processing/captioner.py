#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/12/23 9:24
"""
import json
import os
from typing import Dict

import torch
from promptcap import PromptCap
from tqdm import tqdm

def ok_load(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_ok_cap(caps: Dict, cap_path: str):
    with open(cap_path, "w", encoding='utf-8') as f:
        json.dump(caps, f, ensure_ascii=False, indent=2)

def main(mode: str = 'train'):
    caps_path = f"okvqa/caption/captions_okvqa_{mode}.json"
    cap_dict = ok_load(caps_path) if os.path.exists(caps_path) else {}

    oks = ok_load(f"okvqa/okvqa_{mode}.json")
    folder_path = f"E:/2023/VQA/OFAALL/OFA_MAIN/data/{mode}2014/"

    model = PromptCap("tifa-benchmark/promptcap-coco-vqa")
    if torch.cuda.is_available():
        model.cuda()

    updated = False
    for qid, ok in tqdm(oks.items(), desc="Generating captions"):
        if qid in cap_dict:
            continue
        question = ok["question"]
        iid = ok["image_id"]
        file_name = f"COCO_{mode}2014_{iid:0>12}.jpg"
        prompt = f"please describe this image according to the given question: {question}"
        image = os.path.join(folder_path, file_name)
        try:
            caption = model.caption(prompt, image)
        except Exception as e:
            print(f"Error processing {qid}: {e}")
            continue
        cap_dict[qid] = caption
        updated = True
        save_ok_cap(cap_dict, caps_path)

    if updated:
        save_ok_cap(cap_dict, caps_path)

if __name__ == '__main__':
    main('val')