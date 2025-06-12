#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/22 15:58
@desc: 
"""
import json
import os

import chromadb
import yaml
from tqdm import tqdm

from vector_store.utils import data_loader
from vqa.embedder import get_joint_embedding

BATCH_SIZE = 5461

client = chromadb.PersistentClient(path="vector_store/chroma_data")

def db_init(path_map, task="okvqa"):
    questions = data_loader(path_map["txt_path"])
    captions = data_loader(path_map["cap_path"])
    tags = data_loader(path_map["tag_path"])
    answers = data_loader(path_map["ans_path"])

    try:
        client.delete_collection(name=task)
    except chromadb.errors.NotFoundError:
        pass
    collection = client.create_collection(name=task)

    documents = []
    embeddings = []
    metadatas = []
    # 假设questions为list，每个元素为dict，包含"question"字段
    for qid, data in tqdm(questions.items()):
        iid = data["image_id"]
        file_name = f"COCO_train2014_{iid:0>12}.jpg"
        image_path = os.path.join(path_map["img_path"], file_name)
        question = data["question"]
        gt_ans = max(set(data["multi_answers"]), key=data["multi_answers"].count)
        embedding = get_joint_embedding(image_path, question)

        documents.append(question)
        embeddings.append(embedding)
        answer_json = json.dumps(answers.get(str(qid), []), ensure_ascii=False)
        metadatas.append({
            "caption": captions.get(str(qid), ""),
            "tag": tags.get(str(iid), ""),
            "ans": answer_json,
            "gt_ans": gt_ans
        })

    ids = [str(i) for i in range(len(documents))]
    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end]
        )



if __name__ == '__main__':
    task = "okvqa"
    with open("configs/db.yaml", 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    txt_path = configs[task]["txt_path"]
    img_path = configs[task]["img_path"]
    cap_path = configs[task]["cap_path"]
    tag_path = configs[task]["tag_path"]
    ans_path = configs[task]["ans_path"]

    path_map = {
        "txt_path": txt_path,
        "img_path": img_path,
        "cap_path": cap_path,
        "tag_path": tag_path,
        "ans_path": ans_path,
    }
    db_init(path_map, task)
