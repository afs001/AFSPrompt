#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/22 16:02
@desc: 
"""
import chromadb

from vqa.embedder import get_joint_embedding

client = chromadb.PersistentClient(path="vector_store/chroma_data")

def get_topk_demo(image, question, task="okvqa", topk=5):
    query_embedding = get_joint_embedding(image, question)
    existing_collection = client.get_collection(name=task)
    demos = existing_collection.query(query_embedding, n_results=topk)
    return demos

import random

def get_random_10_samples(task="okvqa"):
    existing_collection = client.get_collection(name=task)
    all_items = existing_collection.get(include=["metadatas", "documents"])
    total = len(all_items["documents"])
    if total <= 10:
        indices = list(range(total))
    else:
        indices = random.sample(range(total), 10)
    samples = []
    for idx in indices:
        doc = all_items["documents"][idx]
        meta = all_items["metadatas"][idx]
        # 假设meta包含图像、问题、字幕、标签、答案等字段
        samples.append([
            idx,
            doc,
            meta.get("caption", ""),
            meta.get("tag", ""),
            meta.get("ans", "")
        ])
    return samples