#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/29 15:26
@desc: 
"""
import json

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from AFS_rerank_system.AFScore.ClusterExamples import afsSimilarity
from AFS_rerank_system.AFScore.util import rerank


class AFSReranker:
    def __init__(self, inputs, context, yaml_path:str="configs/afs.yaml"):
        with open(yaml_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.inputs = inputs
        self.context = self.transform(context)

        self.outputs = self.afs4reranker()

    def transform(self, demos):
        ids = demos.get("ids", [[]])[0]
        questions = demos.get("documents", [[]])[0]
        metadatas = demos.get("metadatas", [[]])[0]

        captions = [m.get("caption") for m in metadatas]
        pre_anes = [m.get("ans") for m in metadatas]
        return {
            "ids": ids,
            "question": questions,
            "caption": captions,
            "answers": pre_anes,
        }

    def scoring_function(self):
        in_texts = [self.inputs["question"], self.inputs["caption"]]
        co_texts = [self.context["question"], self.context["caption"]]
        txt_scores = self.encode_txt2cos(in_texts, co_texts)

        in_lists = [[ans_map["answer"] for ans_map in self.inputs["answers"]]]
        co_lists = [[[ans_map["answer"] for ans_map in json.loads(ans_str)] for ans_str in self.context["answers"]]]
        lst_scores = self.calculate_list2num(in_lists, co_lists)

        txt_scores.extend(lst_scores)

        comb_scores = np.array(txt_scores)
        return comb_scores.T

    def afs4reranker(self):
        scores = self.scoring_function()
        x = self.get_concept_str(scores)
        examples_id = self.context["ids"]

        similarity_matrix = afsSimilarity(scores, x, epsilon=0.3)
        max_index = 0
        visited_nodes = [False] * len(similarity_matrix)
        new_rerank = rerank(similarity_matrix, max_index, visited_nodes, [], examples_id)

        return new_rerank

    def init_sim_model(self):
        return SentenceTransformer(self.cfg["SIM_MODEL"]).to('cuda')

    def encode_txt2cos(self, in_txts, co_txts):
        """
        批量计算多组输入文本与上下文文本的嵌入余弦相似度
        :param in_txts: 输入文本组（List[List[str]]，如[[问题, ..., 标签], ...]）
        :param co_txts: 上下文文本组（List[List[List[str]]]，如[[[问题1,...], [标签1,...]], ...]）
        :return: List[List[float]]，每组的相似度分数列表
        """
        model = self.init_sim_model()
        from sentence_transformers import util

        all_texts = []
        idx_map = []
        for i, (in_txt, co_txt) in enumerate(zip(in_txts, co_txts)):
            s = len(all_texts)
            texts = [in_txt] + co_txt
            all_texts.extend(texts)
            n = len(all_texts)
            idx_map.append((s, n))

        embeddings = model.encode(all_texts, batch_size=len(all_texts))
        scores_list = []
        for start, n in idx_map:
            in_emb_q = embeddings[start]
            co_embs_q = embeddings[start + 1 : n]
            scores_q = util.cos_sim(np.expand_dims(in_emb_q, axis=0), co_embs_q).squeeze(0).tolist()
            scores_list.append(scores_q)
        return scores_list

    def calculate_list2num(self, in_lsts, co_lsts):
        """
        批量计算多组输入特征与上下文列表型特征的相似度（交集元素个数）
        :param in_lsts: List[List[str]]，每组输入特征
        :param co_lsts: List[List[List[str]]]，每组对应的上下文特征列表
        :return: List[List[int]]，每组的相似度分数列表
        """
        return [
            [len(set(in_lst) & set(co)) for co in co_lst]
            for in_lst, co_lst in zip(in_lsts, co_lsts)
        ]

    def get_concept_str(self, scores):
        min_scores = np.min(scores, axis=0)
        mean_scores = np.mean(scores, axis=0)
        max_scores = np.max(scores, axis=0)
        # print("每列最小值:", min_scores)
        # print("每列均值:", mean_scores)
        # print("每列最大值:", max_scores)
        return (f"1 0 {min_scores[0]}\n3 0 {mean_scores[0]}\n5 0 {max_scores[0]}\n"
             f"7 1 {min_scores[1]}\n9 1 {mean_scores[1]}\n11 1 {max_scores[1]}\n"
             f"13 2 {min_scores[2]}\n15 2 {mean_scores[2]}\n17 2 {max_scores[2]}\n")
