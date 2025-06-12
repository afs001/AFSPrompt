#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/29 10:32
@desc: 
"""
import pandas as pd


def context_2_view(context):
    ids = context.get("ids", [[]])[0]
    questions = context.get("documents", [[]])[0]
    metadatas = context.get("metadatas", [[]])[0]

    captions = [m.get("caption") for m in metadatas]
    pre_anes = [m.get("ans") for m in metadatas]
    gt_anes = [m.get("gt_ans") for m in metadatas]

    assert len(questions) == len(captions) == len(pre_anes) == len(gt_anes) == 10, \
        f"{len(questions), len(captions), len(pre_anes), len(gt_anes)}"

    data = {
        "id": ids,
        "问题": questions,
        "字幕": captions,
        "候选答案": pre_anes,
        "答案": gt_anes
    }
    return pd.DataFrame(data)