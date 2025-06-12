#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/27 10:36
@desc: 
"""
from vector_store.img_processing.answer import generate_pre_answer
from vector_store.img_processing.captioner import get_img_caption


def feature_process(image, question):
    caption = get_img_caption(question, image)
    answers = generate_pre_answer(question, image)
    return {
        "question": question,
        "caption": caption,
        "answers": answers,
    }