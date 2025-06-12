#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/22 15:50
@desc: 
"""
import pandas as pd
from PIL import Image
import gradio as gr

from AFS_rerank_system.afs_reranker import AFSReranker
from vqa.llmer import ollama_infer
from vqa.processor import feature_process
from vqa.prompter import get_prompt
from vqa.retriever import get_topk_demo, get_random_10_samples
from vqa.utils import context_2_view


def get_samples(task):
    samples = get_random_10_samples(task=task)
    # 转换为DataFrame以便gr.Dataframe显示
    df = pd.DataFrame(samples, columns=["id", "问题", "字幕", "标签", "答案"])
    return df

def db_init(image_input, question_input, caption_input, tag_input, answer_input):
    # 这里调用你的数据库初始化逻辑
    # 例如：from vector_store.db_init import init_db; init_db()
    return "数据库初始化完成！"

# 检索模块（可从图像中提取上下文）
def retrieve_context(image: Image.Image, question: str, settings) -> str:
    context = get_topk_demo(image, question, task=settings["TASK"], topk=settings["Topk"])
    # 示例上下文生成（实际可用图像caption模型或OCR）
    return context


# VQA 模型函数
def vqa_model(image: Image.Image, question: str, L: int, N: int, K: int, progress=gr.Progress(), ) -> (str, str):
    progress(0.0, desc="开始...")
    # 加载yaml配置示例
    import yaml
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if L:
        config["retrieval"]["Topk"] = L
    if N:
        config["retrieval"]["Nsam"] = N
    if K:
        config["retrieval"]["Ktim"] = K

    progress(0.2, desc="配置加载...")

    inputs = feature_process(image, question)
    progress(0.4, desc="输入特征处理完成...")

    demos = retrieve_context(image, question, settings=config['retrieval'])
    ids = AFSReranker(inputs=inputs, context=demos).outputs

    view_context = context_2_view(demos)
    progress(0.6, desc="上下文检索完成...")

    prompt = get_prompt(inputs, demos, ids)

    progress(0.8, desc="提示词构建完成...")
    # 模拟回答生成
    answer, _ = ollama_infer(prompt)

    progress(1.0, desc="任务完成")

    return view_context, answer