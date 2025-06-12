#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/22 16:13
@desc: 
"""
import torch
import clip
from PIL import Image

device='cuda'

model, preprocess = clip.load("ViT-B/16", "cuda", download_root="models/clip_vit")

def embed_image(image_path):
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path.convert("RGB")
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(img).cpu().numpy()[0]

def embed_text(text):
    text_inputs = clip.tokenize(text).to(device)
    with torch.no_grad():
        return model.encode_text(text_inputs).cpu().numpy()[0]

def get_joint_embedding(image_path, text):
    """
    获取图像和文本的联合嵌入（拼接特征向量）
    """
    image_emb = embed_image(image_path)
    text_emb = embed_text(text)
    return image_emb + text_emb