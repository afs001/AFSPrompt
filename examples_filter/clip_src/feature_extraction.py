#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/12/25 21:30
"""
from typing import List

import clip
import numpy as np
from pathlib import Path

import torch
from PIL import Image


@torch.no_grad()
def extract_image_features(image_path, model, preprocess, device, save_path):
    """
    Extract image features for a single image using CLIP's image encoder.
    """
    img = Image.open(image_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)
    image_features = model.encode_image(img).cpu().numpy()[0]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, x=image_features)

@torch.no_grad()
def extract_text_features(question, model, device, save_path=None):
    """
    Extract image features for a single image using CLIP's image encoder.
    """
    text_inputs = clip.tokenize(question).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs).cpu().numpy()[0]

    if save_path is None:
        return text_features
    else:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, x=text_features)


def extract_texts_features(questions: List[str], model, device, batch_size=32):
    if len(questions) > batch_size:
        text_features = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            text_inputs = clip.tokenize(batch).to(device)
            with torch.no_grad():
                batch_features = model.encode_text(text_inputs).cpu().numpy()
            text_features.append(batch_features)

        text_features = np.concatenate(text_features, axis=0)
        return text_features
    else:
        text_inputs = clip.tokenize(questions).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs).cpu().numpy()
        return text_features[0]

