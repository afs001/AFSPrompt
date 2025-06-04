#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/12/25 21:33
"""
import json
import pickle

import numpy as np
from numpy import ndarray


def load_features(filename)->ndarray:
    """Load features from a file."""
    with np.load(filename) as data:
        return data['x']

def load_vqav2(filename):
    """Load the VQA v2 dataset from a file."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
