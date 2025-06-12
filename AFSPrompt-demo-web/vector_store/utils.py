#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/26 10:11
@desc: 
"""
import json


def data_loader(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data