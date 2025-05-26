#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/1/2 10:11
@desc: 
"""
import os
import requests

API_URL = "https://westus.api.cognitive.microsoft.com/vision/v3.2/analyze?visualFeatures=Tags"
DEFAULT_SUBSCRIPTION_KEY = "xxxx"


def get_image_tags(img_path, subscription_key=DEFAULT_SUBSCRIPTION_KEY, url=API_URL):
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "language": "en",
        "model-version": "latest"
    }
    with open(img_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, params=params, headers=headers, files=files, timeout=10, verify=False)
    try:
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, response.status_code


def load_image_paths(directory, exts=('.png', '.jpg', '.jpeg')):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.lower().endswith(exts)
    ]


def main():
    image_directory = 'data/dataset/OKVQA/train2024'
    image_paths = load_image_paths(image_directory)

    for img_path in image_paths:
        tags, status_code = get_image_tags(img_path)
        if status_code == 200:
            print(f"Tags for {img_path}: {tags}")
        else:
            print(f"Failed to get tags for {img_path}, status code: {status_code}, error: {tags.get('error', tags)}")
        break


if __name__ == '__main__':
    main()