#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/12/25 21:35
"""
import argparse
import json
import os
import re

import clip
import numpy as np
import torch
from PIL import Image
import yaml  # 导入 PyYAML 库
from tqdm import tqdm

from candiate_examples_clip.clip_src.feature_extraction import extract_image_features, extract_text_features
from candiate_examples_clip.clip_src.selection import select_in_context_examples
from candiate_examples_clip.clip_src.utils import load_features, load_vqav2


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_questions_from_file(file_path, task):
    if task == "vqav2":
        questions = load_vqav2(file_path)
    else:
        with open(file_path, 'r',  encoding='utf-8') as file:
            questions = json.load(file)
    return questions

def load_images_from_folder(file_path, act=True):
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    if act:
        return list(data.values())
    else:
        return data

def extract_img_and_save_features(image_examples, model, preprocess, device, img_path, save_path):
    print("Extracting image features...")
    for name in tqdm(image_examples):
        full_img_file = img_path+name
        full_save_file = save_path + re.sub(r'\.(jpg|JPEG)', '.npz', name, flags=re.IGNORECASE)
        if os.path.exists(full_save_file):
            continue
        extract_image_features(full_img_file, model, preprocess, device, full_save_file)

def extract_text_and_save_features(questions, model, device, save_path):
    print("Extracting text features...")
    for question in tqdm(questions):
        name = str(question["question_id"])
        full_save_file = save_path+f'{name}.npz'
        if os.path.exists(full_save_file):
            continue
        extract_text_features(question["question"], model, device, full_save_file)

def main(args):
    config = load_config(args.config)
    device = args.device
    print(f"Using device: {device}")
    action = args.action

    model, preprocess = clip.load(args.model_name, device, download_root=args.model_root)

    if action == "extract_features":
        task_paths = config['TASK_PATHS'][f"{args.task}_{args.mode}"]

        image_examples = load_images_from_folder(task_paths['IID2IMAGE_NAME'])
        print(f"Loaded {len(image_examples)} images.")
        extract_img_and_save_features(image_examples, model, preprocess, device,
                                      task_paths['IMAGES_PATH'],
                                      task_paths['IMAGES_FEATS_PATH'])
        if args.mode=="train":
            example_questions = load_questions_from_file(task_paths['QUESTIONS_FILE'], args.task)
            print(f"Loaded {len(example_questions)} questions.")
            extract_text_and_save_features(example_questions, model, device,task_paths['TEXTS_FEATS_PATH'])


    elif action == "select_context":
        train_task_paths = config['TASK_PATHS'][f"{args.task}_train"]
        val_task_paths = config['TASK_PATHS'][f"{args.task}_val"]

        input_questions = load_questions_from_file(val_task_paths['QUESTIONS_FILE'], args.task)
        print(f"Input question: {len(input_questions)}")
        input_images = load_images_from_folder(val_task_paths['IID2IMAGE_NAME'], act=False)

        example_images = load_images_from_folder(train_task_paths['IID2IMAGE_NAME'], act=False)
        example_questions = load_questions_from_file(train_task_paths['QUESTIONS_FILE'], args.task)

        examples_qqid2eqid = {}
        for inq in tqdm(input_questions, desc="input:"):
            input_text_features = extract_text_features([inq["question"]], model, device)
            full_input_image_features = (val_task_paths['IMAGES_FEATS_PATH'] +
                                         re.sub(r'\.(jpg|JPEG)', '.npz', input_images[str(inq["image_id"])], flags=re.IGNORECASE))
            input_image_features = load_features(full_input_image_features)

            print("Selecting in-context examples...")
            top_n = args.top_n

            cached_top100_example = []

            for i in range(0, len(example_questions), args.batch_size):
                batch_questions = example_questions[i:i + args.batch_size]
                batch_question_ids = [str(q["question_id"]) for q in batch_questions]
                batch_image_ids = [example_images[str(q["image_id"])] for q in batch_questions]

                example_text_features = np.array([
                    load_features(train_task_paths['TEXTS_FEATS_PATH'] + f"{qid}.npz") for qid in
                    batch_question_ids])
                example_image_features = np.array([
                    load_features(train_task_paths['IMAGES_FEATS_PATH'] + re.sub(r'\.(jpg|JPEG)', '.npz', img_id, flags=re.IGNORECASE)) for img_id in
                    batch_image_ids])

                combined_similarities, top_n_indices = select_in_context_examples(input_text_features,
                                                                                  input_image_features,
                                                                                  example_text_features,
                                                                                  example_image_features, top_n=top_n)

                # Cache the top 100 examples
                cached_top100_example.extend(
                    [(combined_similarities[idx], batch_questions[idx]) for idx in top_n_indices])
                cached_top100_example = sorted(cached_top100_example, key=lambda x: x[0], reverse=True)[:100]

            print(f"Top {top_n} in-context examples:")
            examples_qqid2eqid.update({inq["question_id"]: [e["question_id"] for _, e in cached_top100_example]})
            with open(args.save_path, 'w', encoding='utf-8') as f:
                json.dump(examples_qqid2eqid, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control the execution of feature extraction or context selection.")
    parser.add_argument("--mode", choices=["train", "val"], required=True, help="Mode to distinguish paths")
    parser.add_argument("--task", choices=["vqav2", "fvqa", "textvqa"], required=True, help="Task to perform")
    parser.add_argument("--action", choices=["extract_features", "select_context"], help="Action to perform")
    parser.add_argument("--batch_size", type = int, default = 1000, help = "Batch size for processing the examples")
    parser.add_argument("--top_n", type=int, default=100, help="Number of top examples to select")
    parser.add_argument("--model_name", type=str, default="ViT-B/16", help="Name of the model to use")
    parser.add_argument("--model_root", type=str, default="candiate_examples_clip/clip_model", help="Root directory for the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--config", type=str, default="candiate_examples_clip/config.yaml", help="Path to the configuration file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output")
    args = parser.parse_args()
    main(args)