#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2024/12/25 21:32
"""
import numpy as np

def select_in_context_examples(input_question_feats,
                               input_image_feats,
                               example_question_feats,
                               example_image_feats, top_n=5):
    """
    Select the top n in-context examples based on cosine similarity between input question and image features.
    Args:
        input_question_feats (ndarray): Features of the input question.
        input_image_feats (ndarray): Features of the input image.
        example_question_feats (ndarray): Features of all in-context questions.
        example_image_feats (ndarray): Features of all in-context images.
        top_n (int): Number of top examples to select.
    Returns:
        tuple: Combined similarities and list of int indices of the top n selected examples.
    """
    # Calculate the cosine similarity between input question and example questions
    question_similarities = np.dot(example_question_feats, input_question_feats) / (
            np.linalg.norm(example_question_feats, axis=1) * np.linalg.norm(input_question_feats))

    # Calculate the cosine similarity between input image and example images
    image_similarities = np.dot(example_image_feats, input_image_feats) / (
            np.linalg.norm(example_image_feats, axis=1) * np.linalg.norm(input_image_feats))

    # Combine the similarities
    combined_similarities = (question_similarities + image_similarities) / 2.0

    # Get the top n indices
    top_indices = np.argsort(combined_similarities)[-top_n:][::-1]

    return combined_similarities, top_indices.tolist()



