#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/23 10:25
@desc: 
"""
import argparse

import yaml

from configs.task_cfgs import Cfgs
from evaluation.aokvqa_evaluate import AOKEvaluater
from evaluation.okvqa_evaluate import OKEvaluater
from .prompt_runner import Runner


def prompt_login_args(parser):
    parser.add_argument('--debug', dest='DEBUG', help='debug mode', action='store_true')
    parser.add_argument('--resume', dest='RESUME', help='resume previous run', action='store_true')
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)

    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, default='configs/prompt.yml')
    parser.add_argument('--examples_path', dest='EXAMPLES_PATH',
                        help='answer-aware example file path',
                        type=str,
                        default="assets/aok/val/avg_a_ok_val_examples.json")
    parser.add_argument('--candidates_path', dest='CANDIDATES_PATH',
                        help='candidates file path', type=str,
                        default="assets/candidates_aokvqa_val.json")
    parser.add_argument('--captions_path', dest='CAPTIONS_PATH',
                        help='captions file path', type=str,
                        default="assets/captions_aokvqa.json")
    parser.add_argument('--tags_path', dest='TAGS_PATH',
                        help='tags file path', type=str,
                        default="assets/tags_okvqa.json")
    parser.add_argument('--knowledge_path', dest='KNOWLEDGE_PATH',
                        help='knowledge file path', type=str,
                        default="assets/gpt3_okvqa.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heuristics-enhanced Prompting')
    prompt_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)

    # build runner
    if 'aok' in __C.TASK:
        evaluater = AOKEvaluater(
            __C.EVAL_ANSWER_PATH,
            __C.EVAL_QUESTION_PATH,
        )
    else:
        evaluater = OKEvaluater(
            __C.EVAL_ANSWER_PATH,
            __C.EVAL_QUESTION_PATH,
        )

    runner = Runner(__C, evaluater)
    runner.run()
