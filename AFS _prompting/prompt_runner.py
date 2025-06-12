#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/23 10:27
@desc: 
"""
import json
import os
from pathlib import Path

from model import llm_infer
from utils.data_utils import Qid2Data
from utils.fancy_pbar import info_column, progress


class Runner:
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater
        # openai.api_key = __C.OPENAI_KEY

    def sample_make(self, ques, capt, know=None, cands=None, ans=None):
        line_prefix = self.__C.LINE_PREFIX
        # if know is not None:
        #     capt += " "+know
        prompt_text = line_prefix + f'Context: {capt}\n'
        if know is not None:
            prompt_text += line_prefix + f'Knowledge: {know}\n'
        prompt_text += line_prefix + f'Question: {ques}\n'
        if cands is not None:
            cands = cands[:self.__C.K_CANDIDATES]
            cands_with_conf = [f'{cand["answer"]}({cand["confidence"]:.2f})' for cand in cands]
            cands = ', '.join(cands_with_conf)
            prompt_text += line_prefix + f'Candidates: {cands}\n'
        prompt_text += line_prefix + 'Answer:'
        if ans is not None:
            prompt_text += f' {ans}'
        return prompt_text

    def get_context(self, example_qids):
        # making context text for one testing input
        prompt_text = self.__C.PROMPT_HEAD
        examples = []
        for key in example_qids:
            ques = self.trainset.get_question(key)
            caption = self.trainset.get_caption(key)
            know = self.trainset.get_knowledge(key)
            cands = self.trainset.get_topk_candidates(key)
            # all_ans = self.trainset.get_gt_answers(key)
            # if isinstance(all_ans, dict):
            #     all_ans = [{"answer": k, "confidence": v} for k, v in all_ans.items()]
            # all_ans.extend(cands)
            gt_ans = self.trainset.get_most_answer(key)
            examples.append((ques, caption, cands, gt_ans))
            prompt_text += self.sample_make(ques, caption, know=None, cands=cands, ans=gt_ans)
            prompt_text += '\n\n'
        return prompt_text

    def run(self):
        # where logs will be saved
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        # where results will be saved
        Path(self.__C.RESULT_DIR).mkdir(parents=True, exist_ok=True)

        self.cache = {}
        self.cache_file_path = os.path.join(
            self.__C.RESULT_DIR,
            'cache.json'
        )
        if self.__C.RESUME:
            self.cache = json.load(open(self.cache_file_path, 'r'))

        print('Note that the accuracies printed before final evaluation (the last printed one) are rough, '
              'just for checking if the process is normal!!!\n')
        self.trainset = Qid2Data(
            self.__C,
            self.__C.TRAIN_SPLITS,
            True
        )
        self.valset = Qid2Data(
            self.__C,
            self.__C.EVAL_SPLITS,
            self.__C.EVAL_NOW,
            json.load(open(self.__C.EXAMPLES_PATH, 'r'))
        )

        infer_times = self.__C.T_INFER
        N_inctx = self.__C.N_EXAMPLES

        print("begin")

        for qid in progress.track(self.valset.qid_to_data, description="Working...  "):
            # print(qid)
            if qid in self.cache:
                continue
            ques = self.valset.get_question(qid)
            caption = self.valset.get_caption(qid)
            know = self.valset.get_knowledge(qid)
            cands = self.valset.get_topk_candidates(qid, self.__C.K_CANDIDATES)

            prompt_query = self.sample_make(ques, caption, know=None, cands=cands)

            example_qids = self.valset.get_similar_qids(qid, k=infer_times * N_inctx)

            # random.shuffle(example_qids)

            prompt_info_list = []
            ans_pool = {}
            # multi-times infer
            for t in range(infer_times):
                # print(f'Infer {t}...')
                prompt_in_ctx = self.get_context(example_qids[(N_inctx * t):(N_inctx * t + N_inctx)])
                prompt_text = prompt_in_ctx + prompt_query
                gen_text, gen_prob = llm_infer(prompt_text)

                ans = self.evaluater.prep_ans(gen_text)
                if ans != '':
                    ans_pool[ans] = ans_pool.get(ans, 0.) + gen_prob

                prompt_info = {
                    'prompt': prompt_text,
                    'answer': gen_text,
                    'confidence': gen_prob
                }
                prompt_info_list.append(prompt_info)
                # time.sleep(self.__C.SLEEP_PER_INFER)

            # vote
            if len(ans_pool) == 0:
                answer = self.valset.get_topk_candidates(qid, 1)[0]['answer']
            else:
                answer = sorted(ans_pool.items(), key=lambda x: x[1], reverse=True)[0][0]

            self.evaluater.add(qid, answer)
            self.cache[qid] = {
                'question_id': qid,
                'answer': answer,
                'prompt_info': prompt_info_list
            }
            json.dump(self.cache, open(self.cache_file_path, 'w'))

            ll = len(self.cache)
            if self.__C.EVAL_NOW and not self.__C.DEBUG:
                if ll > 11 and ll % 10 == 0:
                    rt_accuracy = self.valset.rt_evaluate(self.cache.values())
                    info_column.info = f'Acc: {rt_accuracy}'

        self.evaluater.save(self.__C.RESULT_PATH)
        if self.__C.EVAL_NOW:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                self.evaluater.evaluate(logfile)