#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/23 10:27
@desc: 
"""
from typing import List

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2023/8/23 14:55
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria, \
    add_start_docstrings
from transformers.generation.stopping_criteria import STOPPING_CRITERIA_INPUTS_DOCSTRING


class StopAtSpecificTokenCriteria(StoppingCriteria):
    """
    当生成出第一个指定token时，立即停止生成
    """

    def __init__(self, token_id_list: List[int] = None):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.token_id_list = token_id_list

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # return np.argmax(scores[-1].detach().cpu().numpy()) in self.token_id_list
        # 储存scores会额外占用资源，所以直接用input_ids进行判断
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list


# 词表中，英文\n的id是13
token_id_list = [13, 2, 0]
stopping_criteria = StoppingCriteriaList()
stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=token_id_list))

model_dir = r"/sdb/wang/premodels/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map='auto',
    # load_in_8bit=True,
    torch_dtype=torch.float16,
    # max_memory=max_memory,
    trust_remote_code=True
).to('cuda')
model = model.eval()


def llm_infer(prompt):
    model_inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    current_token_len = model_inputs.input_ids.shape[1]
    generate = model.generate(**model_inputs,
                              pad_token_id=pad_token_id,
                              temperature=0.1,
                              do_sample=True,
                              top_p=0.95,
                              top_k=40,
                              max_new_tokens=8,
                              stopping_criteria=stopping_criteria,
                              output_scores=True,
                              return_dict_in_generate=True,
                              return_legacy_cache=True
                              )

    generate_ids = generate["sequences"][0, current_token_len:].unsqueeze(0)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
        0].strip()

    plist = []
    for ii in range(len(generate_ids[0])):
        if generate_ids[0][ii] in token_id_list:
            break
        logits = torch.softmax(generate["scores"][ii], dim=1)
        plist.append(logits[0, generate_ids[0][ii]])
    if len(plist) < 1:
        prob = 0
    else:
        prob = torch.prod(torch.stack(plist)).item()
    # print(response)
    return response, prob


