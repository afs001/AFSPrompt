#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/27 10:15
@desc: 
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

host="127.0.0.1"
port="11434" #默认的端口号为11434
llm=OllamaLLM(base_url=f"http://{host}:{port}", model="xxxx",temperature=0.1, num_predict=10, stop=["\n"])

template = "{question}"

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm

def out_parse(text):
    b_index = text.find(":")
    if b_index > -1:
        text = text[b_index:]
    my_index = [text.find('\n'), text.rfind('('), text.rfind(',')]
    is_all_one = all(x==-1 for x in my_index)
    if is_all_one:
        return text.strip()

    else:
        index = min(x for x in my_index if x>-1)
        return text[:index].strip()

def ollama_infer(entry):
    res = chain.invoke({"question": entry})
    # print(res)
    res_parse = out_parse(res)
    return res_parse, 1.0
