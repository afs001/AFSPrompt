#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/5/22 15:21
@desc: 
"""
import gradio as gr

from app_server import vqa_model, db_init, get_samples

with gr.Blocks(title="一种基于公理模糊集上下文重排的视觉问答系统") as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>一种基于公理模糊集上下文重排的视觉问答系统</h1>",
        elem_id="main-title"
    )
    with gr.Tab("数据库初始化"):
        gr.Markdown("### 示例展示")
        task_input = gr.Radio(choices=["okvqa",], label="选择任务", value="okvqa")
        init_output = gr.Dataframe(headers=["id","问题", "字幕", "标签", "答案"], value=get_samples("okvqa"))
        task_input.change(
            fn=lambda task: get_samples(task) if task else None,
            inputs=task_input,
            outputs=init_output
        )

        # gr.Markdown("### 演示库输入")
        # with gr.Row():
        #     image_input = gr.File(label="图像存储路径文件（JSON文件）", file_types=[".json"], height=40)
        #     question_input = gr.File(label="VQA问题（JSON文件）", file_types=[".json"], height=40)
        #     caption_input = gr.File(label="VQA图像字幕（JSON文件）", file_types=[".json"], height=40)
        #     tag_input = gr.File(label="VQA图像标签（JSON文件）", file_types=[".json"], height=40)
        #     answer_input = gr.File(label="VQA答案（JSON文件）", file_types=[".json"], height=40)
        # init_btn = gr.Button("增加任务演示库")
        # init_btn.click(fn=db_init,
                       # inputs=[image_input, question_input, caption_input, tag_input, answer_input],
                       # outputs=init_output)

    with gr.Tab("视觉问答系统"):
        gr.Markdown("### 上传图像并输入问题")
        with gr.Row():
            image_input = gr.Image(type="pil", label="上传图像", value="img/demo.jpg")
            with gr.Column(scale=1, min_width=600):
                question_input = gr.Textbox(label="输入你的问题", value="这个人在干嘛？")
                # L代表检索示例个数，N代表单个提示使用多少示例，K代表推理次数
                L_input = gr.Slider(label="L", minimum=1, maximum=10, step=1, interactive=True, value=10)
                N_input = gr.Slider(label="N", minimum=1, maximum=5, step=1, interactive=True, value=5)
                K_input = gr.Slider(label="K", minimum=1, maximum=2, step=1, interactive=True, value=1)
        gr.Markdown("### 提示头：")
        gr.Markdown("#### Please answer the question according to the context and candidate answers. Each candidate answer is associated with a confidence score within a bracket. The true answer may not be included in the candidate answers. \n\n")
        # context_output1 = gr.Textbox(label="上下文信息")
        gr.Markdown("### 上下文信息：")
        context_output = gr.Dataframe(headers=["id", "问题", "字幕", "候选答案", "答案"])
        answer_output = gr.Textbox(label="模型回答")
        submit_btn = gr.Button("提交")
        submit_btn.click(
            fn=vqa_model,
            inputs=[image_input, question_input, L_input, N_input, K_input],
            outputs=[context_output, answer_output]
        )

if __name__ == "__main__":
    demo.launch()