import logging
import os

import gradio as gr

from ChatWorld import ChatWorld

logging.basicConfig(level=logging.INFO, filename="demo.log", filemode="w",
                    format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

chatWorld = ChatWorld()


def getContent(input_file):
    # 读取文件内容
    with open(input_file.name, 'r', encoding='utf-8') as f:
        logging.info(f"read file {input_file.name}")
        input_text = f.read()
        logging.info(f"file content: {input_text}")

    # 保存文件内容
    input_text_list = input_text.split("\n")
    chatWorld.initDB(input_text_list)
    role_name_set = set()

    # 读取角色名
    for line in input_text_list:
        role_name_set.add(line.split(":")[0])

    role_name_list = [i for i in role_name_set if i != ""]
    logging.info(f"role_name_list: {role_name_list}")

    return gr.Radio(choices=role_name_list, interactive=True, value=role_name_list[0]), gr.Radio(choices=role_name_list, interactive=True, value=role_name_list[-1])


def submit_message(message, history, model_role_name, role_name, model_role_nickname, role_nickname):
    chatWorld.setRoleName(model_role_name, model_role_nickname)
    response = chatWorld.chat(message,
                              role_name, role_nickname, use_local_model=True)
    return response


with gr.Blocks() as demo:

    upload_c = gr.File(label="上传文档文件")

    with gr.Row():
        model_role_name = gr.Radio([], label="模型角色名")
        model_role_nickname = gr.Textbox(label="模型角色昵称")

    with gr.Row():
        role_name = gr.Radio([], label="角色名")
        role_nickname = gr.Textbox(label="角色昵称")

    upload_c.upload(fn=getContent, inputs=upload_c,
                    outputs=[model_role_name, role_name])

    chatBox = gr.ChatInterface(
        submit_message, chatbot=gr.Chatbot(height=400, render=False), additional_inputs=[model_role_name, role_name, model_role_nickname, role_nickname])


demo.launch(share=True, debug=True, server_name="0.0.0.0")
