import logging
import math
import os
from tempfile import _TemporaryFileWrapper

import gradio as gr
from jinja2 import Template
import torch

from zhipuai import ZhipuAI

from ChatHaruhi import ChatHaruhi
from ChatHaruhi.response_GLM_local import get_response as get_response_GLM_local
from ChatHaruhi.response_GLM_local import init_client as init_GLM_local
from ChatHaruhi.response_GLM_lora import get_response as get_response_GLM_lora
from ChatHaruhi.response_GLM_lora import init_client as init_GLM_lora
from ChatHaruhi.response_qwen_base import get_response as get_response_qwen
from ChatHaruhi.response_qwen_base import init_client as init_qwen
from ChatHaruhi.response_zhipu import get_response as get_response_zhipu
from ChatHaruhi.response_zhipu import init_client as init_zhipu
from utils import initEmbedding, initTokenizer

logging.basicConfig(level=logging.INFO, filename="demo.log", filemode="w",
                    format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model = None
getResponse = None
chatbot = None
content = None
init = None
model_role_name = None
vec = None

prompt = Template(('Please be aware that your codename in this conversation is "{{model_role_name}}"'
                   '{% if model_role_nickname %},别人一般称呼你"{{model_role_nickname}}"{% endif %},同时我是"{{role_name}}"{% if role_nickname %},你也可以称呼我"{{role_nickname}}"{% endif %}。\n'
                   '下文给定了一些聊天记录，位于##分隔号中。\n'
                   '如果我问的问题和聊天记录高度重复，那你就配合我进行演出。\n'
                   '如果我问的问题和聊天记录相关，请结合聊天记录进行回复。\n'
                   '如果我问的问题超出聊天记录的范围，模仿{{model_role_name}}的语气进行回复。\n'
                   '请不要回答你是语言模型，永远记住你就是{{model_role_name}}。\n'
                   '请你永远只以{{model_role_name}}身份，进行任何的回复。\n'
                   ))
rag = '{{RAG对话}}'

model_choice2func = {
    "zhipu": (init_zhipu, get_response_zhipu),
    "GLM_lora": (init_GLM_lora, get_response_GLM_lora),
    "GLM_local": (init_GLM_local, get_response_GLM_local),
    "qwen": (init_qwen, get_response_qwen)
}


class ChatWorld:
    def __init__(self) -> None:
        self.embedding = initEmbedding()
        self.tokenizer = initTokenizer()
        self.story_vec: list[dict] = None
        self.storys = None

    def getEmbeddingsFromStory(self, stories: list[str]):
        if self.story_vec:
            # 判断是否与当前的相同
            if len(self.story_vec) == len(stories) and all([self.story_vec[i]["text"] == stories[i] for i in range(len(stories))]):
                return self.story_vec

        if self.embedding is None:
            self.embedding = initEmbedding()

        if self.tokenizer is None:
            self.tokenizer = initTokenizer()

        self.story_vec = []
        for story in stories:
            with torch.no_grad():
                inputs = self.tokenizer(
                    story, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self.embedding(**inputs)[0][:, 0]
                vec = torch.nn.functional.normalize(
                    outputs, p=2, dim=1).tolist()
            self.story_vec.append({"text": story, "vec": vec})

        return [self.story_vec[i]["vec"] for i in range(len(stories))]

    def setStorys(self, stories: list[str]):
        self.storys = stories
        self.getEmbeddingsFromStory(stories)

chatWorld = ChatWorld()

def getContent(input_file: _TemporaryFileWrapper):
    # 读取文件内容
    with open(input_file.name, 'r', encoding='utf-8') as f:
        logging.info(f"read file {input_file.name}")
        input_text = f.read()
        logging.info(f"file content: {input_text}")

    # 保存文件内容
    input_text_list = input_text.split("\n")

    role_name_set = set()

    # 读取角色名
    for line in input_text_list:
        role_name_set.add(line.split(":")[0])

    role_name_list = [i for i in role_name_set if i != ""]
    logging.info(f"role_name_list: {role_name_list}")

    return gr.Radio(choices=role_name_list, interactive=True, value=role_name_list[0]), gr.Radio(choices=role_name_list, interactive=True, value=role_name_list[-1])


def changeModel(model_choice):
    global model
    global getResponse
    global chatBox
    global init

    if model_choice in model_choice2func:
        init, getResponse = model_choice2func[model_choice]
        model = model_choice
    else:
        logging.error(f"model {model_choice} not found")
        raise ValueError(f"model {model_choice} not found")


def init_chatbot(status, _model_role_name, _model_role_nickname, _role_name, _role_nickname):
    global init
    global prompt
    global getResponse
    global chatbot
    global model_role_name
    global content

    if status == "加载模型":
        logging.info(
            f"model_role_name: {_model_role_name}, model_role_nickname: {_model_role_nickname}, role_name: {_role_name}, role_nickname: {_role_nickname}")
        personal_prompt = prompt.render(
            model_role_name=_model_role_name, model_role_nickname=_model_role_nickname, role_name=_role_name, role_nickname=_role_nickname) + rag
        logging.info(f"personal_prompt: {personal_prompt}")
        # 初始化模型
        chatbot = ChatHaruhi(role_name=_model_role_name,
                             persona=personal_prompt, llm=getResponse, stories=content, story_vecs=chatWorld.getEmbeddingsFromStory(content), verbose=True)
        model_role_name = _model_role_name
        return "卸载模型"
    if status == "卸载模型":
        chatbot = None
        return "加载模型"


def submit_message(message, history):
    global chatbot
    global model_role_name

    response = chatbot.chat(user=model_role_name, text=message)
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
    with gr.Row():
        model_choice = gr.Radio(
            choices=model_choice2func.keys(),
            label="选择模型",
            value="zhipu"
        )
        model_load = gr.Button(
            value="加载模型",
        )

    chatBox = gr.ChatInterface(
        submit_message, chatbot=gr.Chatbot(height=400, render=False))

    model_choice.change(changeModel, model_choice)
    model_load.click(init_chatbot, inputs=[
                     model_load, model_role_name, model_role_nickname, role_name, role_nickname], outputs=model_load)


demo.launch(share=True, debug=True, server_name="0.0.0.0")
