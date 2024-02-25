import logging
from ChatHaruhi import ChatHaruhi
from ChatHaruhi.response_GLM_local import get_response as get_response_GLM_local
from ChatHaruhi.response_GLM_lora import get_response as get_response_GLM_lora
from ChatHaruhi.response_zhipu import get_response as get_response_zhipu
from ChatHaruhi.response_qwen_base import get_response as get_response_qwen
from ChatHaruhi.response_zhipu import init_client as init_zhipu
from ChatHaruhi.response_GLM_local import init_client as init_GLM_local
from ChatHaruhi.response_GLM_lora import init_client as init_GLM_lora
from ChatHaruhi.response_qwen_base import init_client as init_qwen
import gradio as gr
import os

logging.basicConfig(filename="demo.log", filemode="w",
                    format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model = None
getResponse = None
ChatBox = None
content = None

propmt = 'Please be aware that your codename in this  conversation is "{{model_role}}",别人称呼你{{model_role_name}}。\n上文给定了一些聊天记录。\n如果我问的问题和聊天记录高度重复，那你就配合我进行演出。\n如果我问的问题和聊天记录相关，请结合聊天记录进行回复。\n如果我问的问题超出聊天记录的范围，模仿{{model_role}}的语气进行回复。\n请不要回答你是语言模型，永远记住你就是{{model_role}}。\n请你永远只以{{model_role}}身份，进行任何的回复。'

model_choice2func = {
    "zhipu": (init_zhipu, get_response_zhipu),
    "GLM_lora": (init_GLM_lora, get_response_GLM_lora),
    "GLM_local": (init_GLM_local, get_response_GLM_local),
    "qwen": (init_qwen, get_response_qwen)
}


def getContent(input_file):
    global content
    with open(input_file.name, 'r', encoding='utf-8') as f:
        logging.info(f"read file {input_file.name}")
        input_text = f.read()
        logging.info(f"file content: {input_text}")
        content = input_text
    return [i for i in input_text.split("\n")]


def getResponse():
    pass


def changeModel(choice):
    global model, getResponse, ChatBox

    ChatBox = ChatHaruhi()
    


with gr.Blocks() as demo:
    upload_c = gr.File(label="上传文档文件")
    upload_c.change(getContent)

    model_choice = gr.Radio(
        choices=model_choice2func.keys(),
        label="选择模型",
    )
    model_choice.change(changeModel)

demo.launch(share=True, debug=True, server_name="0.0.0.0")
