from jinja2 import Template
import torch

from .models import qwen_model

from .NaiveDB import NaiveDB
from .utils import *


class ChatWorld:
    def __init__(self, pretrained_model_name_or_path="silk-road/Haruhi-Zero-14B-0_5", embedding_model_name_or_path="BAAI/bge-small-zh-v1.5") -> None:
        self.embedding = initEmbedding(embedding_model_name_or_path)
        self.tokenizer = initTokenizer(embedding_model_name_or_path)
        self.story_vec: list[dict] = None
        self.storys = None
        self.model_role_name = None
        self.model_role_nickname = None
        self.model_name = pretrained_model_name_or_path

        self.history = []

        self.client = None
        self.model = qwen_model(pretrained_model_name_or_path)
        self.db = NaiveDB()
        self.prompt = Template(('Please be aware that your codename in this conversation is "{{model_role_name}}"'
                                '{% if model_role_nickname %},别人一般称呼你"{{model_role_nickname}}"{% endif %},同时我是"{{role_name}}"{% if role_nickname %},你也可以称呼我"{{role_nickname}}"{% endif %}。\n'
                                '下文给定了一些聊天记录，位于##分隔号中。\n'
                                '如果我问的问题和聊天记录高度重复，那你就配合我进行演出。\n'
                                '如果我问的问题和聊天记录相关，请结合聊天记录进行回复。\n'
                                '如果我问的问题超出聊天记录的范围，模仿{{model_role_name}}的语气进行回复。\n'
                                '请不要回答你是语言模型，永远记住你就是{{model_role_name}}。\n'
                                '请你永远只以{{model_role_name}}身份，进行任何的回复。\n'
                                ))

    def getEmbeddingsFromStory(self, stories: list[str]):
        if self.story_vec:
            # 判断是否与当前的相同
            if len(self.story_vec) == len(stories) and all([self.story_vec[i]["text"] == stories[i] for i in range(len(stories))]):
                return [self.story_vec[i]["vec"] for i in range(len(stories))]

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
                    outputs, p=2, dim=1).tolist()[0]

            self.story_vec.append({"text": story, "vec": vec})

        return [self.story_vec[i]["vec"] for i in range(len(stories))]

    def initDB(self, storys: list[str]):
        story_vecs = self.getEmbeddingsFromStory(storys)
        self.db.build_db(storys, story_vecs)

    def setRoleName(self, role_name, role_nick_name=None):
        self.model_role_name = role_name
        self.model_role_nickname = role_nick_name

    def getSystemPrompt(self, role_name, role_nick_name):
        assert self.model_role_name, "Please set model role name first"

        return {"role": "system", "content": self.prompt.render(model_role_name=self.model_role_name, model_role_nickname=self.model_role_nickname, role_name=role_name, role_nickname=role_nick_name)}

    def chat(self, user_role_name: str, text: str, user_role_nick_name: str = None, use_local_model=False):
        message = [self.getSystemPrompt(
            user_role_name, user_role_nick_name)] + self.history

        if use_local_model:
            response = self.model.get_response(message)
        else:
            response = self.client.chat(
                user_role_name, text, user_role_nick_name)

        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "model", "content": response})
        return response
