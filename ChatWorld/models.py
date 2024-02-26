import os
from string import Template
from transformers import AutoTokenizer, AutoModelForCausalLM
from zhipuai import ZhipuAI


class qwen_model:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True).eval()

    def get_response(self, message):
        message = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True)
        print(message)
        model_inputs = self.tokenizer(
            [message], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]

        return response


class GLM():
    def __init__(self, model_name="silk-road/Haruhi-Zero-GLM3-6B-0_4"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        client = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map="auto")

        self.client = client.eval()

    def message2query(self, messages) -> str:
        # [{'role': 'user', 'content': '老师: 同学请自我介绍一下'}]
        # <|system|>
        # You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
        # <|user|>
        # Hello
        # <|assistant|>
        # Hello, I'm ChatGLM3. What can I assist you today?
        template = Template("<|$role|>\n$content\n")

        return "".join([template.substitute(message) for message in messages])

    def get_response(self, message):
        response, history = self.client.chat(
            self.tokenizer, self.message2query(message))
        print(self.message2query(message))
        return response


class GLM_api:
    def __init__(self, model_name="glm-4"):
        API_KEY = os.environ.get(["ZHIPU_API_KEY"])

        if API_KEY is None:
            from google.colab import userdata
            API_KEY = userdata.get("ZHIPU_API_KEY")

        self.client = ZhipuAI(api_key=API_KEY)
        self.model = model_name

    def chat(self, message):
        print(message)
        response = self.client.chat.completions.create(
            model=self.model, messages=message)
        return response.choices[0].message.content
