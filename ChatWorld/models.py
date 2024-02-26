from transformers import AutoTokenizer, AutoModelForCausalLM


class qwen_model:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()

    def get_response(self, message):
        self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return "test"