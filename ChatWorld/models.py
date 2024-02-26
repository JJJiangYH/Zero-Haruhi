from transformers import AutoTokenizer, AutoModelForCausalLM


class qwen_model:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True).eval()

    def get_response(self, message):
        message = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([message], return_tensors="pt")
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
