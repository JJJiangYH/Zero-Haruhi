from transformers import AutoModel, AutoTokenizer


def initEmbedding(model_name="BAAI/bge-small-zh-v1.5", **model_wargs):
    return AutoModel.from_pretrained(model_name, **model_wargs)


def initTokenizer(model_name="BAAI/bge-small-zh-v1.5", **model_wargs):
    return AutoTokenizer.from_pretrained(model_name)
