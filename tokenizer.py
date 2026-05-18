from transformers import AutoTokenizer, PreTrainedTokenizerBase


def build_tokenizer(model_name: str = "bert-base-multilingual-cased") -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)
