from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast


def build_tokenizer(model_name: str = "bert-base-multilingual-cased",
                    training_texts: Iterable[str] | None = None,
                    vocab_size: int = 16000) -> PreTrainedTokenizerBase:
    if model_name == "BPE":
        if training_texts is None:
            raise ValueError("BPE tokenizer requires training_texts")
        return _train_bpe(training_texts, vocab_size)
    return AutoTokenizer.from_pretrained(model_name)


def _train_bpe(texts: Iterable[str], vocab_size: int) -> PreTrainedTokenizerFast:
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )
