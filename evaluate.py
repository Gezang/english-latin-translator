import argparse

import torch
import sacrebleu
from laBERTmodel import Bert_Decoder
from main import (build_loaders, evaluate, get_device, load_data,
                  translate_beam, translate_greedy)
from TED import TransformerEncoderDecoder
from tokenizer import build_tokenizer


def _build_model(name, tokenizer):
    if name == "bert_decoder":
        return Bert_Decoder(tgt_vocab_size=tokenizer.vocab_size)
    if name == "ted":
        return TransformerEncoderDecoder(
            input_dim=tokenizer.vocab_size, emb_dim=512)
    raise ValueError(f"Unknown model: {name}")


def _load_state(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"] if isinstance(
        ckpt, dict) and "model_state" in ckpt else ckpt
    # Checkpoints saved while wrapped in DataParallel have a "module." prefix.
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)


def _corpus_bleu(translate_fn, model, tokenizer, device, samples):
    hypotheses = []
    references = []
    for _, _, latin, english in samples:
        hypotheses.append(translate_fn(model, tokenizer, latin, device))
        references.append(english)
    return sacrebleu.corpus_bleu(hypotheses, [references])


def main():
    parser = argparse.ArgumentParser(
        description="Load a checkpoint and evaluate it on the test set "
                    "with both greedy and beam search.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a .pt checkpoint file.")
    parser.add_argument("--model", choices=["bert_decoder", "ted"],
                        default="bert_decoder")
    parser.add_argument("--tokenizer", default="bert-base-multilingual-cased")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    train_raw, val_raw, test_raw = load_data()

    training_texts = None
    if args.tokenizer == "BPE":
        training_texts = [d["la"] for d in train_raw] + \
            [d["en"] for d in train_raw]
    tokenizer = build_tokenizer(args.tokenizer, training_texts=training_texts)

    _, _, test_loader = build_loaders(
        train_raw, val_raw, test_raw, tokenizer, batch_size=args.batch_size)

    model = _build_model(args.model, tokenizer).to(device)
    _load_state(model, args.checkpoint, device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    test_loss = evaluate(model, test_loader, criterion, device)

    samples = list(test_loader.dataset)
    greedy_bleu = _corpus_bleu(
        translate_greedy, model, tokenizer, device, samples)

    def beam_fn(m, t, sentence, d):
        return translate_beam(m, t, sentence, d, beam_size=args.beam_size)
    beam_bleu = _corpus_bleu(beam_fn, model, tokenizer, device, samples)

    print(f"Test Loss:           {test_loss:.4f}")
    print(f"Greedy BLEU:         {greedy_bleu}")
    print(f"Beam (k={args.beam_size}) BLEU:  {beam_bleu}")


if __name__ == "__main__":
    main()


def evaluate_kaggle(model_checkpoint: str = None):
    device = torch.get_device()

    # get data
    train_raw, val_raw, test_raw = load_data()

    tokenizer = build_tokenizer()
    _, _, test_loader = build_loaders(
        train_raw, val_raw, test_raw, tokenizer, batch_size=32)
    name = "ted"
    model = _build_model(name, tokenizer).to(device)
    _load_state(model, model_checkpoint, device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    test_loss = evaluate(model, test_loader, criterion, device)

    samples = list(test_loader.dataset)
    greedy_bleu = _corpus_bleu(
        translate_greedy, model, tokenizer, device, samples)

    def beam_fn(m, t, sentence, d):
        return translate_beam(m, t, sentence, d, beam_size=5)
    beam_bleu = _corpus_bleu(beam_fn, model, tokenizer, device, samples)

    print(f"Test Loss:           {test_loss:.4f}")
    print(f"Greedy BLEU:         {greedy_bleu}")
    print(f"Beam (k=5) BLEU:  {beam_bleu}")
