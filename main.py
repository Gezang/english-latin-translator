import random
from typing import Callable

import torch
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from TED import TransformerEncoderDecoder
from laBERTmodel import Bert_Decoder
from tokenizer import build_tokenizer


class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer: PreTrainedTokenizerBase, max_length: int = 128):
        self.latintext = [d[0] for d in dataset]
        self.englishtxt = [d[1] for d in dataset]
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.sep_token_id
        self.max_length = max_length

    def __len__(self):
        return len(self.latintext)

    def __getitem__(self, idx):
        latin = self.latintext[idx]
        english = self.englishtxt[idx]

        src = self.tokenizer.encode(
            latin, add_special_tokens=True,
            truncation=True, max_length=self.max_length,
        )
        english_ids = self.tokenizer.encode(
            english, add_special_tokens=False,
            truncation=True, max_length=self.max_length - 2,
        )
        tgt = [self.bos_id] + english_ids + [self.eos_id]
        return torch.tensor(src), torch.tensor(tgt), latin, english


def collate_fn(pad_id: int):
    def collate(batch):
        src_seqs = [item[0] for item in batch]
        tgt_seqs = [item[1] for item in batch]

        src_padded = pad_sequence(
            src_seqs, batch_first=True, padding_value=pad_id)
        tgt_padded = pad_sequence(
            tgt_seqs, batch_first=True, padding_value=pad_id)

        # True where token is padding (PyTorch transformer convention).
        return {
            "src": src_padded,
            "tgt": tgt_padded,
            "src_padding_mask": (src_padded == pad_id),
            "tgt_padding_mask": (tgt_padded == pad_id),
        }
    return collate


class LengthBucketSampler(Sampler):
    def __init__(self, lengths, batch_size: int, bucket_size_multiplier: int = 100):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        random.shuffle(indices)
        for i in range(0, len(indices), self.bucket_size):
            chunk = indices[i: i + self.bucket_size]
            chunk.sort(key=lambda x: self.lengths[x])
            for j in range(0, len(chunk), self.batch_size):
                yield chunk[j: j + self.batch_size]

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data():
    ds = load_dataset("grosenthal/latin_english_translation")
    return ds["train"], ds["valid"], ds["test"]


def preprocess_data(ds):
    return [(d["la"], d["en"]) for d in ds]


def build_loaders(train_raw, val_raw, test_raw,
                  tokenizer: PreTrainedTokenizerBase, batch_size: int = 32):
    train_ds = TranslationDataset(preprocess_data(train_raw), tokenizer)
    val_ds = TranslationDataset(preprocess_data(val_raw), tokenizer)
    test_ds = TranslationDataset(preprocess_data(test_raw), tokenizer)

    collate = collate_fn(tokenizer.pad_token_id)

    def make_loader(ds):
        sampler = LengthBucketSampler(
            [len(src) for src, *_ in ds], batch_size=batch_size)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate)

    return make_loader(train_ds), make_loader(val_ds), make_loader(test_ds)


def translate_greedy(model, tokenizer: PreTrainedTokenizerBase,
                     sentence: str, device: torch.device,
                     max_length: int = 128) -> str:

    model.eval()
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.sep_token_id

    with torch.no_grad():
        # Encode input (source):
        src_ids = tokenizer.encode(sentence, add_special_tokens=True)
        src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
        memory = core.encode(src_tensor)
        tgt_ids = [bos_id]
        for _ in range(max_length):
            tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)
            output = core.decode(tgt_tensor, memory)
            next_id = output.argmax(dim=-1)[0, -1].item()
            if next_id == eos_id:
                break
            tgt_ids.append(next_id)
    return tokenizer.decode(tgt_ids[1:], skip_special_tokens=True)


def translate_beam(model, tokenizer: PreTrainedTokenizerBase,
                   sentence: str, device: torch.device,
                   max_length: int = 128, beam_size: int = 5,
                   length_normalization: float = 1) -> str:
    """Beam-search decoding. Returns the highest length-normalized hypothesis."""
    model.eval()
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.sep_token_id

    with torch.no_grad():
        # Encode the source once and replicate the memory across all beams.
        src_ids = tokenizer.encode(sentence, add_special_tokens=True)
        src_tensor = torch.tensor(src_ids, device=device).unsqueeze(0)
        memory = core.encode(src_tensor)
        memory = memory.expand(beam_size, -1, -1).contiguous()  # (k, S, D)

        # Initialize k beams at BOS. On step 0 all beams are identical, so we
        # mask all but one to -inf — otherwise top-k picks the same token k
        # times and we get k duplicate beams forever.
        tokens = torch.full((beam_size, 1), bos_id,
                            dtype=torch.long, device=device)
        scores = torch.zeros(beam_size, device=device)
        scores[1:] = float("-inf")

        # Completed hypotheses, stored as (length-normalized score, token list).
        finished: list[tuple[float, list[int]]] = []

        for _ in range(max_length):
            # Score the next-token distribution for every active beam.
            logits = core.decode(tokens, memory)                 # (k, L, V)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (k, V)
            vocab_size = log_probs.size(-1)

            # Cumulative log-prob for each (beam, next-token) candidate.
            candidate_scores = scores.unsqueeze(1) + log_probs   # (k, V)

            # Pick the global top-k across all (k * V) flattened candidates.
            flat = candidate_scores.view(-1)
            top_scores, top_idx = flat.topk(beam_size)
            beam_idx = top_idx // vocab_size   # parent beam of each candidate
            token_idx = top_idx % vocab_size   # token to append

            # Rebuild beams: gather parent histories and append the new token.
            tokens = torch.cat(
                [tokens[beam_idx], token_idx.unsqueeze(1)], dim=1)
            scores = top_scores

            # Retire any beam whose new token is EOS. We normalize by length
            # (excluding BOS) to counter beam search's bias toward short
            # outputs, then set the slot's score to -inf so it cannot be
            # selected again on subsequent steps.
            eos_positions = (token_idx == eos_id).nonzero(as_tuple=True)[0]
            for i in eos_positions.tolist():
                seq = tokens[i].tolist()
                length = len(seq) - 1
                finished.append(
                    (scores[i].item() / (length**length_normalization), seq))
                scores[i] = float("-inf")

            # Once we have k complete hypotheses, no still-active beam can
            # improve on the best of them (scores are monotonically decreasing
            # log-probs), so we can stop.
            if len(finished) >= beam_size:
                break

        # Fallback: if max_length was reached before k beams hit EOS, score
        # the still-active beams with the same length normalization so we
        # always have something to return.
        for i in range(beam_size):
            if scores[i].item() == float("-inf"):
                continue
            seq = tokens[i].tolist()
            length = len(seq) - 1
            finished.append(
                (scores[i].item() / (length**length_normalization), seq))

        _, best_tokens = max(finished, key=lambda x: x[0])
        return tokenizer.decode(best_tokens[1:], skip_special_tokens=True)


def _forward_batch(model, batch, criterion, device):
    src = batch["src"].to(device)
    tgt_input = batch["tgt"][:, :-1].to(device)
    tgt_label = batch["tgt"][:, 1:].to(device)
    output = model(
        src, tgt_input,
        src_padding_mask=batch["src_padding_mask"].to(device),
        tgt_padding_mask=batch["tgt_padding_mask"][:, :-1].to(device),
    )
    return criterion(output.reshape(-1, output.size(-1)), tgt_label.reshape(-1))


def evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            total_loss += _forward_batch(model,
                                         batch, criterion, device).item()
    return total_loss / len(loader)


def evaluate_bleu(model, tokenizer, device, samples) -> float:
    model.eval()
    smoothing = SmoothingFunction().method1
    scores = []
    for _, _, latin, english in samples:
        translation = translate_greedy(model, tokenizer, latin, device)
        scores.append(sentence_bleu(
            [english.split()], translation.split(), smoothing_function=smoothing))
    return sum(scores) / len(scores)


def _strip_module_prefix(state_dict):
    # Checkpoints saved from DataParallel/DistributedDataParallel have a
    # "module." prefix on every key; strip it so the state_dict loads into
    # the unwrapped model.
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in state_dict.items()}
    return state_dict


def train(model, train_loader, val_loader, tokenizer, device,
          num_epochs: int = 3, save: bool = False, checkpoint=None,
          checkpoint_prefix: str = "checkpoint"):
    pad_id = tokenizer.pad_token_id

    start_epoch = 0
    if checkpoint is not None:
        state = _strip_module_prefix(checkpoint["model_state"])
        model.load_state_dict(state)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1)
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_id, label_smoothing=0.1)

    if checkpoint is not None:
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    bleu_samples = random.sample(list(val_loader.dataset), 50)
    total_epochs = start_epoch + num_epochs

    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        for batch in progress:
            optimizer.zero_grad()
            loss = _forward_batch(model, batch, criterion, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = evaluate(model, val_loader, criterion, device)
        bleu = evaluate_bleu(model, tokenizer, device, bleu_samples)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{total_epochs}, "
              f"Loss: {total_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, BLEU: {bleu:.4f}")

        if save and (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }, f"{checkpoint_prefix}_epoch{epoch+1}.pt")

    return model


def train_amp(model, train_loader, val_loader, tokenizer, device,
              num_epochs: int = 3, save: bool = False, checkpoint=None,
              checkpoint_prefix: str = "checkpoint"):
    """Training loop with mixed-precision (fp16 autocast + GradScaler). CUDA only."""
    pad_id = tokenizer.pad_token_id

    start_epoch = 0
    if checkpoint is not None:
        state = _strip_module_prefix(checkpoint["model_state"])
        model.load_state_dict(state)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1)
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_id, label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")

    if checkpoint is not None:
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])

    bleu_samples = random.sample(list(val_loader.dataset), 50)
    total_epochs = start_epoch + num_epochs

    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        for batch in progress:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = _forward_batch(model, batch, criterion, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = evaluate(model, val_loader, criterion, device)
        bleu = evaluate_bleu(model, tokenizer, device, bleu_samples)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{total_epochs}, "
              f"Loss: {total_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, BLEU: {bleu:.4f}")

        if save and (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
            }, f"{checkpoint_prefix}_epoch{epoch+1}.pt")

    return model


def evaluate_test(model, tokenizer, device, test_loader):
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    test_loss = evaluate(model, test_loader, criterion, device)
    bleu = evaluate_bleu(model, tokenizer, device, list(test_loader.dataset))
    print(f"Test Loss: {test_loss:.4f}, Test BLEU: {bleu:.4f}")
    return test_loss, bleu


def write_test_translations(model, tokenizer, device, test_loader,
                            num_samples: int = 10,
                            output_file: str = "translations.txt"):
    samples = random.sample(list(test_loader.dataset), num_samples)
    with open(output_file, "w") as f:
        for _, _, latin, reference in samples:
            translation = translate_greedy(model, tokenizer, latin, device)
            f.write(f'"{latin}" : "{reference}" : "{translation}"\n')


ModelFactory = Callable[[PreTrainedTokenizerBase], torch.nn.Module]


def run(model_factory: ModelFactory,
        run_name: str,
        tokenizer_name: str = "bert-base-multilingual-cased",
        load_model: str | None = None,
        batch_size: int = 16,
        num_epochs: int = 20,
        save: bool = True,
        use_amp: bool = False):
    device = get_device()

    train_raw, val_raw, test_raw = load_data()

    training_texts = None
    if tokenizer_name == "BPE":
        training_texts = [d["la"] for d in train_raw] + \
            [d["en"] for d in train_raw]
    tokenizer = build_tokenizer(tokenizer_name, training_texts=training_texts)

    train_loader, val_loader, test_loader = build_loaders(
        train_raw, val_raw, test_raw, tokenizer, batch_size=batch_size)

    model = model_factory(tokenizer)
    checkpoint = torch.load(
        load_model, map_location=device) if load_model else None

    train_fn = train_amp if use_amp else train
    model = train_fn(model, train_loader, val_loader, tokenizer, device,
                     num_epochs=num_epochs, save=save, checkpoint=checkpoint,
                     checkpoint_prefix=run_name)

    evaluate_test(model, tokenizer, device, test_loader)
    write_test_translations(model, tokenizer, device, test_loader,
                            output_file=f"{run_name}_translations.txt")


def _ted_factory(tokenizer: PreTrainedTokenizerBase) -> torch.nn.Module:
    return TransformerEncoderDecoder(
        input_dim=tokenizer.vocab_size, emb_dim=512)


def _bert_decoder_factory(tokenizer: PreTrainedTokenizerBase) -> torch.nn.Module:
    return Bert_Decoder(tgt_vocab_size=tokenizer.vocab_size)


if __name__ == "__main__":
    run(model_factory=_bert_decoder_factory,
        run_name="bert_decoder", num_epochs=15, use_amp=True,
        batch_size=32)
    # run(model_factory=_ted_factory, run_name="ted")
