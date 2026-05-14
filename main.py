from datasets import load_dataset
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset, DataLoader, Sampler
import math
import random
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.latintext = [d[0] for d in dataset]
        self.englishtxt = [d[1] for d in dataset]
        self.tokenizer = tokenizer
        self.sos_token = tokenizer.token_to_id("<sos>")
        self.eos_token = tokenizer.token_to_id("<eos>")
        self.max_length = max_length

    def __len__(self):
        return len(self.latintext)

    def __getitem__(self, idx):
        latin_sentence = self.latintext[idx]
        english_sentence = self.englishtxt[idx]

        src = self.tokenizer.encode(latin_sentence).ids
        src = src[:self.max_length]

        english_tokens = self.tokenizer.encode(english_sentence).ids
        english_tokens = english_tokens[:self.max_length-2]
        tgt_ids = [self.sos_token] + english_tokens + [self.eos_token]
        return torch.tensor(src), torch.tensor(tgt_ids), latin_sentence, english_sentence


def collate_fn(pad_id):
    def collate(batch):
        src_seqs = [item[0] for item in batch]
        tgt_seqs = [item[1] for item in batch]

        # pad_sequence handles the ragged stacking for you
        src_padded = pad_sequence(
            src_seqs, batch_first=True, padding_value=pad_id)
        tgt_padded = pad_sequence(
            tgt_seqs, batch_first=True, padding_value=pad_id)

        # Padding masks: True where token is padding
        src_padding_mask = (src_padded == pad_id)
        tgt_padding_mask = (tgt_padded == pad_id)

        return {
            "src": src_padded,
            "tgt": tgt_padded,
            "src_padding_mask": src_padding_mask,
            "tgt_padding_mask": tgt_padding_mask,
        }
    return collate


class LengthBucketSampler(Sampler):
    def __init__(self, lengths, batch_size, bucket_size_multiplier=100):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        random.shuffle(indices)
        # Sort within large buckets to keep similar lengths together but preserve some randomness
        for i in range(0, len(indices), self.bucket_size):
            chunk = indices[i: i + self.bucket_size]
            chunk.sort(key=lambda x: self.lengths[x])
            for j in range(0, len(chunk), self.batch_size):
                yield chunk[j: j + self.batch_size]

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):             # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoderDecoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, num_layers=3):
        super().__init__()
        self.pos_encoding = PositionalEncoding(emb_dim, dropout=0.3)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=emb_dim, nhead=8,
                                             batch_first=True), num_layers=num_layers
        )
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(d_model=emb_dim, nhead=8,
                                             batch_first=True), num_layers=num_layers
        )
        self.embedding = torch.nn.Embedding(input_dim, emb_dim)
        self.fc_out = torch.nn.Linear(emb_dim, input_dim)

    def forward(self, src, tgt=None, src_padding_mask=None,
                tgt_padding_mask=None, tgt_mask=None):

        # Embedding + pos_encoding
        src_emb = self.pos_encoding(self.embedding(src))
        # Encode source sequence
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        tgt_emb = self.pos_encoding(self.embedding(tgt))
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_padding_mask)
        return self.fc_out(output)

    def encode(self, src, src_padding_mask=None):
        src_emb = self.pos_encoding(self.embedding(src))
        return self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
        tgt_emb = self.pos_encoding(self.embedding(tgt))
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_padding_mask)
        return self.fc_out(output)


def translate(model, tokenizer, latin_sentence, device, max_length=128):
    model.eval()
    with torch.no_grad():
        # Tokenize input:
        src_ids = tokenizer.encode(latin_sentence).ids
        src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
        # Encode:
        memory = model.encode(src_tensor)
        # Start decoding with <sos> token
        tgt_ids = [tokenizer.token_to_id("<sos>")]
        for _ in range(max_length):
            tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                len(tgt_ids), device=device)
            output = model.decode(tgt_tensor, memory, tgt_mask=tgt_mask)
            next_token_id = output.argmax(dim=-1)[0, -1].item()
            if next_token_id == tokenizer.token_to_id("<eos>"):
                break
            tgt_ids.append(next_token_id)
    return tokenizer.decode(tgt_ids[1:])  # Skip <sos> token


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_tokenizer(dataset):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=16000, special_tokens=[
                         "<pad>", "<sos>", "<eos>", "<unk>"])
    tokenizer.train_from_iterator(
        [d[0] for d in dataset] + [d[1] for d in dataset], trainer=trainer)
    return tokenizer


def load_data():
    ds = load_dataset("grosenthal/latin_english_translation")
    train_ds = ds["train"]
    val_ds = ds["valid"]
    test_ds = ds["test"]
    return train_ds, val_ds, test_ds


def preprocess_data(ds):
    return [(d["la"], d["en"]) for d in ds]


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch["src"].to(device)
            tgt_input = batch["tgt"][:, :-1].to(device)
            tgt_label = batch["tgt"][:, 1:].to(device)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                tgt_input.size(1), device=device)
            output = model(src, tgt_input, tgt_mask=tgt_mask,
                           src_padding_mask=batch["src_padding_mask"].to(
                               device),
                           tgt_padding_mask=batch["tgt_padding_mask"][:, :-1].to(device))
            output = output.reshape(-1, output.size(-1))
            loss = criterion(output, tgt_label.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)


def evaluate_bleu(model, tokenizer, device, bleu_samples):
    model.eval()
    bleu_scores = []
    for _, _, latin_sentence, english_sentence in bleu_samples:
        translation = translate(model, tokenizer, latin_sentence, device)
        reference = [english_sentence.split()]
        candidate = translation.split()
        bleu_scores.append(sentence_bleu(
            reference, candidate, smoothing_function=SmoothingFunction().method1))
    return sum(bleu_scores) / len(bleu_scores)


def test_translation(model, tokenizer, device, val_loader, num_samples=5):
    samples = random.sample(list(val_loader.dataset), num_samples)
    for _, _, latin_sentence, english_sentence in samples:
        translation = translate(model, tokenizer, latin_sentence, device)
        print(f"Latin: {latin_sentence}")
        print(f"Reference English: {english_sentence}")
        print(f"Model Translation: {translation}")
        print("-" * 50)


def train(train, val, test, load=None, batch_size=32, num_epochs=3, save=False, emb_dim=256):
    train_ds, val_ds, test_ds = preprocess_data(
        train), preprocess_data(val), preprocess_data(test)
    tokenizer = build_tokenizer(train_ds)
    train_ds = TranslationDataset(train_ds, tokenizer)
    val_ds = TranslationDataset(val_ds, tokenizer)
    test_ds = TranslationDataset(test_ds, tokenizer)

    pad_id = tokenizer.token_to_id("<pad>")
    train_sampler = LengthBucketSampler(
        [len(src) for src, _, _, _ in train_ds], batch_size=batch_size)
    val_sampler = LengthBucketSampler(
        [len(src) for src, _, _, _ in val_ds], batch_size=batch_size)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              collate_fn=collate_fn(pad_id))
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler,
                            collate_fn=collate_fn(pad_id))

    device = get_device()
    start_epoch = 0
    if isinstance(load, str):
        checkpoint = torch.load(load, map_location=device)
        emb_dim = checkpoint["emb_dim"]
        start_epoch = checkpoint["epoch"]

    model = TransformerEncoderDecoder(
        input_dim=tokenizer.get_vocab_size(), emb_dim=emb_dim
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1)
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_id, label_smoothing=0.1)

    if isinstance(load, str):
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    bleu_samples = random.sample(list(val_ds), 50)
    total_epochs = start_epoch + num_epochs
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        progress = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        for batch in progress:
            src = batch["src"].to(device)
            tgt_input = batch["tgt"][:, :-1].to(device)
            tgt_label = batch["tgt"][:, 1:].to(device)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                tgt_input.size(1), device=device)
            optimizer.zero_grad()
            output = model(src, tgt_input, tgt_mask=tgt_mask,
                           src_padding_mask=batch["src_padding_mask"].to(
                               device),
                           tgt_padding_mask=batch["tgt_padding_mask"][:, :-1].to(device))
            output = output.reshape(-1, output.size(-1))
            loss = criterion(output, tgt_label.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        val_loss = evaluate(model, val_loader, criterion, device)
        bleu_score = evaluate_bleu(model, tokenizer, device, bleu_samples)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch+1}/{total_epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu_score:.4f}")
        if save and (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "emb_dim": emb_dim,
            }, f"checkpoint_epoch{epoch+1}.pt")

    return model, tokenizer, device, test_ds


def evaluate_test(model, tokenizer, device, test_ds, batch_size=16, bleu_samples=200):
    pad_id = tokenizer.token_to_id("<pad>")
    test_sampler = LengthBucketSampler(
        [len(src) for src, _, _, _ in test_ds], batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler,
                             collate_fn=collate_fn(pad_id))
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_id, label_smoothing=0.1)

    test_loss = evaluate(model, test_loader, criterion, device)
    samples = random.sample(list(test_ds), bleu_samples)
    bleu = evaluate_bleu(model, tokenizer, device, samples)
    print(f"Test Loss: {test_loss:.4f}, Test BLEU: {bleu:.4f}")
    return test_loss, bleu


def write_test_translations(model, tokenizer, device, test_ds, num_samples=10, output_file="translations.txt"):
    samples = random.sample(list(test_ds), num_samples)
    with open(output_file, "w") as f:
        for _, _, latin, reference in samples:
            translation = translate(model, tokenizer, latin, device)
            f.write(f'"{latin}" : "{reference}" : "{translation}"\n')


def main():
    train_ds, val_ds, test_ds = load_data()
    model, tokenizer, device, test_ds = train(
        train_ds, val_ds, test_ds, num_epochs=40, save=True)
    evaluate_test(model, tokenizer, device, test_ds)
    write_test_translations(model, tokenizer, device, test_ds)


if __name__ == "__main__":
    tr_ds, val, test = load_data()
    tokenizer = build_tokenizer(preprocess_data(tr_ds))
    test_ds = TranslationDataset(preprocess_data(test), tokenizer)

    device = get_device()
    checkpoint = torch.load("checkpoint_epoch40.pt", map_location=device)
    model = TransformerEncoderDecoder(
        input_dim=tokenizer.get_vocab_size(), emb_dim=checkpoint["emb_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    evaluate_test(model, tokenizer, device, test_ds)
    write_test_translations(model, tokenizer, device, test_ds)
