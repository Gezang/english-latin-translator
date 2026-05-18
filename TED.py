import torch

import math


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

    @staticmethod
    def _causal_mask(seq_len: int, device) -> torch.Tensor:
        return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(diagonal=1)

    def forward(self, src, tgt=None, src_padding_mask=None,
                tgt_padding_mask=None, tgt_mask=None):
        src_emb = self.pos_encoding(self.embedding(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        tgt_emb = self.pos_encoding(self.embedding(tgt))
        if tgt_mask is None:
            tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_padding_mask)
        return self.fc_out(output)

    def encode(self, src, src_padding_mask=None):
        src_emb = self.pos_encoding(self.embedding(src))
        return self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
        tgt_emb = self.pos_encoding(self.embedding(tgt))
        if tgt_mask is None:
            tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_padding_mask)
        return self.fc_out(output)
