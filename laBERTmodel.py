import torch
from transformers import BertModel

from TED import PositionalEncoding


class Bert_Decoder(torch.nn.Module):
    def __init__(self, tgt_vocab_size, num_decoder_layers=6,
                 pretrained_name: str = "bert-base-multilingual-cased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        emb_dim = self.bert.config.hidden_size  # 768

        self.pos_encoding = PositionalEncoding(emb_dim, dropout=0.1)
        self.tgt_embedding = torch.nn.Embedding(tgt_vocab_size, emb_dim)
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=emb_dim, nhead=12, batch_first=True),
            num_layers=num_decoder_layers,
        )
        self.fc_out = torch.nn.Linear(emb_dim, tgt_vocab_size)

    @staticmethod
    def _causal_mask(seq_len: int, device) -> torch.Tensor:
        return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(diagonal=1)

    def forward(self, src, tgt, src_padding_mask=None,
                tgt_padding_mask=None, tgt_mask=None):
        src_attention_mask = (~src_padding_mask).long() \
            if src_padding_mask is not None else None
        memory = self.bert(
            input_ids=src, attention_mask=src_attention_mask).last_hidden_state
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        if tgt_mask is None:
            tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        output = self.decoder(
            tgt_emb, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.fc_out(output)

    def encode(self, src, src_padding_mask=None):
        src_attention_mask = (~src_padding_mask).long() \
            if src_padding_mask is not None else None
        return self.bert(input_ids=src, attention_mask=src_attention_mask).last_hidden_state

    def decode(self, tgt, memory, tgt_mask=None,
               tgt_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        if tgt_mask is None:
            tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        output = self.decoder(
            tgt_emb, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(output)
