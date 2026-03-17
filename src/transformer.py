import torch
import torch.nn as nn
import math
from src.attention import MultiHeadAttention
from src.feed_forward import PositionwiseFeedForward
from src.positional_encoding import PositionalEncoding
from src.encoder import Encoder
from src.decoder import Decoder
from src.masks import create_padding_mask, create_decoder_mask


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048,
                 max_seq_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.src_embedding       = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding       = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder             = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder             = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.output_linear       = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_linear(decoder_output)
