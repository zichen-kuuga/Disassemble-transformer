import torch.nn as nn
from PositionEncoding import PositionalEncoding
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(d_model, n_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, num_layers, dropout)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)

        src = self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim)    # (batch_size, src_seq_len, d_model)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim)    # (batch_size, tgt_seq_len, d_model)

        src = self.dropout(src)    # (batch_size, src_seq_len, d_model)
        tgt = self.dropout(tgt)    # (batch_size, tgt_seq_len, d_model)

        src = self.positional_encoding(src)    # (batch_size, src_seq_len, d_model)
        tgt = self.positional_encoding(tgt)    # (batch_size, tgt_seq_len, d_model)

        enc_output = self.encoder(src, src_mask)    # (batch_size, src_seq_len, d_model)
        dec_output = self.decoder(tgt, enc_output, tgt_mask)    # (batch_size, tgt_seq_len, d_model)

        output = self.fc_out(dec_output)    # (batch_size, tgt_seq_len, tgt_vocab_size)
        return output    # (batch_size, tgt_seq_len, tgt_vocab_size)