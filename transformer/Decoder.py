import torch.nn as nn
from MHA import MultiHeadAttention
form FFN import FeedForward

class DecoderLayer(nnn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.Dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.layerNorm(d_model)

    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        # tgt: (batch_size, tgt_seq_len, d_model)
        # memory: (batch_size, src_seq_len, d_model)
        # tgt_mask: (batch_size, 1, 1, tgt_seq_len)
        # src_mask: (batch_size, 1, 1, src_seq_len)

        x = tgt
        output = self.self_attn(x, x, x, tgt_mask)    # (batch_size, tgt_seq_len, d_model)
        x = self.norm1(x + self.dropout1(output))    # add & norm

        output = self.cross_attn(x, src, src, src_mask)    # (batch_size, seq_len, d_model)
        x = self.norm2(x + self.dropout2(output))    # add & norm

        output = self.ffn(x)    # (batch_size, seq_len, d_model)
        x = self.norm3(x + self.dropout3(output))    # add & norm
        return x    # (batch_size, seq_len, d_model)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        # x: (batch_size, tgt_seq_len, d_model)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x    # (batch_size, seq_len, d_model)