import torch.nn as nn
from MHA import MultiHeadAttention
from FFN import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
            
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        attn_output = self.self_attn(x, x, x, mask)    # (batch_size, seq_len, d_model)
        x = self.norm1(x + self.dropout1(attn_output))    # add & norm
        
        ffn_output = self.ffn(x)    
        x = self.norm2(x + self.dropout2(ffn_output))    # add & norm
        return x    # (batch_size, seq_len, d_model)
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask) # (batch_size, seq_len, d_model)
        x = self.norm(x) # (batch_size, seq_len, d_model)
        return x    # (batch_size, seq_len, d_model)