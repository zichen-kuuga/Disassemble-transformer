import torch
import math

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(0,max_len, dtype=torch.float).unsqueese(1) #(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model) #(1,max_len,d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term) #even index
        pe[0, :, 1::2] = torch.cos(position * div_term) #odd index
        self.register_buffer('pe', pe)
        
        def forward(self, x):
            # x: (batch_size, seq_len, d_model)
            x = x + self.pe[:, :x.size(1), :]# add positional encoding to input tensor
            return x