class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
​
        assert (
            self.d_k * n_heads == d_model
        ), f"d_model {d_model} not divisible by n_heads {n_heads}"
​
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
​
        self.dropout = nn.Dropout(dropout)
​
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q: (batch_size, n_heads, seq_len, d_k)
        # K: (batch_size, n_heads, seq_len, d_k)
        # V: (batch_size, n_heads, seq_len, d_k)
​
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)    # apply mask to scores
        
        attn_weights = F.softmax(scores, dim=-1)    # (batch_size, n_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)    # apply dropout to attention weights
        output = torch.matmul(attn_weights, V)    # (batch_size, n_heads, seq_len, d_k)
        return output
    
    def forward(self, Q, K, V, mask=None):
        # Q: (batch_size, seq_len, d_model)
        # K: (batch_size, seq_len, d_model)
        # V: (batch_size, seq_len, d_model)
​
        batch_size = Q.size(0)
​
        # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # (batch_size, n_heads, seq_len, d_k)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # (batch_size, n_heads, seq_len, d_k)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # (batch_size, n_heads, seq_len, d_k)
​
        # scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)    # (batch_size, n_heads, seq_len, d_k)
​
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)    # (batch_size, seq_len, d_model)
        output = self.W_o(attn_output)    # (batch_size, seq_len, d_model)
        return output    # (batch_size, seq_len, d_model)