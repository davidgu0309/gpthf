import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, ff_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        output = self.fc(x)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = FeedForward(hidden_dim, ff_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask=mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_dim = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.fc(attn_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)

    class FeedForward(nn.Module):
        def __init__(self, hidden_dim, ff_dim):
            super().__init__()
            self.fc1 = nn.Linear(hidden_dim, ff_dim)
            self.fc2 = nn.Linear(ff_dim, hidden_dim)

        def forward(self, x):
            x = nn.ReLU()(self.fc1(x))
            x = self.fc2(x)
            return x

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', self.pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x