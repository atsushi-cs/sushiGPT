import torch
import torch.nn as nn
import torch.nn.functional as F
from config import block_size, dropout

class Attention(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()

        self.q = nn.Linear(n_embd, head_size)    
        self.k = nn.Linear(n_embd, head_size)
        self.v = nn.Linear(n_embd, head_size)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        # Raw Scores
        raw = Q @ K.transpose(-2, -1)
        # Scale
        scale = raw / K.shape[-1] ** 0.5
        # Causal Mask
        seq_len = x.shape[1]
        scale = scale.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
        # Softmax + Dropout
        attn = F.softmax(scale, dim= -1)
        attn = self.dropout(attn)
        # Weighted Sum
        out = attn @ V

        return out