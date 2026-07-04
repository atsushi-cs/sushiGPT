import torch
import torch.nn as nn
import torch.nn.functional as F
from config import dropout
from attention import Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()

        head_size = n_embd // n_heads

        self.heads = nn.ModuleList([Attention(n_embd, head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out
