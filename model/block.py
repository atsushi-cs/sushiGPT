import torch
import torch.nn as nn
from model.multihead import MultiHeadAttention
from model.feedforward import FeedForward
from config import n_embd, n_heads

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = MultiHeadAttention()
        self.ff = FeedForward()

        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.attn(self.layernorm1(x))
        x = x + self.ff(self.layernorm2(x))
        return x

    