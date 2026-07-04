import torch
import torch.nn as nn
import torch.nn.functional as F
from config import vocab_size, n_embd, n_heads, n_layers, block_size, dropout
from model.block import Block

class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokens = nn.Embedding(vocab_size, n_embd)
        self.positions = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.output_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device = x.device)
        tok_emb = self.tokens(x)
        pos_emb = self.positions(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)

        return self.output_head(x)
    
    def generate(self, x, max_new_tokens):

        for _ in range(max_new_tokens):
            x = x[:, -block_size:]
            logits = self.forward(x)
            logits = logits[:, -1, :]
            logits = logits / 0.8
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

        return x
