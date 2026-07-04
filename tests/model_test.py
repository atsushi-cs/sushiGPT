import torch
from model.attention import Attention

def test_attention_output_shape():
    head_size = 64
    x = torch.randn(4, 32, 384)  # (batch, seq_len, n_embd)
    attn = Attention(n_embd=384, head_size=head_size)
    out = attn(x)
    assert out.shape == (4, 32, head_size)