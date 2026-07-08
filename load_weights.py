from transformers import GPT2LMHeadModel
import torch
from model.gpt import GPT
from config import n_layers, n_heads, n_embd, vocab_size
import numpy as np

gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

model = GPT()

def load_gpt2_weights(model, gpt2_state_dict):
    sd = model.state_dict()
    head_dim = n_embd // n_heads

    sd['tokens.weight'] = gpt2_state_dict['transformer.wte.weight']
    sd['positions.weight'] = gpt2_state_dict['transformer.wpe.weight']
    sd['ln.weight'] = gpt2_state_dict['transformer.ln_f.weight'] 
    sd['ln.bias'] = gpt2_state_dict['transformer.ln_f.bias']

    for i in range(n_layers):
        sd[f'blocks.{i}.layernorm1.weight'] = gpt2_state_dict[f'transformer.h.{i}.ln_1.weight']
        sd[f'blocks.{i}.layernorm1.bias']   = gpt2_state_dict[f'transformer.h.{i}.ln_1.bias']   
        sd[f'blocks.{i}.layernorm2.weight'] = gpt2_state_dict[f'transformer.h.{i}.ln_2.weight']
        sd[f'blocks.{i}.layernorm2.bias']   = gpt2_state_dict[f'transformer.h.{i}.ln_2.bias']

        sd[f'blocks.{i}.ff.net.0.weight']   = gpt2_state_dict[f'transformer.h.{i}.mlp.c_fc.weight'].T
        sd[f'blocks.{i}.ff.net.0.bias']     = gpt2_state_dict[f'transformer.h.{i}.mlp.c_fc.bias']
        sd[f'blocks.{i}.ff.net.2.weight']   = gpt2_state_dict[f'transformer.h.{i}.mlp.c_proj.weight'].T
        sd[f'blocks.{i}.ff.net.2.bias']     = gpt2_state_dict[f'transformer.h.{i}.mlp.c_proj.bias']

        Q = gpt2_state_dict[f'transformer.h.{i}.attn.c_attn.weight'][:, :n_embd]
        K = gpt2_state_dict[f'transformer.h.{i}.attn.c_attn.weight'][:, n_embd: n_embd * 2]
        V = gpt2_state_dict[f'transformer.h.{i}.attn.c_attn.weight'][: , n_embd * 2:]

        Q_b = gpt2_state_dict[f'transformer.h.{i}.attn.c_attn.bias'][:n_embd]
        K_b = gpt2_state_dict[f'transformer.h.{i}.attn.c_attn.bias'][n_embd :n_embd * 2]
        V_b = gpt2_state_dict[f'transformer.h.{i}.attn.c_attn.bias'][n_embd * 2:]

        for j in range(n_heads):
            start = j * head_dim
            end = (j + 1) * head_dim
            sd[f'blocks.{i}.attn.heads.{j}.q.weight'] = Q[:, start:end].T
            sd[f'blocks.{i}.attn.heads.{j}.q.bias']   = Q_b[ start:end]
            sd[f'blocks.{i}.attn.heads.{j}.k.weight'] = K[:,  start:end].T
            sd[f'blocks.{i}.attn.heads.{j}.k.bias']   = K_b[ start:end]
            sd[f'blocks.{i}.attn.heads.{j}.v.weight'] = V[:,  start:end].T
            sd[f'blocks.{i}.attn.heads.{j}.v.bias']   = V_b[ start:end]

        sd[f'blocks.{i}.attn.projection.weight'] = gpt2_state_dict[f'transformer.h.{i}.attn.c_proj.weight'].T
        sd[f'blocks.{i}.attn.projection.bias'] = gpt2_state_dict[f'transformer.h.{i}.attn.c_proj.bias']
        sd['output_head.weight'] = gpt2_state_dict['transformer.wte.weight']
        sd['output_head.bias'] = torch.zeros(vocab_size)
        print(f'layer{i + 1}/{n_layers} complete!')

    model.load_state_dict(sd)
    
load_gpt2_weights(model, gpt2.state_dict())

SAVE_PATH = "weights/sushiGPT_gpt2.pt"

torch.save(model.state_dict(), SAVE_PATH)

print('saved')