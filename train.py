from model.gpt import GPT
import torch

model = GPT()
x = torch.randint(0, 512, (4, 256))  # (batch_size, seq_len)
logits = model(x)
print(logits.shape) 
