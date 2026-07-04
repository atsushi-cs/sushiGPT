import torch
import torch.nn.functional as F
from model.gpt import GPT
import tiktoken
from config import block_size, vocab_size, batch_size, max_iters, learning_rate, eval_interval
import os

model = GPT()
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

if os.path.exists("data/ids.pt"):
    ids = torch.load("data/ids.pt")
else:
    print("encoding text...")
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor(enc.encode(text), dtype=torch.long)
    torch.save(ids, "data/ids.pt")
    print(f"done — {len(ids)} tokens")

n = int(0.9 * len(ids))
train_data = ids[:n]
val_data = ids[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

for step in range(max_iters):
    x, y = get_batch('train')
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"step {step}: loss {loss.item():.4f}")