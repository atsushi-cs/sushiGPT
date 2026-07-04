import torch
import tiktoken
from model.gpt import GPT

enc = tiktoken.get_encoding("gpt2")
model = GPT()
model.load_state_dict(torch.load("model/sushiGPT.pt"))
model.eval()

prompt = "and	to	Athens	of	the	wide	ways,	and	entered	the"
ids = enc.encode(prompt)
x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

out = model.generate(x, max_new_tokens=200)
print(enc.decode(out[0].tolist()))