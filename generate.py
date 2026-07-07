import torch
import tiktoken
from model.gpt import GPT

enc = tiktoken.get_encoding("gpt2")
model = GPT()
model.load_state_dict(torch.load("weights/sushiGPT_gpt2.pt"))
model.eval()

prompt = "hi how are you?"
ids = enc.encode(prompt)
x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

out = model.generate(x, max_new_tokens=200)
print(enc.decode(out[0].tolist()))