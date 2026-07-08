import tiktoken
import torch
from model.gpt import GPT

enc = tiktoken.get_encoding('gpt2')

model = GPT()
model.load_state_dict(torch.load("weights/sushiGPT_finetuned.pt"))
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(device)
model.eval()

user_input = "who is the main character of the Odyssey"
prompt = f"User: {user_input}\nAssistant:"
ids = enc.encode(prompt)
x = torch.tensor(ids).unsqueeze(0).to(device)
with torch.no_grad():
    out = model.generate(x, max_new_tokens=200)
response = enc.decode(out[0].tolist())

response = response[len(prompt):]

if "\nUser:" in response:
    response = response[:response.index("\nUser:")]

print(response.strip())