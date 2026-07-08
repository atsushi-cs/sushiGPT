from model.gpt import GPT
import torch
from finetune.dataset import AlpacaDataset
from torch.utils.data import DataLoader
from config import finetune_lr, num_epochs, batch_size, vocab_size
import torch.nn.functional as F

model = GPT()
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(device)

model.load_state_dict(torch.load("weights/sushiGPT_gpt2.pt"), strict= False)
model.train()

def collate_fn(batch):
    input_ids, labels = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return input_ids, labels

dataset = AlpacaDataset()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print(f"dataset size: {len(dataset)}")

optimizer = torch.optim.Adam(model.parameters(), lr = finetune_lr)

for epoch in range(num_epochs):
    accumulation_steps = 8  
    for step, (input_ids, labels) in enumerate(loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"epoch {epoch+1}, step {step}: loss {loss.item():.4f}")


torch.save(model.state_dict(), "weights/sushiGPT_finetuned.pt")
print("saved to weights/sushiGPT_finetuned.pt")