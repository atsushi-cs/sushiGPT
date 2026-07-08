from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from config import block_size
import tiktoken
import torch

class AlpacaDataset(Dataset):
    def __init__(self):
        super().__init__()
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        self.data = []
        enc = tiktoken.get_encoding('gpt2')

        def format_example(example):
            if example['input']:
                return f"User: {example['instruction']}\n{example['input']}\nAssistant: "
            else:
                return f"User: {example['instruction']}\nAssistant: "
        for example in ds:
            prompt = format_example(example)
            full = prompt + example['output']

            prompt_ids = enc.encode(prompt)
            full_ids = enc.encode(full)
            
            # if len(full_ids) > 512: (Only if your device is not powerful enough)
            #     continue
            
            prompt_len = len(prompt_ids)

            labels = [-100] * prompt_len + full_ids[prompt_len:]

            pair = (full_ids, labels)

            if len(full_ids) <= block_size:
                self.data.append(pair)

            
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_ids, labels = self.data[index]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
