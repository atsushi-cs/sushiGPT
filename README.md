# sushiGPT

A GPT-style language model built from scratch in PyTorch, following Andrej Karpathy's Zero to Hero curriculum. Includes a custom BPE tokenizer with regex chunking, a full transformer implementation, GPT-2 weight loading, and instruction fine-tuning.

## What's in here

- **Custom BPE tokenizer** — byte-pair encoding with regex-based chunking matching GPT-2/GPT-4 preprocessing style
- **Transformer architecture** — multi-head self-attention, feedforward blocks, residual connections, layer norm, built from scratch
- **GPT-2 weight loading** — manual weight mapping from HuggingFace's GPT-2 into this custom architecture
- **Instruction fine-tuning** — supervised fine-tuning on the Alpaca dataset with prompt loss masking
- **Generation** — autoregressive text generation with temperature and stop token support

## Project structure

```
sushiGPT/
├── data/                        # training text and encoded token ids
├── tokenizer/
│   ├── basic_tokenizer.py       # simple BPE tokenizer
│   ├── regex_tokenizer.py       # BPE with regex chunking
│   └── sushiGPT.model           # saved tokenizer
├── model/
│   ├── attention.py             # single-head self-attention
│   ├── multihead.py             # multi-head attention
│   ├── feedforward.py           # feedforward block
│   ├── block.py                 # full transformer block
│   └── gpt.py                   # full GPT model
├── finetune/
│   ├── dataset.py               # Alpaca dataset loader with prompt masking
│   ├── trainer.py               # fine-tuning loop
│   └── generate.py              # prompt → response generation
├── weights/
│   ├── sushiGPT_gpt2.pt         # GPT-2 weights mapped to this architecture
│   └── sushiGPT_finetuned.pt    # fine-tuned weights
├── config.py                    # all hyperparameters
├── train.py                     # pretraining loop
├── train_tokenizer.py           # tokenizer training script
├── load_weights.py              # GPT-2 weight mapping
└── generate.py                  # base model generation
```

## Setup

```bash
pip install torch transformers datasets tiktoken regex
```

## Usage

### Train the tokenizer

```bash
python3 train_tokenizer.py
```

Trains a BPE tokenizer on `data/input.txt` and saves it to `tokenizer/sushiGPT.model`.

### Pretrain the model

```bash
python3 train.py
```

Trains a GPT model from scratch on your dataset. Saves weights to `model/sushiGPT.pt`.

### Load GPT-2 pretrained weights

```bash
python3 load_weights.py
```

Downloads GPT-2 from HuggingFace and maps its weights into this architecture. Saves to `weights/sushiGPT_gpt2.pt`.

### Fine-tune on Alpaca

```bash
python3 -m finetune.trainer
```

Fine-tunes the GPT-2 weighted model on the Alpaca instruction dataset. Only computes loss on assistant response tokens. Saves to `weights/sushiGPT_finetuned.pt`.

### Generate from the fine-tuned model

```bash
python3 -m finetune.generate
```

Prompts the fine-tuned model and prints the response.

## Config

All hyperparameters live in `config.py`:

```python
# model architecture (GPT-2 small)
vocab_size = 50257
block_size = 1024
n_embd = 768
n_heads = 12
n_layers = 12
dropout = 0.2

# pretraining
batch_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4

# fine-tuning
finetune_lr = 3e-5
num_epochs = 3
```

## Architecture

The model follows the GPT-2 architecture closely:

- Token embeddings + learned positional embeddings
- N transformer blocks, each with:
  - Pre-norm multi-head self-attention with causal masking
  - Pre-norm feedforward MLP (4x expansion)
  - Residual connections around both sub-layers
- Final layer norm + linear output head (weight-tied with token embeddings in GPT-2)

The main architectural difference from GPT-2 is that Q, K, V projections are stored as separate per-head matrices rather than a single combined `c_attn` matrix. The weight loading script handles this mapping.

## Tokenizer

The `RegexTokenizer` uses the GPT-4 regex pattern to split text into chunks before BPE merging, preventing merges from crossing word/punctuation boundaries. Supports `train`, `encode`, `decode`, `save`, and `load`.

## Fine-tuning

Training examples are formatted as:

```
User: {instruction}
Assistant: {response}
```

Loss is only computed on assistant tokens — prompt tokens are masked with `-100` which `F.cross_entropy` ignores automatically. This teaches the model to respond rather than to predict the prompt.

## References

- [Andrej Karpathy — Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
