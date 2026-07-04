from tokenizer.regex_tokenizer import RegexTokenizer

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tok = RegexTokenizer()
tok.train(text, vocab_size=1024, verbose=True)
tok.save("tokenizer/sushiGPT")

print(f"vocab size: {len(tok.vocab)}")
print("tokenizer saved to tokenizer/sushiGPT.model")