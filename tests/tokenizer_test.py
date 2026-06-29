from tokenizer import BasicTokenizer
from tokenizer import RegexTokenizer

with open("data/taylorswift.txt", "r") as f:
    text = f.read()

def test_train_vocab_size():
    tok = BasicTokenizer()
    tok.train(text, vocab_size=300, verbose= True)
    assert len(tok.vocab) == 300

def test_train_merges_most_frequent_pair_first():
    tok = BasicTokenizer()
    tok.train("aaaabbbb", vocab_size=257 + 1)  # one merge
    first_merge = list(tok.merges.keys())[0]
    assert first_merge == (97, 97)  # 'a','a' byte values

def test_train_is_deterministic():
    tok1, tok2 = BasicTokenizer(), BasicTokenizer()
    tok1.train(text, 300)
    tok2.train(text, 300)
    assert tok1.merges == tok2.merges

def test_roundtrip_on_unseen_unicode_text():
    tok = RegexTokenizer()
    tok.train(text, vocab_size=300)  # trained separately

    test = "hello world!!!? (안녕하세요!) lol123 😉"
    ids = tok.encode(test)
    decoded = tok.decode(ids)

    assert decoded == test