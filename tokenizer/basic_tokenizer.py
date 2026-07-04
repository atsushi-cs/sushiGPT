class BasicTokenizer:
    def train(self, text, vocab_size, verbose=False):
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))

        num_merges = vocab_size - 256
        merges = {}
        
        for i in range(num_merges):
            stats = self.get_stats(tokens)
            pair = max(stats, key= stats.get)
            idx = 256 + i
            if verbose:
                print(f"merging {pair} into a new token {idx}")
            tokens = self.merge(tokens, pair, idx)
            merges[pair] = idx
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    
    def decode(self, ids):
        #given ids (list of ints), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors='replace')
        return text

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
    # in the list of ints(ids), replace all consecutive occurences of pair with the new token idx
        new_ids = []
        i = 0
        while i < len(ids):
            # if we are not at the very last position AND the pair matches, replace it
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

import regex as re
