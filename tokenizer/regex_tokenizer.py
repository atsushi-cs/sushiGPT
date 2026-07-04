import regex as re

class RegexTokenizer:
    pat = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def train(self, text, vocab_size, verbose=False):
        #separate based off pattern, then encode each word
        text = re.findall(self.pat, text)
        tokens = [item.encode('utf-8') for item in text]
        tokens = [list(map(int, item)) for item in tokens]

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
        text = re.findall(self.pat, text)
        tokens = [item.encode('utf-8') for item in text]
        tokens = [list(map(int, item)) for item in tokens]
        
        while True:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)

        tokens = sum(tokens, [])

        return tokens
    
    def decode(self, ids):
        # given ids (list of ints), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors='replace')
        return text

    def get_stats(self, ids):
        counts = {}
        for word in ids: # instead of it being across a giant list, do it for each chunk
            for pair in zip(word, word[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
    # in the list of ints(ids), replace all consecutive occurences of pair with the new token idx
        new_ids = []

        for item in ids:
            i = 0 # resets per word
            chunk_ids = [] # per chunk merging vs whole list, preserving chunks
            while i < len(item):
                # if we are not at the very last position AND the pair matches, replace it
                if i < len(item) - 1 and item[i] == pair[0] and item[i+1] == pair[1]:
                    chunk_ids.append(idx)
                    i += 2
                else:
                    chunk_ids.append(item[i])
                    i += 1

            new_ids.append(chunk_ids)

        return new_ids
    
    def save(self, file_prefix):
        with open(file_prefix + ".model", "w") as f:
            f.write(self.pat + "\n") 
            for (p0, p1), idx in self.merges.items():
               f.write(f"{p0} {p1} {idx}\n")

    def load(self, model_file):
        merges = {}
        with open(model_file, "r") as f:
            self.pat = f.readline().strip()
            for line in f:
                p0, p1, idx = map(int, line.split())
                merges[(p0, p1)] = idx
        self.merges = merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab