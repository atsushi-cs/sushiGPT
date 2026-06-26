class BasicTokenizer:
    def train(self, text, vocab_size, verbose=False):
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))

        def get_stats(ids):
            counts = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            return counts

        def merge(ids, pair, idx):
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
        
        num_merges = vocab_size - 256
        merges = {}
        
        for i in range(num_merges):
            stats = get_stats(tokens)
            pair = max(stats, key= stats.get)
            idx = 256 + i
            if verbose:
                print(f"merging {pair} into a new token {idx}")
            tokens = merge(tokens, pair, idx)
            merges[pair] = idx
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
            
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        pass
    def decode(self, ids):
        pass


