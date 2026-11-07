class SimpleTokenizer:
    def __init__(self, text):
        words = text.lower().split()
        vocab = sorted(set(words))

        self.itos = ["<pad>", "<unk>"] + vocab
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [
            self.stoi.get(word, self.unk_id)
            for word in text.lower().split()
        ]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)
