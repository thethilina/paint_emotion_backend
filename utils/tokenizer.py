import re
from collections import defaultdict

class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}

    def build_vocab(self, texts):
        word_freq = defaultdict(int)

        for text in texts:
            words = self.clean(text).split()
            for w in words:
                word_freq[w] += 1

        idx = 2
        for word in word_freq:
            if word not in self.vocab:
                self.vocab[word] = idx
                idx += 1

    def clean(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text

    def encode(self, text, max_len=30):
        words = self.clean(text).split()

        tokens = [self.vocab.get(w, 1) for w in words]  # 1 = UNK

        # padding
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        return tokens