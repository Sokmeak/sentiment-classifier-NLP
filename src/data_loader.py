from pathlib import Path
import math

class DataLoader:
    def __init__(self, data_dir="../dataset"):
        self.data_dir = Path(data_dir)

    def read_lines(self, filename):
        return (self.data_dir / filename).read_text(
            encoding="utf-8"
        ).splitlines()

    def split_top_80(self, lines):
        cut = int(math.floor(0.8 * len(lines)))
        return lines[:cut], lines[cut:]

    def load_reviews(self):
        pos = self.read_lines("positive-reviews.txt")
        neg = self.read_lines("negative-reviews.txt")

        pos_train, pos_test = self.split_top_80(pos)
        neg_train, neg_test = self.split_top_80(neg)

        X_train = pos_train + neg_train
        y_train = [1]*len(pos_train) + [0]*len(neg_train)

        X_test = pos_test + neg_test
        y_test = [1]*len(pos_test) + [0]*len(neg_test)

        return X_train, y_train, X_test, y_test

    def load_lexicon(self):
        pos_words = set(self.read_lines("positive-words.txt"))
        neg_words = set(self.read_lines("negative-words.txt"))
        return pos_words, neg_words
