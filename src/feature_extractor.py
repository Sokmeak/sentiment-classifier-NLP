import numpy as np

class FeatureExtractor:
    PRONOUNS = {"i", "me", "my", "you", "your"}
    NEGATIONS = {"not", "never", "n't", "no"}

    def __init__(self, tokenizer, pos_words, neg_words):
        self.tokenizer = tokenizer
        self.pos_words = pos_words
        self.neg_words = neg_words

#  extract features from a given text (15 features)
    def extract(self, text):
        tokens = self.tokenizer.tokenize(text)
        total_words = len(tokens)

        pos_count = sum(t in self.pos_words for t in tokens)
        neg_count = sum(t in self.neg_words for t in tokens)

        has_no = 1 if "no" in tokens else 0
        pron_count = sum(t in self.PRONOUNS for t in tokens)

        pos_ratio = pos_count / (total_words + 1)
        neg_ratio = neg_count / (total_words + 1)

        negation_count = sum(t in self.NEGATIONS for t in tokens)

        neg_near_sentiment = 0
        for i in range(len(tokens) - 1):
            if tokens[i] in self.NEGATIONS and (
                tokens[i+1] in self.pos_words or tokens[i+1] in self.neg_words
            ):
                neg_near_sentiment = 1
                break

        has_excl = 1 if "!" in text else 0
        excl_count = text.count("!")
        q_count = text.count("?")

        log_len = np.log(1 + total_words)
        polarity_score = pos_count - neg_count
        pos_neg_ratio = (pos_count + 1) / (neg_count + 1)
        avg_word_len = sum(len(t) for t in tokens) / (total_words + 1)
        

        return [
            pos_count, neg_count, has_no, pron_count,
            pos_ratio, neg_ratio,
            negation_count, neg_near_sentiment,
            has_excl, excl_count, q_count,
            log_len, polarity_score, pos_neg_ratio, avg_word_len
        ]
