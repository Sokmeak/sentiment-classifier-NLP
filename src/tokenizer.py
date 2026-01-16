import re

class Tokenizer:
    # in the form of \bword\b. A "word character"
    token_re = re.compile(r"\b\w+\b") 

    def tokenize(self, text):
        return self.token_re.findall(text.lower())
