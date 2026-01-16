import joblib
import numpy as np
from pathlib import Path
from scipy.sparse import hstack

from data_loader import DataLoader
from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor


def main():
    # ---------- Load model & vectorizer ----------
    model = joblib.load("../results/logistic_model.joblib")
    tfidf = joblib.load("../results/tfidf_vectorizer.joblib")

    # ---------- Load feature resources ----------
    loader = DataLoader()
    tokenizer = Tokenizer()
    pos_words, neg_words = loader.load_lexicon()
    extractor = FeatureExtractor(tokenizer, pos_words, neg_words)

    # ---------- Read challenge data ----------
    reviews = Path("../dataset/challenge_data.txt").read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(reviews) == 5000, "Input file must contain exactly 5000 reviews"

    predictions = []

    for review in reviews:
        # TF-IDF
        X_tfidf = tfidf.transform([review])

        # Handcrafted
        X_hand = np.array([extractor.extract(review)])

        # Combine
        X_final = hstack([X_tfidf, X_hand])

        # Predict
        label = model.predict(X_final)[0]
        predictions.append(str(label))

    # ---------- Write output ----------
    output = "".join(predictions)


    Path("../challenge_result.txt").write_text(output, encoding="utf-8")

    print("Submission file created: challenge_result.txt")
    print("First 20 characters:", output[:20])


if __name__ == "__main__":
    main()
