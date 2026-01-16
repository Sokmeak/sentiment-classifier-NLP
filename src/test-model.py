import joblib
import numpy as np
from scipy.sparse import hstack

from data_loader import DataLoader
from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor


def predict_sentence(model, tfidf, extractor, sentence, threshold=0.5):
    # TF-IDF features
    X_tfidf = tfidf.transform([sentence])

    # Handcrafted features
    X_hand = np.array([extractor.extract(sentence)])

    # Combine
    X_final = hstack([X_tfidf, X_hand])

    prob = model.predict_proba(X_final)[0][1]
    label = 1 if prob >= threshold else 0

    return label, prob


def main():
    print("Loading hybrid model and vectorizer...")

    model = joblib.load("../results/logistic_model.joblib")
    tfidf = joblib.load("../results/tfidf_vectorizer.joblib")

    loader = DataLoader()
    tokenizer = Tokenizer()
    pos_words, neg_words = loader.load_lexicon()
    extractor = FeatureExtractor(tokenizer, pos_words, neg_words)

    print("\nType a sentence (or 'quit' to exit)\n")

    while True:
        sentence = input(">> ")

        if sentence.lower() == "quit":
            break

        label, prob = predict_sentence(
            model, tfidf, extractor, sentence, threshold=0.5
        )

        sentiment = "POSITIVE" if label == 1 else "NEGATIVE"

        print(f"Sentiment: {sentiment}")
        print(f"Confidence (positive): {prob:.3f}\n")


if __name__ == "__main__":
    main()
