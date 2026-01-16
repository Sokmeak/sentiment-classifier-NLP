import numpy as np
import joblib
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


from data_loader import DataLoader
from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor
from models import Models
from evaluator import Evaluator

def main():
    print("Step1: Loading data...")
    loader = DataLoader()
    time.sleep(2)
    
    
    print("Step2: Initializing tokenizer...")
    tokenizer = Tokenizer()
    time.sleep(2)


    print("Step3: Preparing dataset for training and testing...")
    X_train_text, y_train, X_test_text, y_test = loader.load_reviews()
    
    # let try build a TF-IDF feature extractor
    print("Step3.1: Creating TF-IDF features...") 
    
    tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),      # unigrams + bigrams
    min_df=3,                # ignore rare words
    max_df=0.9,              # ignore very common words
    max_features=5000        # keep it reasonable
    )
    
    

    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)


    joblib.dump(tfidf, "../results/tfidf_vectorizer.joblib")

    print("Step3.2: Loading lexicon...") 
    pos_words, neg_words = loader.load_lexicon()
    time.sleep(2)

    print("Step4: Extracting features...")
    extractor = FeatureExtractor(tokenizer, pos_words, neg_words)
    time.sleep(2)


    print("Step5: Extracting training and testing features...")
    X_train_hand = np.array([extractor.extract(t) for t in X_train_text])
    X_test_hand = np.array([extractor.extract(t) for t in X_test_text])
    
    
    X_train = hstack([X_train_tfidf, X_train_hand])
    X_test = hstack([X_test_tfidf, X_test_hand])

    time.sleep(2)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    time.sleep(2)

    print("Step6: Initializing models...")
    log_model = Models.logistic()
    rf_model = Models.random_forest()
    time.sleep(2)

    print("Step7.1: Training models logistic regression...")
    log_model.fit(X_train, y_train)
    time.sleep(2)
    
    print("Step7.2: Training models random forest...")
    rf_model.fit(X_train, y_train)
    time.sleep(2)

    print("Step8.1: Evaluating Logistic Regression...")
    Evaluator.evaluate(log_model, X_test, y_test, "Logistic Regression", "../results/logistic_evaluation.txt")
    time.sleep(2)
    
    print("Step8.2: Evaluating Random Forest...")
    Evaluator.evaluate(rf_model, X_test, y_test, "Random Forest", "../results/rf_evaluation.txt")
    time.sleep(2)

    print("Step9: Saving models results...")
    joblib.dump(log_model, "../results/logistic_model.joblib")
    joblib.dump(rf_model, "../results/rf_model.joblib")
    time.sleep(2)

    print("Models saved to results/")

if __name__ == "__main__":
    main()
