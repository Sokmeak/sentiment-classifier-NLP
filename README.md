# Mini Project 2 – Text Classification (Sentiment Analysis)

## Overview

This project builds a **sentiment classification** system for movie reviews.
It predicts whether a review is:

- **Positive (1)**
- **Negative (0)**

The project uses **feature engineering** (15 features) and trains two models using **scikit-learn**:

- Logistic Regression
- Random Forest

---

## Dataset Files

The dataset files are located in the `dataset/` folder:

- `positive-reviews.txt`
- `negative-reviews.txt`
- `positive-words.txt`
- `negative-words.txt`
- `challenge_data.txt`

---

## Project Structure

```
text-classification/
├── dataset/
│   ├── positive-reviews.txt
│   ├── negative-reviews.txt
│   ├── positive-words.txt
│   ├── negative-words.txt
│   └── challenge_data.txt
├── src/
│   ├── main.py
│   ├── data_loader.py
│   ├── tokenizer.py
│   ├── feature_extractor.py
│   ├── models.py
│   ├── evaluator.py
│   ├── challenge.py
│   └── test-model.py
├── results/
│   ├── logistic_model.joblib
│   ├── rf_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_evaluation.txt
│   └── rf_evaluation.txt
├── requirements.txt
└── README.md
```

---

## Installation (Local Machine)

Create and activate a virtual environment:

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Project

From the project root folder:

```bash
python src/main.py
```

The program will:

1. Load the dataset
2. Extract 15 features
3. Train Logistic Regression and Random Forest
4. Print evaluation results (accuracy, confusion matrix, classification report)

---

## Models Used

- **Logistic Regression**
- **Random Forest (Decision Tree based)**

Both models are implemented using **scikit-learn**.

---

## Output

Evaluation results will be printed in the terminal and saved to the `results/` folder.
Trained models are also saved as `.joblib` files for reuse.

---

## Author

TEAM 7
