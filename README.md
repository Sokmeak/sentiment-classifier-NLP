Here is a simple `README.md` you can use:

```md
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
Place these files inside the `data/` folder:

- `positive-reviews.txt`
- `negative-reviews.txt`
- `positive-words.txt`
- `negative-words.txt`

---

## Project Structure
```

mini_project_2/
│── data/
│── src/
│   └── main.py
│── results/
│── requirements.txt
│── .gitignore
└── README.md

````

---

## Installation (Local Machine)
Create and activate a virtual environment:

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
````

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

* **Logistic Regression**
* **Random Forest (Decision Tree based)**

Both models are implemented using **scikit-learn**.

---

## Output

Evaluation results will be printed in the terminal.
(Optional) Saved models can be stored inside the `results/` folder.

---

## Author

TEAM 7

```

If you want, I can update it with your **real folder names**, and add a section for **15 features list** inside the README too.
```

