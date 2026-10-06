# 🛡️ DARKGUARD

**AI-powered Dark Pattern Detector**

DARKGUARD is a Streamlit machine-learning prototype that analyzes website text and predicts whether it contains a dark pattern.

## Problem

Online platforms sometimes use wording that creates artificial pressure or nudges users toward a decision. Examples include fake urgency, scarcity, and social proof.

## What the prototype does

A user pastes website text and DARKGUARD returns:

- Dark Pattern / Not Dark Pattern
- Model confidence
- Top TF-IDF features that influenced the prediction
- A separate quick scan for urgency, scarcity, and social-proof signals
- Basic model and dataset insights

## ML Pipeline

```
Website Text
    ↓
Basic Text Cleaning
    ↓
TF-IDF
    ↓
Logistic Regression
    ↓
Dark Pattern / Not Dark Pattern
```

## Baseline

The current baseline uses:

- 80/20 stratified train/test split
- TF-IDF with 1- and 2-word features
- Logistic Regression

The prototype achieved approximately **93.4% test accuracy** on the current dataset under this split.

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## Dataset

The dataset contains website snippets with a binary label and a pattern category.

Dataset source: [Yamanalab/ec-darkpattern](https://github.com/yamanalab/ec-darkpattern)

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Contribution Areas

Open issues are used as competition challenges. Current directions include:

- Improve baseline accuracy
- Reduce false negatives
- Improve short-text classification
- Add multi-category detection
- Improve phrase-level explanations

For ML changes, contributors should report results on held-out data and include accuracy, precision, recall, F1-score, and a short explanation of the change.

## Project Structure

```
darkguard/
├── app.py
├── model.py
├── dataset.tsv
├── requirements.txt
├── CONTRIBUTING.md
├── PRESENTATION.md
├── LICENSE
├── .gitignore
└── tests/
    └── test_model.py
```

## Note

A model prediction is an indicator produced from the training data; it is not proof of intent, deception, or harm by a website.
