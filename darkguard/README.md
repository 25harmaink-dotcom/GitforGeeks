# 🛡️ DARKGUARD

AI-powered Dark Pattern Detector

DARKGUARD is a machine-learning prototype that analyzes website language and predicts whether a text snippet contains a dark pattern.

## Problem

Shopping, booking, and other online platforms can use manipulative wording such as fake urgency, scarcity, and social proof. DARKGUARD is designed to identify these patterns automatically from text.

## Solution

Website text → basic cleaning → TF-IDF → Logistic Regression → Dark Pattern / Not Dark Pattern

## Prototype features

- Dark-pattern prediction
- Confidence score
- Model-based feature contribution explanation
- Rule-based signal scan for urgency, scarcity, and social proof
- Model and dataset insights
- Baseline confusion matrix

## Baseline

The baseline uses an 80/20 stratified train/test split with random_state=42.

Current prototype test accuracy is approximately 93.4% on the supplied dataset.

## Tech stack

- Python
- Pandas
- NumPy
- scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## Dataset

The project includes dataset.tsv with website snippets, binary labels, and pattern categories.

## Run locally

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

## Contribution challenges

The baseline is intentionally open to improvement. See the GitHub Issues for:

- Improving baseline accuracy
- Investigating false negatives
- Improving short-text performance
- Multi-category detection
- Better phrase-level explanations

For ML changes, contributors should report accuracy, precision, recall, and F1-score on an appropriate held-out test set.

## Project structure

    app.py
    model.py
    dataset.tsv
    requirements.txt
    CONTRIBUTING.md
    LICENSE
    PRESENTATION.md
    tests/test_model.py
