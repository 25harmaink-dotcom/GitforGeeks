# DARKGUARD — Presentation

## Slide 1 — DARKGUARD
AI-powered Dark Pattern Detector

## Slide 2 — The Problem
Online platforms can use manipulative wording to create artificial pressure or influence, including fake urgency, scarcity, and social proof.

## Slide 3 — The Solution
DARKGUARD accepts website text and predicts whether it contains a dark pattern.

## Slide 4 — ML Pipeline
Website Text → Cleaning → TF-IDF → Logistic Regression → Prediction

## Slide 5 — Prototype
Show the Streamlit detector with input text, prediction, confidence, explanation, and signal scan.

## Slide 6 — Baseline Results
Dataset: 2,356 samples
Train/test split: 80/20
Model: TF-IDF + Logistic Regression
Test accuracy: approximately 93.4%

## Slide 7 — Contribution Challenge
Participants improve the baseline model on unseen data.

Challenges:
- Improve accuracy
- Reduce false negatives
- Improve short-text handling
- Add category-level detection
- Improve explanations

## Slide 8 — Tech Stack
Python, Pandas, NumPy, scikit-learn, TF-IDF, Logistic Regression, Streamlit, GitHub
