# DARKGUARD — Slide Outline

## 1. Title
### DARKGUARD
AI-powered Dark Pattern Detector

**Tagline:** Detect manipulative website language using NLP and machine learning.

## 2. Problem
Online platforms can use language that creates artificial urgency, perceived scarcity, or social pressure.

Examples:
- "Only 2 seats left!"
- "Offer ends in 5 minutes!"
- "127 people are viewing this!"

## 3. Solution
DARKGUARD analyzes a text snippet and predicts:

**Dark Pattern** or **Not Dark Pattern**

It also shows model confidence and the features that influenced the decision.

## 4. ML Pipeline
Website text → Cleaning → TF-IDF → Logistic Regression → Prediction

## 5. Prototype
Show:
- DARKGUARD interface
- Example selector
- Analysis result
- Confidence bar
- Model explanation
- Pattern signal scan

## 6. Baseline Results
- Dataset: 2,356 samples
- Split: 80/20 stratified
- Model: TF-IDF + Logistic Regression
- Test accuracy: approximately 93.4%

## 7. Competition Challenge
Participants are not starting from an empty project.

They receive a working baseline and improve the ML.

Example challenges:
- Improve accuracy
- Reduce false negatives
- Improve short-text handling
- Add multi-category classification
- Improve explanations

## 8. Tech Stack
Python • Pandas • NumPy • scikit-learn • TF-IDF • Logistic Regression • Streamlit • GitHub

## 9. Contribution
Open GitHub Issues contain the participant tasks and acceptance criteria.
