# Experiment Plan

## 1. Models
- **Baseline Model:** TF-IDF feature extraction combined with a Logistic Regression classifier.
- **Deep Learning Model:** BERTurk pretrained model equipped with 7 distinct output heads for multi-aspect classification.

## 2. Evaluation Metrics
- Macro-F1 score per aspect.
- Overall Macro-F1 score across all aspects.
- Confusion matrix for each aspect to analyze misclassifications.

## 3. Data Split
- The dataset will be split into **70% Train / 15% Validation / 15% Test**.
- A fixed seed will be used across all splits to ensure reproducibility.

## 4. Experiment Tracking
For every experimental run, the following information must be recorded:
- The model used (architecture and baseline details).
- Data version and split configuration.
- Hyperparameter settings (learning rate, batch size, etc.).
- Evaluation scores (as defined in the metrics section).
