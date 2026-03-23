# Movie Review Sentiment Analyzer

> **CSA2001 – Fundamentals of AI and ML | BYOP Project**  
> Aadi Jain · 25BCE10331 · B.Tech CSE, First Year · VIT Bhopal University



## The Problem

Every day, millions of movie reviews, product ratings, and social media posts are written online. Reading all of them manually to understand whether the audience is happy or unhappy is impossible at scale. **Can a machine learn to understand the sentiment (positive or negative) expressed in free-form text?**

This project builds a complete NLP + Neural Network pipeline to classify movie reviews as **POSITIVE** or **NEGATIVE** with over **91% accuracy**.



## What This Project Does

A full text classification pipeline:

1. **NLP Preprocessing** — lowercase, HTML removal, punctuation stripping, stopword filtering, tokenisation
2. **TF-IDF Vectorisation** — converts cleaned text into a numerical feature matrix (unigrams + bigrams)
3. **Three Models Trained and Compared:**

| Model | Role |
|-------|------|
| Naïve Bayes | Classic NLP baseline; works well with TF-IDF |
| Logistic Regression | Strong linear baseline for text |
| **MLP Neural Network** | 3-hidden-layer network (256→128→64→output) |

4. **Full Evaluation** — Accuracy, AUC-ROC, 5-fold CV, confusion matrix, classification report
5. **Live Prediction Demo** — Predict sentiment of any new review string



## AI/ML Concepts Applied

- **Natural Language Processing (NLP):** text cleaning, tokenisation, stopword removal, TF-IDF with n-grams
- **Neural Networks:** Multi-Layer Perceptron (MLP) with ReLU activations, Adam optimiser, early stopping
- **Supervised Learning (Classification):** binary sentiment classification
- **Model Evaluation:** Accuracy, Precision, Recall, F1-score, AUC-ROC, Confusion Matrix
- **Cross-Validation:** 5-fold Stratified CV to detect overfitting
- **Feature Engineering:** TF-IDF with bigrams (captures phrases like "not good", "very bad")



## Results

| Model | Accuracy | AUC-ROC | CV Accuracy |
|-------|----------|---------|-------------|
| Naïve Bayes | 91.50% | 0.909 | 0.921 ± 0.015 |
| Logistic Regression | 91.50% | 0.909 | 0.921 ± 0.015 |
| **MLP Neural Network** | **91.50%** | **0.908** | **0.920 ± 0.016** |

All three models achieve ~91.5% accuracy. The MLP Neural Network matches the baselines, demonstrating that the NLP preprocessing stage (not just model complexity) is the key driver of performance in text classification tasks.



## Project Structure

```
sentiment-analyzer/
│
├── sentiment_analyzer.py    ← Full pipeline (NLP + 3 models + evaluation)
├── reviews.csv              ← Labelled dataset (2000 reviews, auto-generated)
├── requirements.txt         ← Python dependencies
├── plots/
│   ├── 01_eda.png           ← Class balance, review length, top words
│   ├── 02_preprocessing.png ← Top words before vs after preprocessing
│   ├── 03_confusion_matrices.png  ← Confusion matrix for all 3 models
│   ├── 04_roc_curves.png    ← ROC curves with AUC scores
│   ├── 05_model_comparison.png    ← Accuracy + AUC bar charts
│   └── 06_nn_architecture.png     ← MLP layer diagram
└── README.md                ← This file
```



## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/aadi25bce10331-netizen/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python sentiment_analyzer.py
```

The script will:
- Load (or auto-generate) the dataset
- Run the full NLP preprocessing pipeline
- Train and evaluate all three models
- Print metrics to terminal
- Save 6 plots to `plots/`
- Run a live prediction demo on 3 sample reviews

### 4. Predict your own review
At the bottom of `sentiment_analyzer.py`, add your review to the `demo_reviews` list and re-run.



## No Internet or Extra Downloads Required

The NLP preprocessing is implemented from scratch (no NLTK download needed). The dataset is auto-generated if `reviews.csv` is not present. The project runs fully offline.



## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```
Python 3.8 or higher.



## Author

**Aadi Jain**  
Registration No: 25BCE10331  
B.Tech Computer Science and Engineering, First Year  
VIT Bhopal University  
Course: CSA2001 – Fundamentals of AI and ML




