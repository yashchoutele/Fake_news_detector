---
title: Shield — Fake News Detector
emoji: 🛡️
colorFrom: green
colorTo: gray
sdk: streamlit
sdk_version: "1.45.1"
app_file: app/app.py
pinned: false
---

# 🔍 Automated Fake News Detection System
### NLP · TF-IDF · RoBERTa (HF Inference API) · Streamlit · RTX 4060 GPU

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.1-red)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-green)](https://streamlit.io)
[![Live App](https://img.shields.io/badge/🚀_Live_App-shield--fakenewsdetection.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://shield-fakenewsdetection.streamlit.app/)

---

## 🌐 Live Demo

> **Try it now →** [https://shield-fakenewsdetection.streamlit.app/](https://shield-fakenewsdetection.streamlit.app/)

No setup required — paste any news article and get an instant AI-powered verdict with confidence scores, word-level explanations, sentiment analysis, and linguistic feature breakdowns.

---

## 📁 Project Structure

```
fake-news-detector/
├── data/
│   ├── raw/              ← Place train.csv here (Kaggle dataset)
│   └── processed/        ← Auto-generated cleaned data + feature matrices
├── notebooks/
│   ├── 01_eda.py         ← Exploratory Data Analysis
│   ├── 02_preprocessing.py
│   ├── 03_feature_extraction.py
│   ├── 04_ml_models.py   ← LR, NB, SVM, RF, XGBoost
│   └── 05_bert.py        ← DistilBERT fine-tuning experiments (GPU)
├── src/
│   ├── preprocessing.py  ← Text cleaning pipeline
│   ├── features.py       ← TF-IDF, embeddings, sentiment
│   ├── model.py          ← ML + BERT training code
│   └── utils.py          ← Logging, plotting, data loading
├── app/
│   ├── app.py            ← Streamlit web app
│   └── pages/
│       ├── 1_Privacy_Policy.py
│       └── 2_Terms_and_Conditions.py
├── models/               ← Saved .pkl + BERT checkpoints
├── reports/figures/      ← Auto-generated plots
├── requirements.txt
├── setup.bat             ← One-click setup (Windows)
└── README.md
```

---

## 🚀 Quick Start

### Step 1: Setup Environment

```bash
# Run the setup script (creates venv, installs all packages)
setup.bat
```

Or manually:
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 2: Download Dataset

1. Go to https://www.kaggle.com/competitions/fake-news/data
2. Download `train.csv`
3. Place it in `data/raw/train.csv`

### Step 3: Run Notebooks in Order

Open notebooks in VS Code (install **Jupyter** extension) and run cell by cell:

```
notebooks/01_eda.py              → Explore the data
notebooks/02_preprocessing.py   → Clean and preprocess text
notebooks/03_feature_extraction.py → TF-IDF + feature engineering
notebooks/04_ml_models.py       → Train 5 ML models
notebooks/05_bert.py            → DistilBERT fine-tuning experiments (GPU)
```

### Step 4: Launch Web App

```bash
venv\Scripts\activate
streamlit run app/app.py
```

---

## 📊 Expected Results

| Model | Accuracy | F1-Score | AUC-ROC | Speed | Mode |
|-------|----------|----------|---------|-------|------|
| Complement NB | ~94% | ~0.94 | ~0.97 | ⚡ <1s | Fast |
| Logistic Regression | ~97% | ~0.97 | ~0.99 | ⚡ <1s | Fast |
| Linear SVC | ~97% | ~0.97 | ~0.99 | ⚡ <1s | Fast |
| XGBoost (GPU) | ~96% | ~0.96 | ~0.98 | ⚡ <5s | Fast |
| Voting Ensemble | ~98% | ~0.98 | ~0.99 | ⚡ <2s | Fast |
| **RoBERTa** ([hamzab/roberta-fake-news-classification](https://huggingface.co/hamzab/roberta-fake-news-classification)) | **~97%** | **~0.97** | **~0.99** | 🌐 ~2–3s/article | Accurate (HF API) |

---

## 🧠 NLP Concepts Covered

| Concept | Where Used |
|---------|-----------|
| Tokenization | `src/preprocessing.py` |
| Stop Word Removal | `src/preprocessing.py` |
| Lemmatization (NLTK + spaCy) | `src/preprocessing.py` |
| TF-IDF (with bigrams) | `src/features.py` |
| Bag of Words | `src/features.py` |
| VADER Sentiment Analysis | `src/features.py` |
| Linguistic Feature Engineering | `src/features.py` |
| Logistic Regression | `src/model.py` |
| Naive Bayes (Complement) | `src/model.py` |
| Support Vector Machine | `src/model.py` |
| Random Forest | `src/model.py` |
| XGBoost (GPU) | `src/model.py` |
| Soft Voting Ensemble | `src/model.py` |
| Transformer (RoBERTa via HF Inference API) | `app/app.py` |
| Transfer Learning (Pre-trained HF model) | `app/app.py` |
| Cross-Validation (Stratified K-Fold) | `notebooks/04_ml_models.py` |
| ROC-AUC, F1, Precision, Recall | `notebooks/04_ml_models.py` |

---

## 🖥️ Web App Features

- **Dual mode**: Fast (TF-IDF + Logistic Regression, instant) or Accurate ([RoBERTa via HF Inference API](https://huggingface.co/hamzab/roberta-fake-news-classification), ~2–3s)
- **Confidence gauge**: Visual probability display
- **Word explanation**: Shows which words push toward FAKE/REAL (LR mode)
- **Sentiment breakdown**: VADER negative/neutral/positive/compound scores
- **Linguistic analysis**: Capital ratio, exclamation marks, vocabulary richness
- **Example articles**: Load sample fake/real articles instantly

---

## ⚙️ GPU Info

This project was trained locally on an **NVIDIA RTX 4060 Laptop GPU (8GB VRAM)**:
- PyTorch CUDA 12.1 for ML model training
- XGBoost GPU training via `device="cuda"`
- DistilBERT fine-tuning experiments: `batch_size=16`, `max_length=256` → fits in 8GB
- Mixed precision (`torch.cuda.amp`) → 2x training speedup

> **Live app** uses the [`hamzab/roberta-fake-news-classification`](https://huggingface.co/hamzab/roberta-fake-news-classification) model served via **HuggingFace Inference API** — no GPU required on the server side.

---

## 📚 References

- [Kaggle Fake News Dataset](https://www.kaggle.com/competitions/fake-news)
- [hamzab/roberta-fake-news-classification (HF Model)](https://huggingface.co/hamzab/roberta-fake-news-classification)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [Scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference)
