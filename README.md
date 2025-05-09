# 🚀 Cryptocurrency Market Sentiment Analysis: A Trasformers, Lexicon and classical Models based approach

**Author:** Teddy Thomas  
**Last Updated:** May 2025

---

## 🧠 Motivation

The cryptocurrency market is highly sentiment-driven, especially among retail investors who actively use platforms like [StockTwits](https://stocktwits.com).  
Understanding user sentiment can help:

- Develop automated trading systems
- Perform real-time market trend analysis
- Track investor psychology at scale

---

## ❓ Problem Statement

**Objective:**  
Accurately classify sentiment (Bullish, Bearish, Neutral) of user-generated posts on StockTwits related to cryptocurrencies (Bitcoin, Ethereum, Shiba Inu) using a mix of:

- Lexicon-based models
- Classical machine learning classifiers
- Word embeddings
- Transformer-based NLP models

---

## 🗃️ Code Structure (Modular Design)

The project codebase follows a modular structure for clarity and maintainability:

project_root/
│
├── data/ # Raw and processed data files
├── notebooks/ # Jupyter/Colab notebooks for EDA and experiments
├── src/ # Source code for preprocessing, modeling, evaluation
│ ├── preprocessing.py # Text cleaning and tokenization
│ ├── feature_engineering.py # Vectorizers and embeddings
│ ├── models.py # ML, embedding-based, and transformer models
│ ├── utils.py # Helper functions and metrics
│ └── config.py # Centralized configuration
├── requirements.txt # Dependency file
├── run_pipeline.py # End-to-end training + evaluation script
└── README.md # Project documentation


---

## 💬 Comments and Inline Documentation

All Python files in the `src/` directory are thoroughly documented with:

- **Function-level docstrings** describing inputs, outputs, and logic
- **Inline comments** to explain preprocessing decisions, model configs, and evaluation logic
- **Notebook annotations** for clarity on steps during experimentation

---

## 🔁 Reproducibility

The project is designed for full reproducibility:

- ✅ **`requirements.txt`** to install dependencies
- ✅ **Clear `README.md`** (this file) for setup and run instructions
- ✅ **Executable scripts** (`run_pipeline.py`) for end-to-end training
- ✅ **Random seed control** for consistent results
- ✅ **Jupyter notebooks** for reproducible exploratory analysis

> To get started:
```bash
git clone https://github.com/yourusername/crypto-sentiment-nlp.git
cd crypto-sentiment-nlp
pip install -r requirements.txt
python run_pipeline.py





