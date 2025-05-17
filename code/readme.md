
# Cryptocurrency Market Sentiment Analysis Pipeline

**Author:** Teddy Thomas  
**Institution:** University of Maryland, College Park  
**Cluster:** CBCB HPC Cluster   
**Citation:**  
M. Kulakowski and F. Frasincar, "Sentiment Classification of Cryptocurrency-Related Social Media Posts," IEEE Intelligent Systems, vol. 38, no. 4, pp. 5-9, July-Aug. 2023, doi: 10.1109/MIS.2023.3283170.

---

## Project Overview

This project implements a scalable, modular pipeline for **sentiment classification** of cryptocurrency-related StockTwits posts. It leverages **lexicon-based**, **classical machine learning**, and **transformer-based** NLP models, optimized for distributed processing on the CBCB HPC cluster using Apache Spark and GPU acceleration.

**Supported Models:**
- VADER (lexicon-based)
- CountVectorizer/TF-IDF + Naive Bayes/Logistic Regression/Linear SVM (classical ML)
- Word2Vec + Logistic Regression (embeddings)
- BERT (transformer-based, fine-tuned for crypto sentiment)

---

## Problem Statement

**Objective:**  
Classify StockTwits posts related to Bitcoin, Ethereum, and Shiba Inu as:
- **Bullish** (2)
- **Bearish** (1)
- **Neutral** (0)

**Goal:**  
Compare the performance of different NLP approaches for accurate, scalable sentiment detection in a highly sentiment-driven market.

---

## Dataset

- **Source:** StockTwits (ElKulako/stocktwits-crypto)
- **Period:** Train: Nov 2021–Jun 2022 | Test: Jun 2022
- **Volume:** 1.33M train posts, 83k test posts
- **Labels:** Bullish (2), Bearish (1), Neutral (0, inferred)
- **Cryptos:** BTC.X, ETH.X, SHIB.X

---

## Pipeline Architecture

1. **Data Loading:** Distributed loading from Parquet files.
2. **Preprocessing:**  
   - Remove URLs, usernames, cashtags, hashtags, wallet addresses, emojis (except select ones), non-English scripts.
   - Lowercase, fix encoding, drop duplicates/short posts.
3. **Feature Engineering:**  
   - Tokenization, stopword removal.
   - TF-IDF/CountVectorizer or BERT embeddings.
4. **Model Training:**  
   - Classical ML (Logistic Regression, SVM)
   - Transformer-based (BERT with GPU acceleration)
5. **Evaluation:**  
   - Accuracy, Precision, Recall, F1-Score
6. **HPC Integration:**  
   - Spark for distributed data processing.
   - SLURM for job scheduling.
   - GPU support for transformer models.

---

## Quickstart

### 1. Clone the Repository

```

git clone https://github.com/teddylittishya/Data651-Market-Sentiment-Analysis-LLM-NLP-Approach.git
cd crypto-sentiment-pipeline

```

### 2. Set Up the Environment

```

module load python/3.10 cuda/11.7 spark/3.3.0
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### 3. Prepare the Data

Download and place the dataset in `/hpc/storage/crypto/` as `train.parquet` and `test.parquet`.

### 4. Run the Pipeline on CBCB HPC

Submit the job using SLURM:

```

sbatch scripts/run_hpc.sh

```

Or run locally for testing:

```

python main.py --method tfidf --model lr
python main.py --method bert --model lr

```

---

## 🔧 Configuration

- **Pipeline options:**  
  - `--method`: `tfidf` or `bert`
  - `--model`: `lr` (Logistic Regression) or `svm` (Linear SVM)
  - `--hpc_config`: JSON string for Spark/HPC parameters

- **Edit** `scripts/run_hpc.sh` for cluster-specific resources (nodes, GPUs, memory).

---

## 📈 Results

| Model                       | Accuracy | Precision | Recall | F1-Score |
|-----------------------------|----------|-----------|--------|----------|
| VADER                       | 0.33     | 0.34      | 0.32   | 0.31     |
| Naive Bayes                 | 0.59     | 0.59      | 0.58   | 0.58     |
| Logistic Regression         | 0.63     | 0.63      | 0.62   | 0.62     |
| Linear SVM                  | 0.65     | 0.65      | 0.64   | 0.64     |
| Word2Vec + LR               | 0.51     | 0.52      | 0.51   | 0.50     |
| **BERT (Spark NLP)**        | **0.71** | **0.72**  | **0.71**| **0.71** |

---

## 🧠 Insights

- Lexicon-based models underperform on short, informal text.
- Classical models benefit from robust vectorization.
- Transformer-based models (BERT) yield the best accuracy.
- Spark and HPC parallelism enable scalable processing of millions of posts.

---

## 📚 References

1. M. Kulakowski and F. Frasincar, "Sentiment Classification of Cryptocurrency-Related Social Media Posts," IEEE Intelligent Systems, vol. 38, no. 4, pp. 5-9, July-Aug. 2023.
2. J. Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” NAACL-HLT, 2019.
3. Databricks 2025. Apache, Apache Spark, Spark and the Spark logo are trademarks of the Apache Software Foundation.


---

## 📬 Contact

For questions or contributions, open an issue or contact Teddy Thomas at tedthom1@umd.edu

---


