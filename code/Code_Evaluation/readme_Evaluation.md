



# Cryptocurrency Market Sentiment Analysis

**Author:** Teddy Thomas  

**Citation:** M. Kulakowski and F. Frasincar, "Sentiment Classification of Cryptocurrency-Related Social Media Posts," IEEE Intelligent Systems, vol. 38, no. 4, pp. 5-9, July-Aug. 2023.

---

## Project Overview

This repository provides a scalable, reproducible pipeline for classifying the sentiment of StockTwits posts about cryptocurrencies (BTC, ETH, SHIB) using:

- **Lexicon-based models** (VADER)
- **Classical ML models** (Naive Bayes, Logistic Regression, Linear SVC)
- **Word Embeddings** (Word2Vec + Logistic Regression)
- **Transformer-based models** (BERT)

The pipeline is optimized for high-performance computing (HPC) on the CBCB UMD cluster, supporting distributed processing and GPU acceleration.


---

## Evaluation Metrics

All models are evaluated on the same test set using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

| Model                      | Accuracy | Precision | Recall | F1-Score |
|----------------------------|----------|-----------|--------|----------|
| VADER                      | 0.33     | 0.34      | 0.32   | 0.31     |
| Naive Bayes                | 0.59     | 0.59      | 0.58   | 0.58     |
| Logistic Regression        | 0.63     | 0.63      | 0.62   | 0.62     |
| Linear SVC                 | 0.65     | 0.65      | 0.64   | 0.64     |
| Word2Vec + Logistic Reg.   | 0.51     | 0.52      | 0.51   | 0.50     |
| **BERT (Transformer)**     | **0.71** | **0.72**  | **0.71**| **0.71** |

---

## How to Run Evaluations

### 1. Clone the Repository

```

git clone https://github.com/teddylittishya/Data651-Market-Sentiment-Analysis-LLM-NLP-Approach.git
cd crypto-sentiment-analysis

```

### 2. Install Dependencies

```

pip install -r requirements.txt

```

### 3. Data Preparation

- Download the StockTwits Crypto dataset (see `data/README.md` for instructions).
- Preprocessing is handled automatically by the pipeline.

### 4. Running Model Evaluations

#### On Local Machine

```

python main.py --method tfidf
python main.py --method bert

```

#### On CBCB HPC Cluster

Submit a job using SLURM:

```

sbatch scripts/hpc_submit.sbatch

```

Monitor with:

```

squeue -u \$USER

```

### 5. Results

Evaluation results are saved in the `results/` directory as CSV and summary plots.

---

## Reproducibility

- All random seeds are fixed for reproducibility.
- The pipeline supports configuration via command-line arguments and config files.
- Results can be reproduced by running the provided scripts with the same dataset splits.

---

## Model Comparison

- **Lexicon-based (VADER):** Fast, interpretable, but lower accuracy on short/informal text.
- **Classical ML:** Improved performance with TF-IDF and SVC.
- **Word2Vec:** Sensitive to data quality; moderate results.
- **Transformer (BERT):** Best accuracy; leverages sentence context; requires HPC/GPU.

---

## References

1. M. Kulakowski and F. Frasincar, "Sentiment Classification of Cryptocurrency-Related Social Media Posts," IEEE Intelligent Systems, vol. 38, no. 4, pp. 5-9, July-Aug. 2023.
2. J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL-HLT, 2019.

---

