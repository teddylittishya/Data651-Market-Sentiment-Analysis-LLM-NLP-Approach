# main.py
"""
Cryptocurrency Sentiment Analysis Pipeline
Combines Lexicon, Classical ML, and Transformer Approaches
Optimized for CBCB HPC Cluster with Spark Distributed Processing
"""

import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer, 
    StopWordsRemover, 
    HashingTF, 
    IDF
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoAnalysisPipeline:
    """Main pipeline class handling end-to-end processing"""
    
    def __init__(self, hpc_config):
        self.spark = self._init_spark_session(hpc_config)
        self.models = {
            'lr': LogisticRegression(featuresCol='features', labelCol='sentiment'),
            'svm': LinearSVC(featuresCol='features', labelCol='sentiment')
        }
        
    def _init_spark_session(self, config):
        """Initialize Spark session with HPC parameters"""
        return SparkSession.builder \
            .appName("Crypto_Sentiment_HPC") \
            .config("spark.driver.memory", config['driver_mem']) \
            .config("spark.executor.memory", config['executor_mem']) \
            .config("spark.executor.cores", config['cores']) \
            .config("spark.executor.instances", config['instances']) \
            .getOrCreate()
    
    def load_data(self, path):
        """Load and repartition data for cluster processing"""
        df = self.spark.read.parquet(path)
        return df.repartition(64)  # Optimized for cluster parallelism
    
    def preprocess(self, df):
        """Data cleaning pipeline"""
        from pyspark.sql.functions import col, lower, regexp_replace
        
        return df.withColumn("clean_text", 
            regexp_replace(lower(col("text")), 
            r"http\S+|@\w+|#\w+|\$[A-Z]+|[\U00010000-\U0010ffff]", "")) \
            .dropna() \
            .filter("LENGTH(clean_text) > 4")
    
    def create_feature_pipeline(self, method='tfidf'):
        """Feature engineering pipeline"""
        tokenizer = RegexTokenizer(
            inputCol="clean_text", 
            outputCol="words", 
            pattern="\\W+"
        )
        
        stopwords = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words"
        )
        
        if method == 'tfidf':
            hashing_tf = HashingTF(
                inputCol="filtered_words", 
                outputCol="raw_features", 
                numFeatures=2**18
            )
            
            idf = IDF(
                inputCol="raw_features", 
                outputCol="features"
            )
            
            return Pipeline(stages=[tokenizer, stopwords, hashing_tf, idf])
            
        elif method == 'bert':
            from pyspark.sql.functions import pandas_udf
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased")
            
            @pandas_udf("array<float>")
            def bert_embeddings(texts):
                inputs = tokenizer(
                    texts.tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                return pd.Series(outputs.last_hidden_state[:, 0].cpu().numpy())
                
            return bert_embeddings
    
    def train(self, df, method='tfidf', model_type='lr'):
        """Model training pipeline"""
        feature_pipeline = self.create_feature_pipeline(method)
        
        if method == 'tfidf':
            feature_model = feature_pipeline.fit(df)
            transformed_data = feature_model.transform(df)
            model = self.models[model_type].fit(transformed_data)
            return feature_model, model
            
        elif method == 'bert':
            from pyspark.ml.feature import VectorAssembler
            
            df = df.withColumn("bert_features", self.create_feature_pipeline('bert')(df["clean_text"]))
            assembler = VectorAssembler(inputCols=["bert_features"], outputCol="features")
            assembled_data = assembler.transform(df)
            return self.models[model_type].fit(assembled_data)
    
    def evaluate(self, model, test_data):
        """Model evaluation with multiple metrics"""
        predictions = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="sentiment", 
            predictionCol="prediction"
        )
        
        return {
            "accuracy": evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}),
            "f1": evaluator.evaluate(predictions, {evaluator.metricName: "f1"}),
            "precision": evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"}),
            "recall": evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=['tfidf', 'bert'], default='tfidf')
    parser.add_argument("--model", choices=['lr', 'svm'], default='lr')
    parser.add_argument("--hpc_config", type=json.loads, default='{"driver_mem": "32g", "executor_mem": "64g", "cores": 8, "instances": 16}')
    args = parser.parse_args()
    
    pipeline = CryptoAnalysisPipeline(args.hpc_config)
    
    # Load and preprocess data
    logger.info("Loading data...")
    train_df = pipeline.load_data("/hpc/storage/crypto/train.parquet")
    test_df = pipeline.load_data("/hpc/storage/crypto/test.parquet")
    
    logger.info("Preprocessing data...")
    train_clean = pipeline.preprocess(train_df)
    test_clean = pipeline.preprocess(test_df)
    
    # Train and evaluate
    logger.info(f"Training {args.method} model...")
    feature_model, trained_model = pipeline.train(train_clean, args.method, args.model)
    
    logger.info("Evaluating model...")
    metrics = pipeline.evaluate(trained_model, test_clean)
    
    # Save results
    pd.DataFrame([metrics]).to_csv(f"results/{args.method}_{args.model}_metrics.csv")
    trained_model.write().overwrite().save(f"models/{args.method}_{args.model}_model")
    
    logger.info(f"Completed {args.method} {args.model} training with metrics: {metrics}")
