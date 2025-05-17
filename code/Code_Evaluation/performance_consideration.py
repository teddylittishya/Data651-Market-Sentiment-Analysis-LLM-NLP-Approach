# Feature extraction optimizations
from pyspark.ml import Pipeline
from pyspark.sql.functions import pandas_udf

@pandas_udf('array<float>')
def bert_embedding(texts: pd.Series) -> pd.Series:
    # Batch processing for transformer models
    inputs = tokenizer(texts.tolist(), return_tensors='pt', 
                      padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return pd.Series(outputs.logits.numpy())
