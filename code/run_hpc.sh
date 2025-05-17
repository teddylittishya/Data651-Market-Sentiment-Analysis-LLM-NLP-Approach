#!/bin/bash
#SBATCH --job-name=crypto_sentiment
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module purge
module load python/3.10.4 cuda/11.7.1 spark/3.3.0

# Initialize Spark cluster in HPC environment
export SPARK_MASTER_HOST=$(hostname)
export SPARK_LOCAL_DIRS=$TMPDIR
export SPARK_WORKER_DIR=$TMPDIR
export SPARK_LOG_DIR=$TMPDIR/logs
export SPARK_CONF_DIR=$SLURM_SUBMIT_DIR/conf

# Start Spark cluster master
srun --nodes=1 --ntasks=1 --exact spark-class org.apache.spark.deploy.master.Master &
MASTER_PID=$!
sleep 10

# Get Spark master URL
MASTER_URL=$(grep -Po '(spark://.*)' $SPARK_LOG_DIR/spark*.out)

# Start workers on remaining nodes
srun --nodes=$((SLURM_JOB_NUM_NODES-1)) --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    spark-class org.apache.spark.deploy.worker.Worker $MASTER_URL &

# Run pipeline with different configurations
python main.py --method tfidf --model lr
python main.py --method bert --model lr

# Cleanup
kill $MASTER_PID
