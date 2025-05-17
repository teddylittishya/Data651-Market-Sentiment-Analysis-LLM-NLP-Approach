# hpc_submit.sbatch (Slurm job script)
#!/bin/bash
#SBATCH --job-name=crypto-sentiment
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8 
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=%x-%j.out

module load python/3.10 cuda/11.7
source venv/bin/activate

# Distributed Spark configuration
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8080
export SPARK_WORKER_CORES=8
export SPARK_WORKER_MEM=64g

# Launch pipeline with different feature methods
python main.py --method tfidf --hpc_config driver_mem=32g
python main.py --method bert --use_gpu
