#!/bin/bash

#SBATCH --job-name=test_crows_pairs
#SBATCH --output=/home/xudong.han/llama2/logs/%A_%x.txt
#SBATCH --error=/home/xudong.han/llama2/logs/%A_%x.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --reservation=eval
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000
#SBATCH --gres=gpu:1
#SBATCH --partition=A100
#SBATCH --qos gpu-8
#SBATCH --time=12:00:00

cmd="python experiments/crows.py \
  --model LlamaLMModel \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --bias_type gender"
echo $cmd
eval $cmd