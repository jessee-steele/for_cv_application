#!/bin/bash
 
#Submit this script with: sbatch myjob-onlyTest-noTrain.slurm
 
#SBATCH --time=12:00:00 #changed down from 24 hrs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --constraint="a100|a40|a6000|v100"
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=rxv7nr@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/%x_%A.out
 
cd ~/us-tree-detection-geo-share
source env/bin/activate

CONFIG=config-1mmx_FINETUNE_JS_edited #changed from ensemble.

# 1. Clean and format input images and labels for analysis. Split data into training, validation, and testing sets.
python3 -m scripts.organize --tile --config "$CONFIG"
python3 -m scripts.split --config "$CONFIG"
python3 -m scripts.prepare --config "$CONFIG"

# 1. Fine-tune the neural network
python3 -m scripts.train --config "$CONFIG"

# 2. Find the optimal detection thresholds for the new model
python3 -m scripts.tune --config "$CONFIG"

# 3. Test the model using the optimal thresholds found by 'tune'
# (Do not use the --peak_value flag here, so it defaults to the tuned params_detection.yaml)
python3 -m scripts.test --config "$CONFIG" #--peak_value 0.4