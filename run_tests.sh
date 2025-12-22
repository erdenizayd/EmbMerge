#!/bin/bash

DATASET_DIGIX="./data/dataset_digix.pkl"
DATASET_AMAZON="./data/dataset_amazon.pkl"


echo "Starting Amazon Standard."
python3 traincli.py --model StandardModel --data_path "$DATASET_AMAZON" --epochs 10 --n_items 26867

echo "Starting Amazon Direct."
python3 traincli.py --model DirectPrediction --data_path "$DATASET_AMAZON" --epochs 10 --n_items 26867

echo "Starting Digix Standard."
python3 traincli.py --model StandardModel --data_path "$DATASET_DIGIX" --epochs 10 --n_items 12557

echo "Starting Digix Direct."
python3 traincli.py --model DirectPrediction --data_path "$DATASET_DIGIX" --epochs 10 --n_items 12557

