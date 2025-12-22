import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import random
from dataset import DigixDataset, collate_fn
from model import DirectPrediction, StandardModel
import pickle
from sklearn.metrics import recall_score
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import KFold
import argparse

def evaluate(model, dataloader, device, k=10):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            for k_, v in batch.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            batch[k_][kk] = vv.to(device)
                elif isinstance(v, torch.Tensor):
                    batch[k_] = v.to(device)

            scores = model.forward(batch)

            if torch.isnan(scores).any():
                raise ValueError("NaNs detected in model output")

            scores = scores.cpu()

            labels = batch["list1"]["labels"].cpu() 

            all_preds.append(scores)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    ndcg = ndcg_score(all_labels, all_preds, k=k)
    ndcg_100 = ndcg_score(all_labels, all_preds)

    topk_indices = torch.topk(torch.tensor(all_preds), k=k, dim=1).indices
    relevant_at_k = torch.gather(torch.tensor(all_labels), 1, topk_indices)

    total_relevant = all_labels.sum().item()
    if total_relevant == 0:
        recall = 0.0
    else:
        recall = relevant_at_k.sum().item() / total_relevant

    return recall, ndcg, ndcg_100

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        loss = model.calculate_loss(batch)

        loss = loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def train_loop(model, train_loader, test_loader, config, device, fold, model_name):
    optimizer = Adam(model.parameters(), lr=config['lr'])
    epochs = config['epochs']
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        recall, ndcg, ndcg_100 = evaluate(model, test_loader, device, k=10)

        print(f"Fold {fold+1} - Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | "
              f"Test Recall@{config['top_k']}: {recall:.4f} | Test NDCG@10: {ndcg:.4f} | Test NDCG@100: {ndcg_100:.4f}")

    final_path = os.path.join(checkpoint_dir, f'{model_name}_final_model_fold_{fold+1}.pt')
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training complete for fold {fold+1}. Final model saved to {final_path}")

def test_model(model, test_loader, device, k=10):
    recall, ndcg, ndcg_100 = evaluate(model, test_loader, device, k=k)
    print(f"Test Recall@{k}: {recall:.4f} | Test NDCG@{k}: {ndcg:.4f} | Test NDCG@100: {ndcg_100:.4f}")
    return recall, ndcg, ndcg_100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Recommender System Models")
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['DirectPrediction', 'StandardModel'],
        help="Name of the model class to train"
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True, 
        help="Path to the dataset pickle file"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        '--n_items',
        type=int,
        default=12557,
        help="Number of items in dataset"
    )

    args = parser.parse_args()

    MODEL_CLASSES = {
        'DirectPrediction': DirectPrediction,
        'StandardModel': StandardModel 
    }

    ModelClass = MODEL_CLASSES[args.model]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected Model: {args.model}")
    print(f"Dataset Path: {args.data_path}")

    config = {
        "n_items": args.n_items,
        "n_models": 3,
        "top_k": 100,
        "hidden_size": 8,
        "inner_size": 64,
        "n_heads": 2,
        "layer_norm_eps": 1e-5,
        "hidden_act": "relu",
        "num_layers": 1,
        "lr": 1e-4 if args.data_path == "./data/dataset.pkl" else 1e-5,
        "epochs": args.epochs,
        "checkpoint_dir": "./checkpoints"
    }

    print("Reading data...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_test_recalls = []
    all_test_ndcgs = []
    all_test_ndcgs_100 = []

    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"\n--- Fold {fold+1}/5 ---")
        
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]
        
        print("Creating datasets and loaders...")
        train_dataset = DigixDataset(train_data)
        test_dataset = DigixDataset(test_data)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

        model = ModelClass(config)
        model.to(device)

        print("Starting training...")
        train_loop(model, train_loader, test_loader, config, device, fold, args.model)

        print(f"--- Evaluating on Test Set for Fold {fold+1} ---")
        recall, ndcg, ndcg_100 = test_model(model, test_loader, device)
        all_test_recalls.append(recall)
        all_test_ndcgs.append(ndcg)
        all_test_ndcgs_100.append(ndcg_100)

    print(f"\n--- Summary for Model: {args.model} ---")
    print("--- Average Test Metrics Across 5 Folds ---")
    print(f"Average Test Recall@{config['top_k']}: {np.mean(all_test_recalls):.4f}")
    print(f"Average Test NDCG@10: {np.mean(all_test_ndcgs):.4f}")
    print(f"Average Test NDCG@100: {np.mean(all_test_ndcgs_100):.4f}")
