#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_from_disk
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from safetensors import safe_open
from safetensors.torch import save_file
import copy
from datetime import datetime

class PauseTypeProbe(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.linear(x)

def prepare_silence_datasets(dataset_path, device="cuda"):
    """Prepare datasets with middle silence labeling."""
    train_ds = load_from_disk(str(Path(dataset_path) / 'train'))
    val_ds = load_from_disk(str(Path(dataset_path) / 'val'))

    def process_dataset(ds):
        hidden_states = []
        labels = []
        edge_silence_mask = []

        for sample in ds:
            states = sample['hidden_states']
            silence_mask = sample['is_silence']
            n_frames = len(states)

            # Find silence runs
            silence_runs = []
            in_run = False
            run_start = 0

            for i in range(n_frames):
                if silence_mask[i] and not in_run:
                    in_run = True
                    run_start = i
                elif not silence_mask[i] and in_run:
                    silence_runs.append((run_start, i))
                    in_run = False

            if in_run:
                silence_runs.append((run_start, n_frames))

            # Mark edge runs
            edge_runs = set()
            for start, end in silence_runs:
                if start == 0 or end == n_frames:
                    edge_runs.update(range(start, end))

            # Process frames
            for i in range(n_frames):
                hidden_states.append(states[i])
                is_middle_silence = (silence_mask[i] and
                                   i not in edge_runs and
                                   0.1 < i/n_frames < 0.9)
                labels.append(float(is_middle_silence))
                edge_silence_mask.append(i in edge_runs)

        hidden_states = np.stack(hidden_states)
        edge_silence_mask = np.array(edge_silence_mask)

        return (torch.FloatTensor(hidden_states).to(device),
                torch.FloatTensor(labels).to(device),
                torch.BoolTensor(edge_silence_mask).to(device))

    print("Preparing datasets...")
    train_data = process_dataset(train_ds)
    val_data = process_dataset(val_ds)

    # Print statistics
    for name, (X, y, edge) in [("Training", train_data), ("Validation", val_data)]:
        pos_rate = (y == 1).float().mean()
        edge_rate = edge.float().mean()
        print(f"\n{name} set ({len(X)} frames):")
        print(f"- {pos_rate*100:.1f}% middle silence")
        print(f"- {edge_rate*100:.1f}% edge silence")
        print(f"- {(1-pos_rate-edge_rate)*100:.1f}% non-silence")

    return train_data, val_data

def train_pause_type_probe(train_data, val_data, batch_size=256, lr=1e-3, epochs=20):
    """Train the silence probe model."""
    X_train, y_train, _ = train_data
    X_val, y_val, edge_val = val_data

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    model = PauseTypeProbe(hidden_dim=X_train.shape[1]).to(X_train.device)
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(X_train.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0
    best_model = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X).squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend((torch.sigmoid(pred) > 0.5).cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = (np.array(train_preds) == np.array(train_labels)).mean()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_pred_prob = torch.sigmoid(val_pred)
            val_pred_bool = val_pred_prob > 0.5

            val_loss = criterion(val_pred, y_val)
            val_acc = (val_pred_bool == y_val).float().mean()

            # Core metrics
            true_pos = (val_pred_bool & (y_val == 1)).sum().item()
            false_pos = (val_pred_bool & (y_val == 0)).sum().item()
            false_neg = (~val_pred_bool & (y_val == 1)).sum().item()

            precision = true_pos / (true_pos + false_pos + 1e-8)
            recall = true_pos / (true_pos + false_neg + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            edge_trigger_rate = (val_pred_bool & edge_val).float().mean()

            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Edge silence trigger rate: {edge_trigger_rate:.4f}")

    print(f"\nBest validation F1: {best_f1:.4f}")
    return best_model

def main():
    # Training settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_PATH = "./silence_dataset"
    BATCH_SIZE = 256
    LR = 1e-3
    EPOCHS = 20

    # Train model
    train_data, val_data = prepare_silence_datasets(DATASET_PATH, device=DEVICE)
    model = train_pause_type_probe(train_data, val_data,
                                 batch_size=BATCH_SIZE,
                                 lr=LR,
                                 epochs=EPOCHS)

    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"silence_probe_{timestamp}.safetensors"
    state_dict = model.state_dict()
    save_file(state_dict, save_path)
    print(f"\nSaved model to: {save_path}")

if __name__ == "__main__":
    main()
