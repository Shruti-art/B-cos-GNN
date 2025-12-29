# bcos_gcn_tune.py
# Node classification with multi-seed evaluation (10 seeds)
# Adds mean/std reporting for Baseline GCN and BCos-GCN for each B.

import argparse
import os
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_adj_edge_weight(edge_index, num_nodes, dtype=torch.float):
    edge_index_with_self, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index_with_self

    deg = degree(col, num_nodes=num_nodes, dtype=dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index_with_self, edge_weight


# -----------------------------------------------------------
# BCos layer
# -----------------------------------------------------------
class BcosGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, B: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.B = float(B)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / np.sqrt(in_features))
        )

    def forward(self, z, edge_index=None, edge_weight=None):
        lin = torch.matmul(z, self.weight.t())

        z_norm = F.normalize(z, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        cos = torch.matmul(z_norm, w_norm.t())
        scale = cos.abs().pow(self.B - 1.0)

        return lin * scale


# -----------------------------------------------------------
# BCos GCN model
# -----------------------------------------------------------
class BcosGCN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, B=2.0, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1 = BcosGCNLayer(in_channels, hidden, B)
        self.layer2 = BcosGCNLayer(hidden, out_channels, B)

    def forward(self, x, edge_index, edge_weight):
        row, col = edge_index
        z = torch.zeros_like(x)
        z.index_add_(0, row, x[col] * edge_weight.unsqueeze(-1))

        h1 = self.layer1(z)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        z2 = torch.zeros_like(h1)
        z2.index_add_(0, row, h1[col] * edge_weight.unsqueeze(-1))

        return self.layer2(z2)


# -----------------------------------------------------------
# Baseline GCN model
# -----------------------------------------------------------
class BaselineGCN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden, bias=False)
        self.conv2 = GCNConv(hidden, out_channels, bias=False)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


# -----------------------------------------------------------
# Train / Eval routines
# -----------------------------------------------------------
def train_epoch(model, data, optimizer, device, edge_index, edge_weight):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), edge_index.to(device), edge_weight.to(device))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, device, edge_index, edge_weight):
    model.eval()
    out = model(data.x.to(device), edge_index.to(device), edge_weight.to(device))
    preds = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = preds[mask].eq(data.y[mask].to(device)).sum().item()
        accs.append(correct / int(mask.sum().item()))
    return accs, out


# ===========================================================
# Multi-seed experiment for one model
# ===========================================================
def run_single_model(model_class, model_args, data, edge_index, edge_weight,
                     lr, weight_decay, epochs, device, seeds):

    val_list = []
    test_list = []
    loss_list = []

    for seed in seeds:
        set_seed(seed)

        model = model_class(*model_args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_val = 0.0
        best_test = 0.0
        final_loss = 0.0

        for _ in range(epochs):
            final_loss = train_epoch(model, data, optimizer, device, edge_index, edge_weight)
            accs, _ = evaluate(model, data, device, edge_index, edge_weight)
            train_acc, val_acc, test_acc = accs

            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc

        val_list.append(best_val)
        test_list.append(best_test)
        loss_list.append(final_loss)

    return {
        "val_mean": float(np.mean(val_list)),
        "val_std": float(np.std(val_list)),
        "test_mean": float(np.mean(test_list)),
        "test_std": float(np.std(test_list)),
        "loss_mean": float(np.mean(loss_list)),
        "loss_std": float(np.std(loss_list)),
    }


# ===========================================================
# Tuning Loop
# ===========================================================
def run_tuning(dataset_name, device, B_grid, hidden, lr, weight_decay, epochs, base_seed):
    print(f"\n=== Dataset: {dataset_name} | device={device} ===")

    dataset = Planetoid(root=f"./data/{dataset_name}", name=dataset_name)
    data = dataset[0]

    edge_index, edge_weight = normalize_adj_edge_weight(data.edge_index, data.num_nodes)

    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes

    # Multi-seed setup
    seeds = [base_seed + i for i in range(10)]

    # ------------------------------------------------------
    # Standard GCN
    # ------------------------------------------------------
    print("\nRunning Standard GCN across 10 seeds...")
    base_results = run_single_model(
        BaselineGCN,
        (in_channels, hidden, out_channels),
        data,
        edge_index,
        edge_weight,
        lr,
        weight_decay,
        epochs,
        device,
        seeds,
    )

    print("\n=== Standard GCN Results (mean ± std) ===")
    print(f"Val Acc : {base_results['val_mean']:.4f} ± {base_results['val_std']:.4f}")
    print(f"Test Acc: {base_results['test_mean']:.4f} ± {base_results['test_std']:.4f}")
    print(f"Loss    : {base_results['loss_mean']:.4f} ± {base_results['loss_std']:.4f}")

    # ------------------------------------------------------
    # BCos-GCN for each B
    # ------------------------------------------------------
    bcos_results = {}

    for B in B_grid:
        print(f"\nRunning BCos-GCN (B={B}) across 10 seeds...")
        res = run_single_model(
            BcosGCN,
            (in_channels, hidden, out_channels, B),
            data,
            edge_index,
            edge_weight,
            lr,
            weight_decay,
            epochs,
            device,
            seeds,
        )
        bcos_results[B] = res

        print(f"\n=== BCos-GCN (B={B}) Results ===")
        print(f"Val Acc : {res['val_mean']:.4f} ± {res['val_std']:.4f}")
        print(f"Test Acc: {res['test_mean']:.4f} ± {res['test_std']:.4f}")
        print(f"Loss    : {res['loss_mean']:.4f} ± {res['loss_std']:.4f}")

    return base_results, bcos_results


# ===========================================================
# CLI
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora",
                        choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    B_grid = [1.0, 1.5, 2.0, 2.5, 3.0]

    run_tuning(
        args.dataset,
        device=args.device,
        B_grid=B_grid,
        hidden=args.hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        base_seed=args.seed,
    )
