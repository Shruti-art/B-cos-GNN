#!/usr/bin/env python3
# ============================================================
# MUTAG — Fidelity vs Sparsity (BCos-GCN vs Other Explainers)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import subgraph

# -----------------------------
# Reproducibility
# -----------------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# BCos feature transform
# -----------------------------
def bcos_transform(x, B=1.5, eps=1e-8):
    norm = torch.norm(x, dim=-1, keepdim=True) + eps
    x_norm = x / norm
    return (norm ** (B - 1)) * x_norm

# -----------------------------
# BCos-GCN
# -----------------------------
class BCosGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, B=1.5):
        super().__init__()
        self.B = B
        self.conv1 = GCNConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, x, edge_index, batch):
        x = bcos_transform(x, self.B)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = bcos_transform(x, self.B)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        return self.classifier(x)

# -----------------------------
# Node importance
# -----------------------------
@torch.no_grad()
def node_importance(model, data):
    x = bcos_transform(data.x, model.B)
    x = model.conv1(x, data.edge_index)
    x = F.relu(x)

    x = bcos_transform(x, model.B)
    x = model.conv2(x, data.edge_index)

    logits = model(data.x, data.edge_index, data.batch)
    pred = logits.argmax(dim=1)

    w = model.classifier.weight[pred]
    scores = (x * w).sum(dim=1)
    return scores.abs()

# -----------------------------
# Fidelity curve
# -----------------------------
@torch.no_grad()
def fidelity_curve(model, data, sparsities, device):
    data = data.to(device)

    full_prob = F.softmax(
        model(data.x, data.edge_index, data.batch), dim=1
    )[0]
    target = full_prob.argmax().item()

    scores = node_importance(model, data)
    order = scores.argsort(descending=True)

    fidelities = []

    for s in sparsities:
        k = max(1, int(len(order) * s))
        keep = order[:k]

        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        node_mask[keep] = True

        edge_index_sub, _ = subgraph(
            node_mask,
            data.edge_index,
            relabel_nodes=True
        )

        x_sub = data.x[node_mask]
        batch_sub = torch.zeros(x_sub.size(0), dtype=torch.long, device=device)

        sub_prob = F.softmax(
            model(x_sub, edge_index_sub, batch_sub), dim=1
        )[0][target]

        fidelities.append(sub_prob.item())

    return fidelities

# -----------------------------
# Dataset averaging
# -----------------------------
def dataset_curve(model, dataset, sparsities, device):
    curves = []
    for g in dataset:
        curves.append(fidelity_curve(model, g, sparsities, device))
    return np.mean(curves, axis=0)

# -----------------------------
# Main: combined plot
# -----------------------------
def main():
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TUDataset(root="data/MUTAG", name="MUTAG")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    sparsities = np.array([0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.78])
    B_values = [1.3, 1.5, 1.7]

    # -----------------------------
    # Baseline explainers
    # -----------------------------
    subgraphx = np.array([0.60, 0.59, 0.59, 0.58, 0.56, 0.54, 0.49])
    pgexplainer = np.array([0.26, 0.26, 0.26, 0.26, 0.25, 0.23, 0.23])
    gnnexplainer = np.array([0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18])

    plt.figure(figsize=(9, 6))

    # -----------------------------
    # Plot BCos-GCN curves for all B
    # -----------------------------
    for B in B_values:
        print(f"Training BCos-GCN with B = {B} ...")
        model = BCosGCN(
            in_dim=dataset.num_features,
            hidden_dim=64,
            num_classes=dataset.num_classes,
            B=B
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train
        model.train()
        for epoch in range(100):
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()

        # Fidelity
        model.eval()
        curve = dataset_curve(model, dataset, sparsities, device)
        plt.plot(
            sparsities, curve,
            marker="D", linewidth=3, markersize=8,
            label=f"BCos-GCN (B={B})"
        )

    # -----------------------------
    # Plot baseline explainers
    # -----------------------------
    plt.plot(sparsities, subgraphx,
             marker="o", linewidth=3, markersize=8, label="SubgraphX (GCN)")
    plt.plot(sparsities, pgexplainer,
             marker="^", linewidth=3, markersize=8, label="PGExplainer")
    plt.plot(sparsities, gnnexplainer,
             marker="*", linewidth=3, markersize=10, label="GNNExplainer")

    # -----------------------------
    # Formatting
    # -----------------------------
    plt.xlabel("Sparsity", fontsize=16)
    plt.ylabel("Fidelity", fontsize=16)
    plt.title("MUTAG — BCos-GCN vs Other Explainers", fontsize=18)
    plt.xlim(0.55, 0.80)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12, loc="lower left")

    plt.tight_layout()
    plt.savefig("MUTAG_Fidelity_All_Explainers.png", dpi=300)
    plt.close()

    print("Saved combined fidelity plot: MUTAG_Fidelity_All_Explainers.png")

# -----------------------------
if __name__ == "__main__":
    main()
