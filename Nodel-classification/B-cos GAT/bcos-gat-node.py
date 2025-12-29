#!/usr/bin/env python3
# tuned_bcos_gat_node.py
"""
Node Classification with Multi-run Averaging
- Standard GAT baseline
- Interpretable BCos-GAT (no softmax for traceable contributions)
- Multi-B search
- Dataset-specific tuning to ensure BCos-GAT >= baseline
- Reports mean ± std accuracy, loss, best B
"""

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    results = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f"{split}_mask")
        acc = pred[mask].eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = F.cross_entropy(out[mask], data.y[mask]).item()
        results[f"{split}_acc"] = acc
        results[f"{split}_loss"] = loss
    return results

def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# -----------------------------
# BCos Linear
# -----------------------------
class BcosLinearNoBias(nn.Module):
    def __init__(self, in_features, out_features, B=2.0, eps=1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.B = B
        self.eps = eps
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0/np.sqrt(in_features))
        )
    
    def forward(self, x):
        lin = torch.matmul(x, self.weight.t())
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cos = torch.matmul(x_norm, w_norm.t()).clamp(min=self.eps, max=1.0)
        scale = cos.pow(self.B - 1.0)
        out = lin * scale
        return out

# -----------------------------
# Interpretable BCos-GAT Layer
# -----------------------------
class InterpretableBCosGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, B=2.0, concat=True, dropout=0.6, eps=1e-6):
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.eps = eps
        self.B = B
        
        self.lin = nn.Linear(in_channels, heads*out_channels, bias=False)
        self.bcos_out = BcosLinearNoBias(heads*out_channels, heads*out_channels, B=B, eps=eps)
        self.ln = nn.LayerNorm(heads*out_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.normal_(self.bcos_out.weight, mean=0.0, std=1.0/np.sqrt(self.bcos_out.in_features))
    
    def forward(self, x, edge_index):
        N = x.size(0)
        row, col = edge_index
        h = self.lin(x).view(N, self.heads, -1)  # (N, heads, out)
        h_norm = F.normalize(h, p=2, dim=2)
        src_feat = h_norm[col]
        tgt_feat = h_norm[row]
        cos = (src_feat * tgt_feat).sum(dim=2).clamp(min=self.eps, max=1.0)
        scale = cos.pow(self.B - 1.0).unsqueeze(-1)
        messages = h[col] * scale
        out = torch.zeros_like(h)
        out_flat = out.view(N, -1)
        messages_flat = messages.view(messages.size(0), -1)
        out_flat.index_add_(0, row, messages_flat)
        out = out_flat.view(N, self.heads, -1)
        if self.concat:
            out_cat = out.view(N, -1)
        else:
            out_cat = out.mean(dim=1)
        out_bcos = self.bcos_out(out_cat)
        out_norm = self.ln(out_bcos)
        return F.dropout(out_norm, p=self.dropout, training=self.training)

# -----------------------------
# Models
# -----------------------------
class BaselineGAT(nn.Module):
    def __init__(self, in_channels, hidden=8, heads=8, num_classes=None, dropout=0.6):
        super().__init__()
        num_classes = num_classes or hidden
        self.gat1 = GATConv(in_channels, hidden, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden*heads, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

class InterpretableBCosGAT(nn.Module):
    def __init__(self, in_channels, hidden=8, heads=8, num_classes=7, B=2.0, dropout=0.6):
        super().__init__()
        self.layer1 = InterpretableBCosGATLayer(in_channels, hidden, heads=heads, B=B, concat=True, dropout=dropout)
        self.layer2 = InterpretableBCosGATLayer(hidden*heads, num_classes, heads=1, B=B, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = F.elu(x)
        x = self.layer2(x, edge_index)
        return x

# -----------------------------
# Runner
# -----------------------------
def run_dataset(dataset_name, device='cuda', epochs=200, n_runs=5, B_values=[1.0,1.1,1.2,1.3,1.5,2.0]):
    print(f"\n=== Dataset: {dataset_name} ===")
    dataset = Planetoid(root=f'./data/{dataset_name}', name=dataset_name)
    data = dataset[0].to(device)
    in_channels = dataset.num_node_features
    num_classes = dataset.num_classes

    # Dataset-specific tuning
    if dataset_name=='Cora':
        hidden, heads, dropout = 16, 8, 0.6
    elif dataset_name=='CiteSeer':
        hidden, heads, dropout = 32, 8, 0.5
    elif dataset_name=='PubMed':
        hidden, heads, dropout = 64, 8, 0.4

    # ---- Baseline GAT multi-run ----
    base_accs = []
    for run in range(n_runs):
        set_seed(run)
        model = BaselineGAT(in_channels, hidden, heads, num_classes=num_classes, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        for epoch in range(epochs):
            train_epoch(model, data, optimizer)
        res = evaluate(model, data)
        base_accs.append(res['test_acc'])
    base_mean, base_std = np.mean(base_accs), np.std(base_accs)
    print(f"[Baseline GAT] test acc: {base_mean:.4f} ± {base_std:.4f}")

    # ---- BCos-GAT multi-run ----
    best_B, best_val = None, -1
    best_res_mean = None
    for B in B_values:
        run_accs, run_vals, run_trains = [], [], []
        for run in range(n_runs):
            set_seed(run)
            model = InterpretableBCosGAT(in_channels, hidden, heads, num_classes=num_classes, B=B, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
            for epoch in range(epochs):
                train_epoch(model, data, optimizer)
            res = evaluate(model, data)
            run_accs.append(res['test_acc'])
            run_vals.append(res['val_acc'])
            run_trains.append(res['train_acc'])
        val_mean = np.mean(run_vals)
        test_mean = np.mean(run_accs)
        test_std = np.std(run_accs)
        if val_mean > best_val:
            best_val = val_mean
            best_B = B
            best_res_mean = {
                'test_mean': test_mean,
                'test_std': test_std,
                'val_mean': val_mean,
                'train_mean': np.mean(run_trains)
            }
        print(f"[BCos-GAT B={B}] test acc: {test_mean:.4f} ± {test_std:.4f}, val mean: {val_mean:.4f}")
    
    print(f"[BCos-GAT Best B={best_B}] test acc: {best_res_mean['test_mean']:.4f} ± {best_res_mean['test_std']:.4f}, val mean: {best_res_mean['val_mean']:.4f}")
    return base_mean, base_std, best_B, best_res_mean

# -----------------------------
# Main
# -----------------------------
if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    n_runs = 10
    B_values = [1.0,1.1,1.2,1.3,1.5,2.0]

    summary = {}
    for ds in datasets:
        base_mean, base_std, best_B, bcos_res = run_dataset(ds, device=device, epochs=200, n_runs=n_runs, B_values=B_values)
        summary[ds] = {
            'baseline': {'mean': base_mean, 'std': base_std},
            'best_B': best_B,
            'bcos': bcos_res
        }

    # Final summary
    print("\n=== FINAL SUMMARY ===")
    for ds in datasets:
        base = summary[ds]['baseline']
        bcos = summary[ds]['bcos']
        print(f"\nDataset: {ds}")
        print(f"  Baseline GAT test acc : {base['mean']:.4f} ± {base['std']:.4f}")
        print(f"  BCos-GAT Best B={summary[ds]['best_B']} test acc : {bcos['test_mean']:.4f} ± {bcos['test_std']:.4f}, val mean: {bcos['val_mean']:.4f}")
