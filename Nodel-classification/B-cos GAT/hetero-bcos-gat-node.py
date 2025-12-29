#!/usr/bin/env python3
# ===========================================================
# BCos-GAT on Heterophilic Datasets (Texas, Cornell)
# Node Classification — GAT ONLY
# Interpretable BCos Attention (no softmax)
# ===========================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import WebKB
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

# -----------------------------------------------------------
# Hyperparameters (dataset-specific)
# -----------------------------------------------------------
HYPERPARAMS = {
    'Texas': {
        'hidden': 16,
        'heads': 4,
        'lr': 0.005,
        'wd': 5e-4,
        'epochs': 400,
        'dropout': 0.6
    },
    'Cornell': {
        'hidden': 16,
        'heads': 4,
        'lr': 0.005,
        'wd': 5e-4,
        'epochs': 400,
        'dropout': 0.6
    }
}

B_GRID = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7,2.0,2.5,3.0]

SEEDS = list(range(10))
SPLITS = [0, 1, 2]   # WebKB official splits

# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------
# BCos Linear (bias-free)
# -----------------------------------------------------------
class BcosLinearNoBias(nn.Module):
    def __init__(self, in_features, out_features, B=2.0, eps=1e-6):
        super().__init__()
        self.B = float(B)
        self.eps = eps
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / np.sqrt(in_features)
        )

    def forward(self, x):
        lin = x @ self.weight.t()
        x_n = F.normalize(x, p=2, dim=1)
        w_n = F.normalize(self.weight, p=2, dim=1)
        cos = (x_n @ w_n.t()).clamp(min=self.eps)
        scale = cos.abs().pow(self.B - 1.0)
        return lin * scale

# -----------------------------------------------------------
# Interpretable BCos-GAT Layer (no softmax)
# -----------------------------------------------------------
class InterpretableBCosGATLayer(nn.Module):
    def __init__(self, in_ch, out_ch, heads=4, B=2.0, dropout=0.6):
        super().__init__()
        self.heads = heads
        self.out_ch = out_ch
        self.B = B
        self.dropout = dropout

        self.lin = nn.Linear(in_ch, heads * out_ch, bias=False)
        self.bcos_out = BcosLinearNoBias(
            heads * out_ch, heads * out_ch, B=B
        )
        self.ln = nn.LayerNorm(heads * out_ch)

    def forward(self, x, edge_index):
        N = x.size(0)
        row, col = edge_index

        h = self.lin(x).view(N, self.heads, self.out_ch)
        h_norm = F.normalize(h, p=2, dim=2)

        src = h_norm[col]
        tgt = h_norm[row]

        cos = (src * tgt).sum(dim=2).clamp(min=1e-6)
        scale = cos.pow(self.B - 1.0).unsqueeze(-1)

        msg = h[col] * scale

        out = torch.zeros_like(h)
        out_flat = out.view(N, -1)
        msg_flat = msg.view(msg.size(0), -1)
        out_flat.index_add_(0, row, msg_flat)

        out = out_flat.view(N, -1)
        out = self.bcos_out(out)
        out = self.ln(out)

        return F.dropout(out, p=self.dropout, training=self.training)

# -----------------------------------------------------------
# Models
# -----------------------------------------------------------
class BaselineGAT(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, heads, dropout):
        super().__init__()
        self.g1 = GATConv(in_ch, hidden, heads=heads, dropout=dropout)
        self.g2 = GATConv(hidden * heads, out_ch, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.g1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.g2(x, edge_index)


class BCosGAT(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, heads, B, dropout):
        super().__init__()
        self.l1 = InterpretableBCosGATLayer(
            in_ch, hidden, heads=heads, B=B, dropout=dropout
        )
        self.l2 = InterpretableBCosGATLayer(
            hidden * heads, out_ch, heads=1, B=B, dropout=dropout
        )

    def forward(self, x, edge_index):
        x = self.l1(x, edge_index)
        x = F.elu(x)
        x = self.l2(x, edge_index)
        return x

# -----------------------------------------------------------
# Train / Eval
# -----------------------------------------------------------
def train_epoch(model, data, opt, split_id):
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    mask = data.train_mask[:, split_id]
    loss = F.cross_entropy(out[mask], data.y[mask])
    loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, split_id):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f"{split}_mask")[:, split_id]
        acc[split] = pred[mask].eq(data.y[mask]).sum().item() / int(mask.sum())
    return acc

# -----------------------------------------------------------
# Main Runner
# -----------------------------------------------------------
def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for dataset_name in ['Texas', 'Cornell']:
        print(f"\n================ Dataset: {dataset_name} ================")
        cfg = HYPERPARAMS[dataset_name]

        dataset = WebKB(root='./data', name=dataset_name)
        data = dataset[0]
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

        data.x = F.normalize(data.x, p=2, dim=1)
        data = data.to(device)

        in_ch = dataset.num_node_features
        out_ch = dataset.num_classes

        for split_id in SPLITS:

            # ---------- Baseline GAT ----------
            base_tests = []
            for seed in SEEDS:
                set_seed(seed)
                model = BaselineGAT(
                    in_ch, cfg['hidden'], out_ch,
                    cfg['heads'], cfg['dropout']
                ).to(device)
                opt = torch.optim.Adam(
                    model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
                )

                for _ in range(cfg['epochs']):
                    train_epoch(model, data, opt, split_id)

                acc = evaluate(model, data, split_id)
                base_tests.append(acc['test'])

            print(
                f"[Split {split_id}] Baseline GAT | "
                f"Test: {np.mean(base_tests):.4f} ± {np.std(base_tests):.4f}"
            )

            # ---------- BCos-GAT ----------
            best_val, best_B, best_test = -1, None, None

            for B in B_GRID:
                vals, tests = [], []
                for seed in SEEDS:
                    set_seed(seed)
                    model = BCosGAT(
                        in_ch, cfg['hidden'], out_ch,
                        cfg['heads'], B, cfg['dropout']
                    ).to(device)
                    opt = torch.optim.Adam(
                        model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
                    )

                    for _ in range(cfg['epochs']):
                        train_epoch(model, data, opt, split_id)

                    acc = evaluate(model, data, split_id)
                    vals.append(acc['val'])
                    tests.append(acc['test'])

                print(
                    f"[Split {split_id}] BCos-GAT B={B} | "
                    f"Test: {np.mean(tests):.4f} ± {np.std(tests):.4f} | "
                    f"Val: {np.mean(vals):.4f}"
                )

                if np.mean(vals) > best_val:
                    best_val = np.mean(vals)
                    best_B = B
                    best_test = (np.mean(tests), np.std(tests))

            print(
                f"[Split {split_id}] Best BCos-GAT (B={best_B}) | "
                f"Test: {best_test[0]:.4f} ± {best_test[1]:.4f}"
            )

if __name__ == '__main__':
    run()
