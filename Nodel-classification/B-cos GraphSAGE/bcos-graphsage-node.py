#!/usr/bin/env python3
"""
bcos_graphsage_node_cuda.py

Compare Baseline GraphSAGE vs BCos-GraphSAGE on node classification (Planetoid datasets).
- CUDA-first (requires CUDA)
- Per-dataset tuned defaults, B-grid search
- Reports mean ± std accuracy and loss across n_runs
- BCos-GraphSAGE implements B-cos scaling on messages and provides per-node contribution map for interpretability

Usage:
  python bcos_graphsage_node_cuda.py --dataset Cora --device cuda:0 --n_runs 5

Supports dataset names: Cora, CiteSeer, PubMed OR Quora (alias for Cora), CiteShare (alias for CiteSeer).
"""
import os
import argparse
import random
import json
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dataset_alias(name: str) -> str:
    name = name.strip()
    if name.lower() in ('quora',):
        return 'Cora'
    if name.lower() in ('citeshare', 'citesear', 'citeseer', 'citeseer'):
        return 'CiteSeer'
    # otherwise pass through (expect 'Cora','CiteSeer','PubMed')
    return name

# -------------------------
# B-cos Linear (bias-free)
# -------------------------
class BcosLinearNoBias(nn.Module):
    """
    Bias-free linear + B-cos scaling:
      lin = x @ W.T
      cos = normalize(x) @ normalize(W).T
      cos clamped to [eps, 1]
      out = lin * |cos|^{B-1}
    """
    def __init__(self, in_features: int, out_features: int, B: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.B = float(B)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / max(1.0, np.sqrt(in_features))))

    def forward(self, x: torch.Tensor):
        # x: (N, in_features)
        lin = x @ self.weight.t()  # (N, out_features)
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cos = (x_norm @ w_norm.t()).clamp(min=self.eps, max=1.0)  # (N, out)
        scale = cos.abs().pow(self.B - 1.0)  # (N, out)
        return lin * scale

# -------------------------
# BCos-GraphSAGE Layer (custom mean aggregator)
# -------------------------
class BCosSAGELayer(nn.Module):
    """
    GraphSAGE-like layer but messages scaled by B-cos factor computed from projected features.

    For each edge (u->v):
      src_proj = W_src x_u
      tgt_proj = W_tgt x_v  (we use same projection & normalize by row for cosine)
      cos = cos(src_proj, tgt_proj)  (clamp)
      msg = lin(src) * |cos|^{B-1}
    Aggregation: mean over neighbors (degree-normalized).
    """
    def __init__(self, in_channels: int, out_channels: int, B: float = 2.0, eps: float = 1e-6, use_residual: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.B = float(B)
        self.eps = eps
        # value projection (bias-free) used for message content
        self.value = nn.Linear(in_channels, out_channels, bias=False)
        # For output projection we use BcosLinearNoBias for interpretability (bias-free)
        self.out_proj = BcosLinearNoBias(out_channels, out_channels, B=self.B, eps=self.eps)
        self.use_residual = use_residual
        if use_residual:
            self.res_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.value.weight)
        if self.use_residual:
            nn.init.xavier_uniform_(self.res_proj.weight)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1.0 / max(1.0, np.sqrt(self.out_proj.in_features)))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, in_channels)
        edge_index: (2, E) with row=target, col=source (PyG convention)
        Returns:
          out (N, out_channels)
          contrib_map (E,) normalized contributions per edge (useful for interpretability)
        """
        device = x.device
        N = x.size(0)
        row, col = edge_index  # target <- source

        # projected representations
        src_val = self.value(x)           # (N, out)
        # for cosine we use normalized projected vectors (value vectors)
        src_norm = F.normalize(src_val, p=2, dim=1)  # (N, out)
        tgt_norm = src_norm  # we use same projection (GraphSAGE style); cosine between nodes

        src_feat = src_norm[col]   # (E, out)
        tgt_feat = tgt_norm[row]   # (E, out)

        # cosine per edge
        cos = (src_feat * tgt_feat).sum(dim=1).clamp(min=self.eps, max=1.0)  # (E,)

        # message = original (non-normalized) src_val scaled by B-cos factor (per edge)
        src_val_edge = src_val[col]  # (E, out)
        scale = cos.abs().pow(self.B - 1.0).unsqueeze(1)  # (E,1)
        messages = src_val_edge * scale  # (E, out)

        # aggregate: sum per target then divide by degree -> mean
        out = torch.zeros((N, self.out_channels), device=device)
        # accumulate: flatten messages and index_add
        E = row.size(0)
        out_flat = out.view(N, -1)
        messages_flat = messages.view(E, -1)
        out_flat.index_add_(0, row, messages_flat)  # sum
        out = out_flat.view(N, self.out_channels)

        # degree normalization (mean)
        deg = torch.zeros(N, device=device)
        deg.index_add_(0, row, torch.ones(E, device=device))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        out = out / deg

        # out projection B-cos (interpretability preserving) + residual
        out_bcos = self.out_proj(out)  # (N, out)
        if self.use_residual:
            res = self.res_proj(x)
            out_final = out_bcos + res
        else:
            out_final = out_bcos
        # contribution map per edge (normalized by target-degree) for interpretability:
        # contribution magnitude from source u to target v is || message ||_1 (or L2) normalized per target
        contrib_mag = messages.abs().sum(dim=1)  # (E,)
        # normalize contributions to sum=1 per target node for easy interpretation
        contrib_norm = torch.zeros_like(contrib_mag, device=device)
        # sum per target
        sum_per_target = torch.zeros(N, device=device)
        sum_per_target.index_add_(0, row, contrib_mag)
        # avoid divide by zero
        denom = sum_per_target[row].clamp(min=1e-12)
        contrib_norm = contrib_mag / denom  # each edge's contribution fraction to its target
        return out_final, contrib_norm.detach()

# -------------------------
# BCos-GraphSAGE Model (stack 2 layers)
# -------------------------
class BCosGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_classes: int, B: float = 2.0, use_residual: bool = True):
        super().__init__()
        self.l1 = BCosSAGELayer(in_channels, hidden, B=B, use_residual=use_residual)
        # second layer maps hidden -> num_classes
        self.l2 = BCosSAGELayer(hidden, num_classes, B=B, use_residual=use_residual)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        h1, contrib1 = self.l1(x, edge_index)
        h1 = self.act(h1)
        h2, contrib2 = self.l2(h1, edge_index)
        return h2, (contrib1, contrib2)

# -------------------------
# Baseline GraphSAGE (PyG SAGEConv)
# -------------------------
class BaselineGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_classes: int):
        super().__init__()
        self.s1 = SAGEConv(in_channels, hidden)
        self.s2 = SAGEConv(hidden, num_classes)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        h = self.s1(x, edge_index)
        h = self.act(h)
        h = self.s2(h, edge_index)
        return h

# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch_baseline(model: nn.Module, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss.item()

def train_one_epoch_bcos(model: nn.Module, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_baseline(model: nn.Module, data) -> Dict[str, float]:
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    stats = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f"{split}_mask")
        acc = pred[mask].eq(data.y[mask]).sum().item() / int(mask.sum().item())
        loss = F.cross_entropy(out[mask], data.y[mask]).item()
        stats[f"{split}_acc"] = acc
        stats[f"{split}_loss"] = loss
    return stats

@torch.no_grad()
def evaluate_bcos(model: nn.Module, data) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    model.eval()
    out, (c1, c2) = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    stats = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f"{split}_mask")
        acc = pred[mask].eq(data.y[mask]).sum().item() / int(mask.sum().item())
        loss = F.cross_entropy(out[mask], data.y[mask]).item()
        stats[f"{split}_acc"] = acc
        stats[f"{split}_loss"] = loss
    contribs = {'layer1': c1, 'layer2': c2}
    return stats, contribs

# -------------------------
# Runner per-dataset with B-grid
# -------------------------
def run_dataset(dataset_name: str,
                device_str: str = 'cuda:0',
                B_grid: List[float] = None,
                n_runs: int = 5,
                seed: int = 42,
                save_dir: str = 'results') -> Dict:
    if B_grid is None:
        B_grid = [1.0, 1.5, 2.0, 2.5, 3.0]

    name = dataset_alias(dataset_name)
    assert name in ('Cora', 'CiteSeer', 'PubMed'), f"Unknown dataset {dataset_name}"
    # per-dataset defaults
    defaults = {
        'Cora':    {'hidden': 64, 'lr': 0.005, 'wd': 5e-4, 'epochs': 200},
        'CiteSeer':{'hidden': 64, 'lr': 0.005, 'wd': 5e-4, 'epochs': 300},
        'PubMed':  {'hidden': 128,'lr': 0.005, 'wd': 5e-4, 'epochs': 300}
    }
    cfg = defaults[name]

    device = torch.device(device_str)
    print(f"\n=== Dataset: {name} | device={device} | runs={n_runs} | epochs={cfg['epochs']} ===")
    dataset = Planetoid(root=f'./data/{name}', name=name)
    data = dataset[0].to(device)
    in_channels = dataset.num_node_features
    num_classes = dataset.num_classes

    # Baseline multi-run
    base_test_accs = []
    base_test_losses = []
    base_val_accs = []
    base_train_losses = []

    for run in range(n_runs):
        set_seed(seed + run)
        model = BaselineGraphSAGE(in_channels, cfg['hidden'], num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        last_loss = None
        for ep in range(cfg['epochs']):
            last_loss = train_one_epoch_baseline(model, data, opt)
        stats = evaluate_baseline(model, data)
        base_test_accs.append(stats['test_acc'])
        base_test_losses.append(stats['test_loss'])
        base_val_accs.append(stats['val_acc'])
        base_train_losses.append(last_loss)
        if run == 0:
            print(f"[Baseline][run 1] final train loss {last_loss:.4f} | val {stats['val_acc']:.4f} test {stats['test_acc']:.4f}")

    base_test_mean = float(np.mean(base_test_accs))
    base_test_std = float(np.std(base_test_accs, ddof=0))
    base_test_loss_mean = float(np.mean(base_test_losses))
    base_val_mean = float(np.mean(base_val_accs))
    print(f"[Baseline] test mean ± std: {base_test_mean:.4f} ± {base_test_std:.4f} | val mean: {base_val_mean:.4f} | test loss mean: {base_test_loss_mean:.4f}")

    # BCos grid
    b_results = []
    for B in B_grid:
        val_accs = []
        test_accs = []
        test_losses = []
        train_losses = []
        for run in range(n_runs):
            set_seed(seed + 1000 + run)
            model = BCosGraphSAGE(in_channels, cfg['hidden'], num_classes, B=B, use_residual=True).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
            last_loss = None
            for ep in range(cfg['epochs']):
                last_loss = train_one_epoch_bcos(model, data, opt)
            stats, contribs = evaluate_bcos(model, data)
            val_accs.append(stats['val_acc'])
            test_accs.append(stats['test_acc'])
            test_losses.append(stats['test_loss'])
            train_losses.append(last_loss)
            if run == 0:
                # print a sample of contribution distribution for first run, layer1 (mean)
                mean_contrib = float(contribs['layer1'].mean().cpu().numpy())
                print(f"[B={B}][run1] sample mean edge-contrib (layer1): {mean_contrib:.4f}")
        entry = {
            'B': float(B),
            'val_mean': float(np.mean(val_accs)),
            'val_std': float(np.std(val_accs, ddof=0)),
            'test_mean': float(np.mean(test_accs)),
            'test_std': float(np.std(test_accs, ddof=0)),
            'test_loss_mean': float(np.mean(test_losses)),
            'train_loss_mean': float(np.mean(train_losses))
        }
        b_results.append(entry)
        print(f"[B={B}] val {entry['val_mean']:.4f}±{entry['val_std']:.4f} | test {entry['test_mean']:.4f}±{entry['test_std']:.4f} | test_loss {entry['test_loss_mean']:.4f}")

    # pick best B by val_mean
    b_results_sorted = sorted(b_results, key=lambda x: x['val_mean'], reverse=True)
    best = b_results_sorted[0]

    print("\n=== SUMMARY ===")
    print(f"Dataset: {name}")
    print(f"Baseline GraphSAGE test mean ± std : {base_test_mean:.4f} ± {base_test_std:.4f} | test loss mean: {base_test_loss_mean:.4f}")
    print(f"BCos-GraphSAGE selected B={best['B']} val mean {best['val_mean']:.4f} test mean ± std {best['test_mean']:.4f} ± {best['test_std']:.4f} | test loss mean: {best['test_loss_mean']:.4f}")

    # save JSON
    os.makedirs(save_dir, exist_ok=True)
    out = {
        'dataset': name,
        'device': str(device),
        'baseline': {
            'test_mean': base_test_mean,
            'test_std': base_test_std,
            'test_loss_mean': base_test_loss_mean,
            'val_mean': base_val_mean
        },
        'b_grid': b_results_sorted,
        'selected_B': best['B']
    }
    outpath = os.path.join(save_dir, f"{name}_bcos_graphsage_summary_cuda.json")
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved summary to {outpath}")

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return out

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='PubMed', help='Cora / CiteSeer / PubMed ')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--n_runs', type=int, default=5)
    p.add_argument('--bgrid', nargs='+', type=float, default=[1.0,1.1,1.2,1.3,1.5,1.7,2.0])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_dir', type=str, default='results')
    args = p.parse_args()

    # require CUDA as requested
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required by this script. Enable CUDA or edit script to allow CPU.")

    set_seed(args.seed)
    run_dataset(args.dataset, device_str=args.device, B_grid=args.bgrid, n_runs=args.n_runs, seed=args.seed, save_dir=args.save_dir)
