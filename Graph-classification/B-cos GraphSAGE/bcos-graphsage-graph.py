#!/usr/bin/env python3
"""
B-cos-GraphSAGE training script for MUTAG and PROTEINS.

Run examples:
  python bcos_graphsage_unified.py --dataset MUTAG
  python bcos_graphsage_unified.py --dataset PROTEINS

Outputs:
 - Per-run logs
 - Mean ± std test accuracy and loss for Standard GraphSAGE and B-cos GraphSAGE (swept B)
 - Prints best B where B-cos >= standard (if any)
"""

import argparse
import math
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, degree
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def summarize(arr):
    a = np.array(arr, dtype=float)
    return float(a.mean()), float(a.std())

# -------------------------
# B-cos Linear (row-normalized)
# -------------------------
class BcosLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, B=2.0, eps=1e-8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.B = float(B)
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * math.sqrt(2.0 / (in_features + out_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: (N, D)
        if x.numel() == 0:
            return x.new_zeros((0, self.out_features))
        x_norm = x.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)  # (N,1)
        w = self.weight  # (out, in)
        w_norm = w.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)  # (out,1)
        w_hat = w / w_norm  # (out, in)
        dot = torch.matmul(x, w_hat.t())  # (N, out)
        cos = dot / x_norm  # (N, out)
        cos_sign = torch.sign(cos)
        cos_abs_pow = torch.abs(cos).clamp_min(self.eps).pow(self.B)
        out = x_norm * cos_sign * cos_abs_pow
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out

# -------------------------
# CustomSAGEConv (message-level B-cos)
# -------------------------
class CustomSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bcos=False, B=2.0, bias=True, use_edge_attr=False, edge_attr_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bcos = use_bcos
        self.B = B
        self.use_edge_attr = use_edge_attr
        if use_bcos:
            self.lin_msg = BcosLinear(in_channels, out_channels, bias=bias, B=B)
            self.lin_self = BcosLinear(in_channels, out_channels, bias=bias, B=B)
        else:
            self.lin_msg = nn.Linear(in_channels, out_channels, bias=bias)
            self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        # optional projection for edge attrs to message dim
        if use_edge_attr and edge_attr_dim is not None:
            self.edge_proj = nn.Linear(edge_attr_dim, out_channels, bias=bias)
        else:
            self.edge_proj = None

    def forward(self, x, edge_index, edge_attr=None):
        # edge_index: [2, E] (src, dst)
        row, col = edge_index
        # msgs computed from source nodes
        msgs_src = self.lin_msg(x)  # (N, out)
        # if edge_attr present, add projected edge embedding to per-edge messages
        if (edge_attr is not None) and (self.edge_proj is not None):
            # shape: (E, out)
            edge_msg = self.edge_proj(edge_attr)
            per_edge_msgs = msgs_src[row] + edge_msg
        else:
            per_edge_msgs = msgs_src[row]  # (E, out)
        # aggregate per target node (mean)
        agg = scatter(per_edge_msgs, col, dim=0, dim_size=x.size(0), reduce='mean')
        self_out = self.lin_self(x)
        out = self_out + agg
        # optionally record last messages for attribution (non-grad)
        if getattr(self, "_record_messages", False):
            # store per-node incoming aggregated message and raw per-edge messages if needed
            self._last_agg = agg.detach().cpu()
            # store msgs_src for later use
            self._last_msgs_src = msgs_src.detach().cpu()
        return out

# -------------------------
# GraphSAGENet (stacked CustomSAGEConv)
# -------------------------
class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 use_bcos=False, B=2.0, dropout=0.2, use_layernorm=True, use_edge_attr=False, edge_attr_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.use_bcos = use_bcos
        self.B = B
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # first
        self.convs.append(CustomSAGEConv(in_channels, hidden_channels, use_bcos=use_bcos, B=B,
                                         use_edge_attr=use_edge_attr, edge_attr_dim=edge_attr_dim))
        self.norms.append(nn.LayerNorm(hidden_channels) if use_layernorm else nn.BatchNorm1d(hidden_channels))
        # remaining
        for _ in range(num_layers - 1):
            self.convs.append(CustomSAGEConv(hidden_channels, hidden_channels, use_bcos=use_bcos, B=B,
                                             use_edge_attr=use_edge_attr, edge_attr_dim=edge_attr_dim))
            self.norms.append(nn.LayerNorm(hidden_channels) if use_layernorm else nn.BatchNorm1d(hidden_channels))
        # head
        self.pool_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None, record_messages=False):
        # x: (N_total, F)
        for i, conv in enumerate(self.convs):
            # pass edge_attr unchanged (we use same edge_attr for every layer)
            x = conv(x, edge_index, edge_attr=edge_attr)
            # record per-layer node feats if requested (only last layer needed usually)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # save last node features for attribution (detach)
        if record_messages:
            self._last_node_feats = x.detach().cpu()
            # per-conv messages/agg could be recorded within convs if needed
        x = global_mean_pool(x, batch)
        out = self.pool_lin(x)
        return out

# -------------------------
# Training & evaluation
# -------------------------
def train_epoch(model, loader, optimizer, device, criterion, clip=5.0):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        out = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
        loss = criterion(out, data.y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += float(loss.item()) * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        out = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
        loss = criterion(out, data.y.to(device))
        total_loss += float(loss.item()) * data.num_graphs
        preds = out.argmax(dim=-1)
        correct += int((preds == data.y.to(device)).sum())
        total += data.num_graphs
    return correct / total, total_loss / total

# -------------------------
# Helpers: preprocess & splits
# -------------------------
def ensure_node_features(dataset, mode="degree"):
    """If no node features, create simple structural features.
       mode: "degree" (1-d), "deg+const" (2-d), "deg+clust+const" (3-d)"""
    if dataset.num_node_features > 0:
        return dataset
    print("No node features found — adding structural features (mode=%s)." % mode)
    for g in dataset:
        E = g.edge_index
        deg = degree(E[1], num_nodes=g.num_nodes).view(-1,1)
        if mode == "degree":
            g.x = deg
        elif mode == "deg+const":
            g.x = torch.cat([deg, torch.ones((g.num_nodes,1))], dim=1)
        else:
            g.x = torch.cat([deg, torch.ones((g.num_nodes,1))], dim=1)
    return dataset

def build_loaders_from_indices(dataset, idx_tuple, batch_size, shuffle_train=True):
    train_idx, val_idx, test_idx = idx_tuple
    train_ds = [dataset[i] for i in train_idx]
    val_ds = [dataset[i] for i in val_idx]
    test_ds = [dataset[i] for i in test_idx]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# -------------------------
# Main experiment orchestration
# -------------------------
def run_experiment(dataset_name="MUTAG",
                   B_values=[1.0,1.5,2.0,2.5],
                   runs=5,
                   device=None):
    # Prefer CUDA > MPS > CPU
    if device is None:
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        except (RuntimeError, AssertionError):
            device = torch.device("cpu")
    else:
        try:
            device = torch.device(device)
        except (RuntimeError, AssertionError):
            device = torch.device("cpu")
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    # load dataset
    ds = TUDataset(root="data/%s" % dataset_name, name=dataset_name)
    # ensure features
    ds = ensure_node_features(ds, mode="deg+const")
    in_channels = ds.num_node_features or 1
    num_classes = ds.num_classes
    print(f"Dataset {dataset_name}: graphs={len(ds)}, node_features={in_channels}, classes={num_classes}")

    # dataset-specific hyperparams
    if dataset_name.upper() == "MUTAG":
        hidden = 128
        batch_size = 16
        epochs = 200
        lr = 5e-4
        weight_decay = 0.0
    elif dataset_name.upper() == "PROTEINS":
        hidden = 160
        batch_size = 32
        epochs = 150
        lr = 8e-4
        weight_decay = 1e-4
    else:
        # generic
        hidden = 128
        batch_size = 32
        epochs = 150
        lr = 5e-4
        weight_decay = 0.0

    # prepare splits: for PROTEINS use 10-fold CV, for MUTAG use repeated stratified shuffles
    indices_splits = []
    n = len(ds)
    if dataset_name.upper() == "PROTEINS":
        # 10-fold indices
        k = 10
        idxs = np.arange(n)
        random_state = np.random.RandomState(42)
        random_state.shuffle(idxs)
        folds = np.array_split(idxs, k)
        for i in range(k):
            test_idx = list(folds[i])
            train_val = np.concatenate([folds[j] for j in range(k) if j != i])
            # split train_val into train/val 90/10
            n_tv = len(train_val)
            n_train = int(0.9 * n_tv)
            train_idx = list(train_val[:n_train])
            val_idx = list(train_val[n_train:])
            indices_splits.append((train_idx, val_idx, test_idx))
        runs_effective = len(indices_splits)
    else:
        # MUTAG or others: repeated random splits (same seeds for reproducibility)
        runs_effective = runs
        base_seed = 42
        for run in range(runs_effective):
            s = base_seed + run
            rng = np.random.RandomState(s)
            perm = list(rng.permutation(n))
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:n_train+n_val]
            test_idx = perm[n_train+n_val:]
            indices_splits.append((train_idx, val_idx, test_idx))

    device_cpu = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()

    # --- Standard GraphSAGE baseline (same arch but use_bcos=False)
    std_accs = []
    std_losses = []
    print("\n=== Running STANDARD GraphSAGE baseline ===")
    for run_no, split in enumerate(indices_splits):
        seed = 100 + run_no
        set_seed(seed)
        train_loader, val_loader, test_loader = build_loaders_from_indices(ds, split, batch_size)
        model = GraphSAGENet(in_channels, hidden, num_classes, num_layers=2, use_bcos=False,
                             B=1.0, dropout=0.2, use_layernorm=True,
                             use_edge_attr=False, edge_attr_dim=None).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val = -1.0
        best_state = None
        for epoch in range(1, epochs+1):
            _ = train_epoch(model, train_loader, optimizer, device, criterion, clip=5.0)
            val_acc, _ = evaluate(model, val_loader, device, criterion)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        # load best and test
        model.load_state_dict(best_state)
        test_acc, test_loss = evaluate(model, test_loader, device, criterion)
        std_accs.append(test_acc)
        std_losses.append(test_loss)
        print(f"[STANDARD] run {run_no+1}/{len(indices_splits)} seed={seed} best_val={best_val:.4f} test_acc={test_acc:.4f} test_loss={test_loss:.4f}")

    std_mean_acc, std_std_acc = summarize(std_accs)
    std_mean_loss, std_std_loss = summarize(std_losses)
    print(f"\nSTANDARD GraphSAGE -> acc: {std_mean_acc:.4f} ± {std_std_acc:.4f}, loss: {std_mean_loss:.4f} ± {std_std_loss:.4f}")

    # --- B-cos GraphSAGE sweep
    b_results = {}
    print("\n=== Running B-cos GraphSAGE sweep ===")
    for B in B_values:
        accs = []
        losses = []
        print(f"\n--- B = {B} ---")
        for run_no, split in enumerate(indices_splits):
            seed = 2000 + run_no
            set_seed(seed)
            train_loader, val_loader, test_loader = build_loaders_from_indices(ds, split, batch_size)
            model = GraphSAGENet(in_channels, hidden, num_classes, num_layers=2, use_bcos=True, B=B,
                                 dropout=0.2, use_layernorm=True, use_edge_attr=False, edge_attr_dim=None).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            best_val = -1.0
            best_state = None
            for epoch in range(1, epochs+1):
                _ = train_epoch(model, train_loader, optimizer, device, criterion, clip=5.0)
                val_acc, _ = evaluate(model, val_loader, device, criterion)
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best_state)
            test_acc, test_loss = evaluate(model, test_loader, device, criterion)
            accs.append(test_acc)
            losses.append(test_loss)
            print(f"[BCOS B={B}] run {run_no+1}/{len(indices_splits)} seed={seed} best_val={best_val:.4f} test_acc={test_acc:.4f} test_loss={test_loss:.4f}")
        mean_acc, std_acc = summarize(accs)
        mean_loss, std_loss = summarize(losses)
        b_results[B] = (mean_acc, std_acc, mean_loss, std_loss)
        print(f"[B={B}] mean_acc={mean_acc:.4f} ± {std_acc:.4f}, loss={mean_loss:.4f} ± {std_loss:.4f}")

    # --- final summary
    print("\n================ FINAL SUMMARY ================")
    print(f"Standard GraphSAGE  - acc: {std_mean_acc:.4f} ± {std_std_acc:.4f}, loss: {std_mean_loss:.4f} ± {std_std_loss:.4f}")
    best_B = None
    best_acc = -1.0
    for B, (macc, sacc, mloss, sloss) in sorted(b_results.items(), key=lambda x: x[0]):
        print(f"B-cos (B={B}) - acc: {macc:.4f} ± {sacc:.4f}, loss: {mloss:.4f} ± {sloss:.4f}")
        if macc > best_acc:
            best_acc = macc
            best_B = B

    if best_acc >= std_mean_acc:
        print(f"\n=> Best B-cos (B={best_B}) matches or beats standard (abs improvement {best_acc-std_mean_acc:.4f}).")
    else:
        print(f"\n=> Best B-cos (B={best_B}) did NOT beat standard in this experiment (best acc {best_acc:.4f}).")

    # optional plot
    try:
        Bs = sorted(list(b_results.keys()))
        accs_plot = [b_results[B][0] for B in Bs]
        errs = [b_results[B][1] for B in Bs]
        plt.errorbar(Bs, accs_plot, yerr=errs, marker='o', capsize=5)
        plt.axhline(std_mean_acc, linestyle='--', color='gray', label='Standard GraphSAGE')
        plt.xlabel('B value')
        plt.ylabel('Mean Test Accuracy')
        plt.title(f'{dataset_name} — B-cos GraphSAGE vs Standard')
        plt.legend()
        outpng = f"bcos_vs_standard_{dataset_name}.png"
        plt.savefig(outpng, dpi=200, bbox_inches='tight')
        print(f"Plot saved to {outpng}")
    except Exception as e:
        print("Plotting failed:", e)

    return {
        "standard": (std_mean_acc, std_std_acc, std_mean_loss, std_std_loss),
        "b_results": b_results,
        "best_B": best_B
    }

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MUTAG", choices=["MUTAG","PROTEINS"],
                        help="Dataset to run")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu (auto-detect if None)")
    args = parser.parse_args()
    dev = None
    if args.device:
        try:
            dev = torch.device(args.device)
        except (RuntimeError, AssertionError):
            print(f"Warning: device '{args.device}' not available, falling back to auto-detect")
            dev = None
    run_experiment(dataset_name=args.dataset, B_values=[1.0,1.5,2.0,2.5], runs=5, device=dev)
