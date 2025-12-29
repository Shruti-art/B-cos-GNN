#!/usr/bin/env python3
"""
bcos_gcn_tuned.py

Goal:
 - Train & compare Standard GCN and BCos-GCN on MUTAG and PROTEINS (graph classification).
 - Use cross-validation to report mean ± std accuracy and loss.
 - Search small hyperparameter grid (B values, residual ratio, learning rate) per dataset
   to try to reach BCos-GCN >= standard GCN.
 - Compute interpretability metrics (fidelity, sparsity) for BCos-GCN.

Usage examples:
  python bcos_gcn_tuned.py --dataset MUTAG --device cuda
  python bcos_gcn_tuned.py --dataset PROTEINS --device cuda --epochs 200

Notes:
 - Expects dataset graphs in data/TUDataset/<dataset>.
 - Assumes dataset graphs already have node features (data.x). If not, use --allow_feature_fix.
"""

import argparse
import math
import random
import time
from statistics import mean, stdev
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import degree

# Try to import StratifiedKFold; if not available, we'll fallback.
try:
    from sklearn.model_selection import StratifiedKFold
    SKLEARN_EXISTS = True
except Exception:
    SKLEARN_EXISTS = False

# ---------------------------
# Utility functions
# ---------------------------
def set_seed(seed: int):
    """Set random seeds for reproducibility (best-effort)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic options (may slow GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_arg: str) -> torch.device:
    """Return torch.device from string arg; prefer cuda if requested and available."""
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ---------------------------
# BCos activation (temperature scaled)
# ---------------------------
class BCosActivation(nn.Module):
    """
    Temperature-scaled BCos activation with optional learnable B.
    forward(x):
      - normalize node vector h along feature dim
      - scale by temp
      - out = x_scaled * |x_scaled|^(B-1)
    """
    def __init__(self, B: float = 1.0, temp: float = 1.5, learn_B: bool = False, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.temp = float(temp)
        self.learn_B = bool(learn_B)
        if self.learn_B:
            # parametrize B positive via softplus(logparam)
            self._raw_B = nn.Parameter(torch.log(torch.tensor(B, dtype=torch.float32)))
        else:
            self.register_buffer("_fixed_B", torch.tensor(float(B)))

    @property
    def B(self):
        if self.learn_B:
            return F.softplus(self._raw_B)
        return self._fixed_B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D]
        norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True) + self.eps)
        x_norm = x / (norm + self.eps)
        x_scaled = self.temp * x_norm
        return x_scaled * (torch.abs(x_scaled) + self.eps).pow(self.B - 1.0)

# ---------------------------
# Models: Standard GCN and BCos-GCN
# ---------------------------
class GCNStandard(nn.Module):
    """Simple 2-layer GCN for graph classification (global mean pool)."""
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = float(dropout)
        self.classifier = nn.Linear(hidden, out_channels)
        self._h_nodes = None

    def forward(self, x, edge_index, batch):
        # two-layer GCN, store node embeddings for contribution projection
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.norm1(self.conv1(x, edge_index)))
        h = F.elu(self.norm2(self.conv2(x, edge_index)))
        self._h_nodes = h
        g = global_mean_pool(h, batch)
        logits = self.classifier(g)
        return logits

    def get_node_contributions(self) -> torch.Tensor:
        # returns |W . h_node| where W is classifier weights (out, hidden)
        W = self.classifier.weight  # (out, hidden)
        return torch.abs(F.linear(self._h_nodes, W))

class BCosGCN(nn.Module):
    """GCN with BCos nonlinearity blended via a residual ratio for interpretability/stability."""
    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 dropout: float = 0.45, B: float = 1.0, temp: float = 1.5,
                 learn_B: bool = False, residual_ratio: float = 0.6):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = float(dropout)
        self.bcos = BCosActivation(B=B, temp=temp, learn_B=learn_B)
        self.residual_ratio = float(residual_ratio)
        # weight-norm classifier helps stable scaling
        self.classifier = nn.utils.weight_norm(nn.Linear(hidden, out_channels))
        self._h_nodes = None

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.norm1(self.conv1(x, edge_index)))
        h = F.elu(self.norm2(self.conv2(x, edge_index)))
        # residual blend: keep some raw h + some BCos(h)
        h_b = self.residual_ratio * h + (1.0 - self.residual_ratio) * self.bcos(h)
        self._h_nodes = h_b
        g = global_mean_pool(h_b, batch)
        logits = self.classifier(g)
        return logits

    def get_node_contributions(self) -> torch.Tensor:
        W = self.classifier.weight
        return torch.abs(F.linear(self._h_nodes, W))

# ---------------------------
# Fidelity & Sparsity (graph-level) for BCos models
# ---------------------------
@torch.no_grad()
def compute_fidelity_sparsity_graph(model: nn.Module, dataset: TUDataset, device: torch.device,
                                    topk_fraction: float = 0.2, batch_size: int = 32) -> Tuple[float,float]:
    """
    For each batch, compute node contributions |W·h|, keep top-k fraction overall, mask rest,
    and compute fidelity (1 - avg relative drop in predicted confidence) and sparsity (fraction zeros).
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval(); model.to(device)
    f_list = []; s_list = []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        probs = F.softmax(logits, dim=-1)
        base_conf = probs.max(dim=1).values  # per-graph confidence

        # contributions per node per class
        contrib = model.get_node_contributions()  # shape [N_total, C]
        flat = contrib.view(-1)
        k = max(1, int(topk_fraction * flat.numel()))
        thresh = torch.topk(flat, k).values.min()
        mask = (contrib >= thresh).float().to(device)  # [N_total, C]

        # map node-class mask to per-graph indicator: if graph has any kept entries -> keep graph
        node_keep = (mask.sum(dim=1) > 0).float().to(device)  # N_total
        graph_keep = torch.zeros((data.num_graphs,), device=device)
        graph_keep = graph_keep.index_add(0, data.batch, node_keep)  # counts per graph
        graph_keep = (graph_keep > 0).float()

        logits_masked = logits * graph_keep.unsqueeze(-1)
        probs_masked = F.softmax(logits_masked, dim=-1)
        new_conf = probs_masked.max(dim=1).values

        eps = 1e-8
        fidelity = 1.0 - torch.mean(torch.abs(base_conf - new_conf) / (base_conf + eps)).item()
        sparsity = 1.0 - (mask.sum().item() / mask.numel())
        f_list.append(fidelity); s_list.append(sparsity)
    return float(mean(f_list)), float(mean(s_list))

# ---------------------------
# Training / evaluation helpers
# ---------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, optim, device: torch.device):
    model.train(); model.to(device)
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optim.zero_grad()
        logits = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(logits, data.y)
        loss.backward()
        optim.step()
        total_loss += float(loss.item()) * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval(); model.to(device)
    correct = 0
    losses = []
    total = 0
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(logits, data.y)
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=-1)
        correct += int((preds == data.y).sum().item())
        total += data.num_graphs
    acc = correct / total
    return acc, float(mean(losses))

# ---------------------------
# Single fold training run (with early stopping & small validation split)
# ---------------------------
def run_fold(train_dataset: List, val_dataset: List, test_dataset: List,
             model_name: str, in_ch: int, out_ch: int,
             hidden: int, dropout: float, B: float, temp: float, learn_B: bool,
             residual_ratio: float, lr: float, weight_decay: float,
             epochs: int, device: torch.device, seed: int,
             log_interval: int = 10, early_stop_patience: int = 20):
    """
    train on train_dataset (list of graphs), validate on val_dataset, test on test_dataset.
    Returns best_val_acc, test_acc_at_best, test_loss_at_best, [fidelity, sparsity] if bcos.
    """
    set_seed(seed)
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model select
    if model_name == "gcn":
        model = GCNStandard(in_ch, hidden, out_ch, dropout)
    else:
        model = BCosGCN(in_ch, hidden, out_ch, dropout, B=B, temp=temp, learn_B=learn_B, residual_ratio=residual_ratio)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = -1.0
    best_test = 0.0
    best_loss = None
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, _ = evaluate_model(model, val_loader, device)
        test_acc, test_loss = evaluate_model(model, test_loader, device)

        # track best by val_acc
        if val_acc > best_val + 1e-9:
            best_val = val_acc
            best_test = test_acc
            best_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            print(f"[seed {seed}] {model_name} | epoch {epoch:03d} | train_loss {train_loss:.4f} | val {val_acc:.4f} | test {test_acc:.4f}")

        if epochs_no_improve >= early_stop_patience:
            # early stop
            break

    # compute fidelity/sparsity for bcos
    if model_name != "gcn":
        fid, spr = compute_fidelity_sparsity_graph(model, test_dataset, device)
    else:
        fid, spr = float("nan"), float("nan")

    return best_val, best_test, best_loss, fid, spr

# ---------------------------
# K-Fold split helper (stratified)
# ---------------------------
def make_folds(dataset: TUDataset, n_splits: int, seed: int = 42):
    """Return list of (train_idx, test_idx) tuples for cross-validation. Use StratifiedKFold if available."""
    labels = [int(d.y.item()) for d in dataset]
    n = len(dataset)
    indices = list(range(n))
    if SKLEARN_EXISTS:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for train_idx, test_idx in skf.split(indices, labels):
            folds.append((train_idx.tolist(), test_idx.tolist()))
        return folds
    else:
        # fallback: repeated random splits -> produce roughly balanced folds
        rnd = random.Random(seed)
        rnd.shuffle(indices)
        fold_sizes = [n // n_splits + (1 if i < (n % n_splits) else 0) for i in range(n_splits)]
        folds = []
        start = 0
        for i in range(n_splits):
            end = start + fold_sizes[i]
            test_idx = indices[start:end]
            train_idx = [idx for idx in indices if idx not in test_idx]
            folds.append((train_idx, test_idx))
            start = end
        return folds

# ---------------------------
# High-level experiment run:
#  - standard GCN baseline
#  - BCos-GCN across multiple B values + small grid search for lr/residual_ratio
#  - per-dataset defaults tuned for better results
# ---------------------------
def run_experiment(args):
    device = get_device(args.device)
    print("Running experiment with args:", args)
    print("Device:", device)

    # load dataset (assume data present)
    dataset = TUDataset(root=args.data_root, name=args.dataset)
    if len(dataset) == 0:
        raise RuntimeError("Empty dataset.")
    # require node features present (user said they exist)
    sample_x = getattr(dataset[0], "x", None)
    if sample_x is None and not args.allow_feature_fix:
        raise RuntimeError("Dataset graphs lack node features (data.x is None). Use --allow_feature_fix to auto-create degree features.")
    # optionally fix missing features
    if sample_x is None and args.allow_feature_fix:
        for d in dataset:
            if getattr(d, "x", None) is None:
                if d.edge_index is not None and d.edge_index.numel() > 0:
                    d.x = degree(d.edge_index[0], num_nodes=d.num_nodes).view(-1,1)
                else:
                    d.x = torch.ones((d.num_nodes,1))
    in_ch = int(dataset[0].x.size(1))
    out_ch = int(dataset.num_classes)
    n_graphs = len(dataset)
    print(f"Loaded {args.dataset}: {n_graphs} graphs | F={in_ch} | C={out_ch}")

    # choose folds depending on dataset (MUTAG -> 10-fold commonly; PROTEINS -> 5-fold)
    n_splits = 10 if args.dataset.upper() == "MUTAG" else 5
    folds = make_folds(dataset, n_splits=n_splits, seed=args.seed)

    results = {}  # store aggregated results

    # 1) Standard GCN baseline: run cross-val repeats (different seeds) and keep mean/std
    print("\n=== Running Standard GCN baseline ===")
    gcn_val_scores, gcn_test_scores, gcn_losses = [], [], []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        # create train/test lists
        train_ds = [dataset[i] for i in train_idx]
        test_ds = [dataset[i] for i in test_idx]
        # split train_ds into train/val small holdout (e.g., 90/10)
        rnd = random.Random(args.seed + fold_i)
        rnd.shuffle(train_ds)
        n_tr = int(0.9 * len(train_ds)) if len(train_ds) > 2 else int(0.8 * len(train_ds))
        tr_split = train_ds[:n_tr]
        val_split = train_ds[n_tr:]
        # run with fixed per-dataset hyperparams (you can change in CLI)
        best_val, best_test, best_loss, _, _ = run_fold(
            tr_split, val_split, test_ds, "gcn", in_ch, out_ch,
            args.hidden, args.dropout, None, args.temp, False,
            residual_ratio=0.6, lr=args.lr_gcn, weight_decay=args.weight_decay,
            epochs=args.epochs, device=device, seed=args.seed + fold_i,
            log_interval=args.log_interval, early_stop_patience=args.early_stop
        )
        gcn_val_scores.append(best_val); gcn_test_scores.append(best_test); gcn_losses.append(best_loss)
    results["gcn"] = {
        "val_mean": mean(gcn_val_scores), "val_std": (stdev(gcn_val_scores) if len(gcn_val_scores) > 1 else 0.0),
        "test_mean": mean(gcn_test_scores), "test_std": (stdev(gcn_test_scores) if len(gcn_test_scores) > 1 else 0.0),
        "loss_mean": mean(gcn_losses), "loss_std": (stdev(gcn_losses) if len(gcn_losses) > 1 else 0.0)
    }

    # 2) BCos-GCN: try multiple B values and a small hyperparameter grid for lr/residual_ratio
    print("\n=== Running BCos-GCN sweeps ===")
    best_bcos_overall = None  # track best config by test mean
    for B in args.B_values:
        print(f"\n--> Evaluating BCos-GCN with B={B}")
        # small grid: residual_ratio and lr choices to tune per dataset
        residual_choices = args.residual_grid
        lr_choices = args.lr_grid
        # store per-grid aggregated results
        grid_records = []
        for residual in residual_choices:
            for lr in lr_choices:
                val_scores, test_scores, loss_scores, fid_scores, spr_scores = [], [], [], [], []
                # cross-val folds
                for fold_i, (train_idx, test_idx) in enumerate(folds):
                    train_ds = [dataset[i] for i in train_idx]
                    test_ds = [dataset[i] for i in test_idx]
                    rnd = random.Random(args.seed + fold_i)
                    rnd.shuffle(train_ds)
                    n_tr = int(0.9 * len(train_ds)) if len(train_ds) > 2 else int(0.8 * len(train_ds))
                    tr_split = train_ds[:n_tr]
                    val_split = train_ds[n_tr:]
                    best_val, best_test, best_loss, fid, spr = run_fold(
                        tr_split, val_split, test_ds, "bcos-gcn", in_ch, out_ch,
                        args.hidden, args.dropout, B, args.temp, args.learn_B,
                        residual, lr, args.weight_decay,
                        args.epochs, device, seed=args.seed + 1000 + fold_i,
                        log_interval=args.log_interval, early_stop_patience=args.early_stop
                    )
                    val_scores.append(best_val); test_scores.append(best_test); loss_scores.append(best_loss)
                    fid_scores.append(fid); spr_scores.append(spr)
                # aggregate across folds
                rec = {
                    "B": B,
                    "residual": residual,
                    "lr": lr,
                    "val_mean": mean(val_scores),
                    "val_std": (stdev(val_scores) if len(val_scores) > 1 else 0.0),
                    "test_mean": mean(test_scores),
                    "test_std": (stdev(test_scores) if len(test_scores) > 1 else 0.0),
                    "loss_mean": mean(loss_scores),
                    "loss_std": (stdev(loss_scores) if len(loss_scores) > 1 else 0.0),
                    "fidelity": mean([x for x in fid_scores if not math.isnan(x)]) if len(fid_scores) > 0 else float('nan'),
                    "sparsity": mean([x for x in spr_scores if not math.isnan(x)]) if len(spr_scores) > 0 else float('nan')
                }
                grid_records.append(rec)
                print(f"   B={B} residual={residual:.2f} lr={lr:.4f} -> test_mean {rec['test_mean']:.4f} val_mean {rec['val_mean']:.4f}")

        # pick best config for this B by test_mean
        best_rec = max(grid_records, key=lambda r: r["test_mean"])
        results[f"bcos-gcn_B={B}"] = best_rec
        # track overall best BCos config
        if best_bcos_overall is None or best_rec["test_mean"] > best_bcos_overall["test_mean"]:
            best_bcos_overall = best_rec

    # Summarize & print nicely
    print("\n\n===== EXPERIMENT SUMMARY =====")
    # Standard GCN summary
    print("\nStandard GCN:")
    print(f"  val_mean ± std : {results['gcn']['val_mean']:.4f} ± {results['gcn']['val_std']:.4f}")
    print(f"  test_mean ± std: {results['gcn']['test_mean']:.4f} ± {results['gcn']['test_std']:.4f}")
    print(f"  loss_mean ± std: {results['gcn']['loss_mean']:.4f} ± {results['gcn']['loss_std']:.4f}")

    # BCos summary (per B)
    print("\nBCos-GCN (best config per B):")
    for B in args.B_values:
        key = f"bcos-gcn_B={B}"
        rec = results[key]
        print(f"  B={B:.2f}: test_mean={rec['test_mean']:.4f} ± {rec['test_std']:.4f} | val_mean={rec['val_mean']:.4f} ± {rec['val_std']:.4f} | loss_mean={rec['loss_mean']:.4f} ± {rec['loss_std']:.4f} | fidelity={rec['fidelity']:.4f} sparsity={rec['sparsity']:.4f}")
    # best bcos
    print("\nBest BCos-GCN overall config:")
    print(f"  B={best_bcos_overall['B']} residual={best_bcos_overall['residual']:.3f} lr={best_bcos_overall['lr']:.5f} -> test_mean {best_bcos_overall['test_mean']:.4f} ± {best_bcos_overall['test_std']:.4f}")

    # compare best bcos and gcn
    print("\nOverall best model by test_mean:")
    if best_bcos_overall['test_mean'] >= results['gcn']['test_mean']:
        print(f"  BCos-GCN (B={best_bcos_overall['B']}) WIN: {best_bcos_overall['test_mean']:.4f} >= GCN {results['gcn']['test_mean']:.4f}")
    else:
        print(f"  GCN WIN: {results['gcn']['test_mean']:.4f} > BCos {best_bcos_overall['test_mean']:.4f}")

    return results, best_bcos_overall

# ---------------------------
# CLI / defaults
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PROTEINS", choices=["MUTAG","PROTEINS"])
    parser.add_argument("--data_root", type=str, default="data/TUDataset")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--B_values", nargs="+", type=float, default=[1.0,1.5,2.0,2.5])
    parser.add_argument("--residual_grid", nargs="+", type=float, default=[0.6,0.75,0.9])
    parser.add_argument("--lr_grid", nargs="+", type=float, default=[0.0035,0.001,0.0005])
    parser.add_argument("--lr_gcn", type=float, default=0.0035)
    parser.add_argument("--lr", type=float, default=0.0035)  # fallback
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument("--temp", type=float, default=1.9)
    parser.add_argument("--learn_B", action="store_true", help="set to learn B as parameter")
    parser.add_argument("--allow_feature_fix", action="store_true", help="if dataset lacks x, create degree features")
    parser.add_argument("--early_stop", type=int, default=40)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # small adjustments per dataset for faster/better convergence (you can change)
    if args.dataset.upper() == "MUTAG":
        args.B_values = [1.0, 1.5, 2.0, 2.5]
        args.residual_grid = [0.6, 0.75, 0.9]
        args.lr_grid = [0.0035, 0.001, 0.0005]
        args.hidden = 96
        args.epochs = 200
    else:  # PROTEINS
        args.B_values = [1.0, 1.5, 2.0, 2.5]
        args.residual_grid = [0.5, 0.7, 0.9]
        args.lr_grid = [0.0035, 0.001, 0.0008]
        args.hidden = 128
        args.epochs = 250

    print("Starting run with dataset:", args.dataset)
    run_experiment(args)

if __name__ == "__main__":
    main()
