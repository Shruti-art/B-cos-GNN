# ===========================================================
# BCos-GraphSAGE on Heterophilic Datasets (Texas, Cornell)
# Node Classification task
# ===========================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import WebKB
from torch_geometric.nn import SAGEConv

# -----------------------------------------------------------
# Hyperparameters (dataset-specific)
# -----------------------------------------------------------
HYPERPARAMS = {
    'Texas':   {'hidden': 64, 'lr': 0.005, 'wd': 5e-4, 'epochs': 500, 'dropout': 0.3},
    'Cornell': {'hidden': 64, 'lr': 0.005, 'wd': 5e-4, 'epochs': 500, 'dropout': 0.3},
}

B_GRID = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0,2.5,3.0]

SEEDS = list(range(10))
SPLITS = [0, 1, 2]   # WebKB provides 3 official splits

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
# Models
# -----------------------------------------------------------
class BaselineGraphSAGE(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, dropout):
        super().__init__()
        self.s1 = SAGEConv(in_ch, hidden)
        self.s2 = SAGEConv(hidden, out_ch)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.s1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.s2(x, edge_index)


class BCosGraphSAGE(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, B=2.0, dropout=0.3):
        super().__init__()

        # Neighbor message projections (BCos)
        self.neigh_lin1 = BcosLinearNoBias(in_ch, hidden, B)
        self.neigh_lin2 = BcosLinearNoBias(hidden, out_ch, B)

        # Self feature projections (standard linear)
        self.self_lin1 = nn.Linear(in_ch, hidden, bias=False)
        self.self_lin2 = nn.Linear(hidden, out_ch, bias=False)

        self.dropout = dropout

    def forward(self, x, edge_index):
        row, col = edge_index
        N = x.size(0)

        # ===== Layer 1 =====
        neigh_msg = self.neigh_lin1(x[col])           # BCos on messages
        agg = torch.zeros(N, neigh_msg.size(1), device=x.device)
        agg.index_add_(0, row, neigh_msg)

        deg = torch.zeros(N, device=x.device)
        deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float))
        agg = agg / deg.clamp(min=1).unsqueeze(1)

        h = agg + self.self_lin1(x)                    # preserve self-info
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ===== Layer 2 =====
        neigh_msg2 = self.neigh_lin2(h[col])
        agg2 = torch.zeros(N, neigh_msg2.size(1), device=x.device)
        agg2.index_add_(0, row, neigh_msg2)

        deg2 = torch.zeros(N, device=x.device)
        deg2.index_add_(0, row, torch.ones_like(row, dtype=torch.float))
        agg2 = agg2 / deg2.clamp(min=1).unsqueeze(1)

        out = agg2 + self.self_lin2(h)
        return out


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

        data.x = F.normalize(data.x, p=2, dim=1)
        data = data.to(device)

        in_ch = dataset.num_node_features
        out_ch = dataset.num_classes

        for split_id in SPLITS:

            # ---------- Baseline GraphSAGE ----------
            base_tests, base_losses = [], []

            for seed in SEEDS:
                set_seed(seed)
                model = BaselineGraphSAGE(
                    in_ch, cfg['hidden'], out_ch, cfg['dropout']
                ).to(device)
                opt = torch.optim.Adam(
                    model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
                )

                for _ in range(cfg['epochs']):
                    loss = train_epoch(model, data, opt, split_id)

                acc = evaluate(model, data, split_id)
                base_tests.append(acc['test'])
                base_losses.append(loss)

            print(
                f"[Split {split_id}] Baseline GraphSAGE | "
                f"Test: {np.mean(base_tests):.4f} ± {np.std(base_tests):.4f} | "
                f"Loss: {np.mean(base_losses):.4f}"
            )

            # ---------- BCos-GraphSAGE ----------
            best_val, best_B, best_test = -1, None, None

            for B in B_GRID:
                vals, tests, losses = [], [], []

                for seed in SEEDS:
                    set_seed(seed)
                    model = BCosGraphSAGE(
                        in_ch, cfg['hidden'], out_ch, B=B, dropout=cfg['dropout']
                    ).to(device)
                    opt = torch.optim.Adam(
                        model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
                    )

                    for _ in range(cfg['epochs']):
                        loss = train_epoch(model, data, opt, split_id)

                    acc = evaluate(model, data, split_id)
                    vals.append(acc['val'])
                    tests.append(acc['test'])
                    losses.append(loss)

                print(
                    f"[Split {split_id}] BCos-GraphSAGE B={B} | "
                    f"Test: {np.mean(tests):.4f} ± {np.std(tests):.4f} | "
                    f"Loss: {np.mean(losses):.4f} | Val: {np.mean(vals):.4f}"
                )

                if np.mean(vals) > best_val:
                    best_val = np.mean(vals)
                    best_B = B
                    best_test = (np.mean(tests), np.std(tests))

            print(
                f"[Split {split_id}] Best BCos-GraphSAGE (B={best_B}) | "
                f"Test: {best_test[0]:.4f} ± {best_test[1]:.4f}"
            )


if __name__ == '__main__':
    run()
