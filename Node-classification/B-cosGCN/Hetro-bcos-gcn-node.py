# ===========================================================
# BCos-GCN on Heterophilic Datasets (Texas, Cornell)
# Node Classification task
# Tuned for better accuracy
# ===========================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WebKB
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree

# -----------------------------------------------------------
# Hyperparameters (tuned)
# -----------------------------------------------------------
HYPERPARAMS = {
    'Texas': {'hidden': 32, 'lr': 0.005, 'wd': 5e-4, 'epochs': 500, 'dropout': 0.3},
    'Cornell': {'hidden': 32, 'lr': 0.005, 'wd': 5e-4, 'epochs': 500, 'dropout': 0.3}
}

B_GRID = [1.0,1.1, 1.2,1.3 ,1.5, 1.7, 2.0,2.5,3.0]

# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_adj(edge_index, num_nodes, device):
    edge_index = edge_index.to(device)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    row, col = edge_index
    deg = degree(col, num_nodes=num_nodes).to(device)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, edge_weight

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
class BaselineGCN(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, dropout):
        super().__init__()
        self.c1 = GCNConv(in_ch, hidden, bias=False)
        self.c2 = GCNConv(hidden, out_ch, bias=False)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.c1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.c2(x, edge_index, edge_weight)


class BCosGCN(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, B=2.0, dropout=0.3):
        super().__init__()
        self.lin1 = BcosLinearNoBias(in_ch, hidden, B)
        self.lin2 = BcosLinearNoBias(hidden, out_ch, B)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        row, col = edge_index

        z = torch.zeros_like(x)
        z.index_add_(0, row, x[col] * edge_weight.unsqueeze(-1))

        h = F.relu(self.lin1(z))
        h = F.dropout(h, p=self.dropout, training=self.training)

        z2 = torch.zeros_like(h)
        z2.index_add_(0, row, h[col] * edge_weight.unsqueeze(-1))

        return self.lin2(z2)

# -----------------------------------------------------------
# Training / Evaluation
# -----------------------------------------------------------
def train_epoch(model, data, opt, edge_index, edge_weight, split_id=0):
    model.train()
    opt.zero_grad()

    out = model(data.x, edge_index, edge_weight)
    train_mask = data.train_mask[:, split_id]

    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, edge_index, edge_weight, split_id=0):
    model.eval()
    out = model(data.x, edge_index, edge_weight)
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
    seeds = list(range(10))
    split_ids = [0,1,2]  # WebKB has 3 splits

    for dataset_name in ['Texas', 'Cornell']:
        print(f"\n================ Dataset: {dataset_name} ================")

        cfg = HYPERPARAMS[dataset_name]
        dataset = WebKB(root='./data', name=dataset_name)

        # Normalize node features
        for split_id in split_ids:
            data = dataset[0]
            data.x = F.normalize(data.x, p=2, dim=1)
            data = data.to(device)

            edge_index, edge_weight = normalize_adj(
                data.edge_index, data.num_nodes, device
            )

            in_ch, out_ch = dataset.num_node_features, dataset.num_classes

            # ---------- Baseline GCN ----------
            base_tests, base_losses = [], []

            for s in seeds:
                set_seed(s)
                model = BaselineGCN(in_ch, cfg['hidden'], out_ch, cfg['dropout']).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
                for _ in range(cfg['epochs']):
                    loss = train_epoch(model, data, opt, edge_index, edge_weight, split_id)
                acc = evaluate(model, data, edge_index, edge_weight, split_id)
                base_tests.append(acc['test'])
                base_losses.append(loss)

            print(
                f"[Split {split_id}] Baseline GCN | "
                f"Test: {np.mean(base_tests):.4f} ± {np.std(base_tests):.4f} | "
                f"Loss: {np.mean(base_losses):.4f}"
            )

            # ---------- BCos-GCN ----------
            best_val, best_B, best_test = -1, None, None

            for B in B_GRID:
                vals, tests, losses = [], [], []

                for s in seeds:
                    set_seed(s)
                    model = BCosGCN(in_ch, cfg['hidden'], out_ch, B=B, dropout=cfg['dropout']).to(device)
                    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
                    for _ in range(cfg['epochs']):
                        loss = train_epoch(model, data, opt, edge_index, edge_weight, split_id)
                    acc = evaluate(model, data, edge_index, edge_weight, split_id)
                    vals.append(acc['val'])
                    tests.append(acc['test'])
                    losses.append(loss)

                print(
                    f"[Split {split_id}] BCos-GCN B={B} | "
                    f"Test: {np.mean(tests):.4f} ± {np.std(tests):.4f} | "
                    f"Loss: {np.mean(losses):.4f} | Val: {np.mean(vals):.4f}"
                )

                if np.mean(vals) > best_val:
                    best_val = np.mean(vals)
                    best_B = B
                    best_test = (np.mean(tests), np.std(tests))

            print(
                f"[Split {split_id}] Best BCos-GCN (B={best_B}) | "
                f"Test: {best_test[0]:.4f} ± {best_test[1]:.4f}"
            )


if __name__ == '__main__':
    run()
