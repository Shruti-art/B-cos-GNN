# ============================================================
# BCos-GCN Node Classification – Final Script (Revised)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Seed
# -----------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Dataset hyperparameters (CiteSeer RETUNED)
# -----------------------------
DATASET_CONFIG = {
    "Cora": dict(hidden=64, lr=0.01, wd=5e-4, epochs=200),
    "CiteSeer": dict(hidden=64, lr=0.01, wd=5e-4, epochs=500),  # 🔧 RETUNED
    "PubMed": dict(hidden=128, lr=0.005, wd=1e-3, epochs=300)
}

B_VALUES = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]
SEEDS = [0, 1, 2, 3, 4]

# -----------------------------
# Font settings
# -----------------------------
PLOT_FONT = 16
plt.rcParams.update({'font.size': PLOT_FONT})

# -----------------------------
# Folder for plots
# -----------------------------
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------
# Models
# -----------------------------
class BCosGCNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, B):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch))
        self.B = B
    def forward(self, z):
        lin = z @ self.weight.t()
        cos = (F.normalize(z, dim=1) @ F.normalize(self.weight, dim=1).t()).clamp(1e-6)
        scale = cos.abs().pow(self.B - 1.0)
        return lin * scale, scale

class BCosGCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, B):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, bias=False)
        self.conv2 = GCNConv(hidden, hidden, bias=False)
        self.bcos1 = BCosGCNLayer(hidden, hidden, B)
        self.bcos2 = BCosGCNLayer(hidden, out_dim, B)
    def forward(self, x, edge_index):
        z1 = self.conv1(x, edge_index)
        h1, s1 = self.bcos1(z1)
        h1 = F.relu(h1)
        z2 = self.conv2(h1, edge_index)
        out, s2 = self.bcos2(z2)
        return out, {"z": (z1.detach(), z2.detach()), "scale": (s1.detach(), s2.detach())}

# -----------------------------
# Dataset loader
# -----------------------------
def load_dataset(name):
    dataset = Planetoid("./data", name)
    return dataset, dataset[0].to(device)

# -----------------------------
# Training & evaluation
# -----------------------------
def train_eval(model, data, cfg):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_val, best_state = 0, None
    for _ in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)[0]
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = (out.argmax(1)[data.val_mask] == data.y[data.val_mask]).float().mean()
        if val > best_val:
            best_val = val
            best_state = model.state_dict()
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)[0]
        acc = (out.argmax(1)[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask]).item()
    return acc, loss

# -----------------------------
# Node importance
# -----------------------------
@torch.no_grad()
def node_importance(model, data):
    logits, info = model(data.x, data.edge_index)
    z2, s2 = info["z"][1], info["scale"][1]
    W = model.bcos2.weight
    imp = []
    for i in range(data.num_nodes):
        c = logits[i].argmax()
        imp.append((z2[i] * W[c] * s2[i, c]).abs().sum().item())
    return np.array(imp)

# -----------------------------
# Fidelity vs Sparsity
# -----------------------------
def fidelity_curve(model, data, imp):
    base = model(data.x, data.edge_index)[0].argmax(1)
    xs, ys = [], []
    fracs = np.linspace(0.1, 1.0, 10)
    for f in fracs:
        x = data.x.clone()
        k = int(len(imp) * f)
        idx = np.argsort(-imp)[:k]
        x[:] = 0
        x[idx] = data.x[idx]
        xs.append(1 - f)
        pred = model(x, data.edge_index)[0].argmax(1)
        ys.append((pred == base).float().mean().item())
    return xs, ys

# -----------------------------
# NEW Contribution Map (Top nodes × Feature dim)
# -----------------------------
def plot_contribution_map(model, data, name, top_k=20):
    logits, info = model(data.x, data.edge_index)
    z2, s2 = info["z"][1], info["scale"][1]
    W = model.bcos2.weight

    scores = node_importance(model, data)
    top_idx = np.argsort(-scores)[:top_k]

    contrib = []
    for i in top_idx:
        c = logits[i].argmax()
        contrib.append((z2[i] * W[c] * s2[i, c]).abs().detach().cpu().numpy())

    contrib = np.stack(contrib)

    plt.figure(figsize=(10, 6))
    plt.imshow(contrib, aspect="auto", cmap="hot")
    plt.colorbar(label="Contribution magnitude")
    plt.xlabel("Feature dimension", fontsize=PLOT_FONT)
    plt.ylabel("Top contributing nodes", fontsize=PLOT_FONT)
    plt.title(f"BCos-GCN Contribution Map ({name}, Optimal B)", fontsize=PLOT_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"BCosGCN_ContributionMap_{name}.png"), dpi=300)
    plt.close()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    FINAL_RESULTS, BEST_MODELS = {}, {}

    # ---- Train and select best B ----
    for name, cfg in DATASET_CONFIG.items():
        dataset, data = load_dataset(name)
        best_acc = 0
        for B in B_VALUES:
            accs, losses = [], []
            for s in SEEDS:
                set_seed(s)
                model = BCosGCN(dataset.num_features, cfg["hidden"], dataset.num_classes, B).to(device)
                acc, loss = train_eval(model, data, cfg)
                accs.append(acc)
                losses.append(loss)
            if np.mean(accs) > best_acc:
                best_acc = np.mean(accs)
                BEST_MODELS[name] = model
                FINAL_RESULTS[name] = {
                    "B": B,
                    "acc_mean": np.mean(accs),
                    "acc_std": np.std(accs),
                    "loss": np.mean(losses)
                }

    # ---- Multi-B curves ----
    for name, cfg in DATASET_CONFIG.items():
        dataset, data = load_dataset(name)
        plt.figure(figsize=(8,6))
        for B in B_VALUES:
            model = BCosGCN(dataset.num_features, cfg["hidden"], dataset.num_classes, B).to(device)
            train_eval(model, data, cfg)
            imp = node_importance(model, data)
            xs, ys = fidelity_curve(model, data, imp)
            plt.plot(xs, ys, marker="o", label=f"B={B}")
        plt.xlabel("Sparsity", fontsize=PLOT_FONT)
        plt.ylabel("Fidelity", fontsize=PLOT_FONT)
        plt.title(f"{name}: Fidelity vs Sparsity (Multi-B)", fontsize=PLOT_FONT)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"BCosGCN_Fidelity_MultiB_{name}.png"), dpi=300)
        plt.close()

    # ---- Merged optimal B curve ----
    plt.figure(figsize=(8,6))
    for name, model in BEST_MODELS.items():
        _, data = load_dataset(name)
        imp = node_importance(model, data)
        xs, ys = fidelity_curve(model, data, imp)
        plt.plot(xs, ys, marker="o", label=f"{name} (B={FINAL_RESULTS[name]['B']})")
    plt.xlabel("Sparsity", fontsize=PLOT_FONT)
    plt.ylabel("Fidelity", fontsize=PLOT_FONT)
    plt.title("BCos-GCN Fidelity vs Sparsity (Optimal B)", fontsize=PLOT_FONT)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "BCosGCN_Fidelity_Merged_OptimalB.png"), dpi=300)
    plt.close()

    # ---- Contribution maps ----
    for name, model in BEST_MODELS.items():
        _, data = load_dataset(name)
        plot_contribution_map(model, data, name)

    print("\nAll plots saved in:", PLOT_DIR)
