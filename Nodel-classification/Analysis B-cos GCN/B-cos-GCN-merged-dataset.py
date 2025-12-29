import matplotlib.pyplot as plt

# Data (EXACTLY same as figure)
sparsity = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

cora =     [1.00, 0.95, 0.915, 0.89, 0.87, 0.845, 0.825, 0.80, 0.75, 0.69]
citeseer = [1.00, 0.94, 0.895, 0.85, 0.815, 0.76, 0.70, 0.645, 0.59, 0.505]
pubmed =   [1.00, 0.96, 0.935, 0.92, 0.905, 0.89, 0.875, 0.86, 0.835, 0.79]

# Increase global font size 
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
})

plt.figure(figsize=(10, 7))

plt.plot(sparsity, cora, marker='o', linewidth=2.5, markersize=8,
         label='Cora (B=1.7)')
plt.plot(sparsity, citeseer, marker='o', linewidth=2.5, markersize=8,
         label='CiteSeer (B=1.3)')
plt.plot(sparsity, pubmed, marker='o', linewidth=2.5, markersize=8,
         label='PubMed (B=2.0)')

plt.xlabel("Sparsity", fontsize=24)
plt.ylabel("Fidelity", fontsize=24)
plt.title("BCos-GCN Fidelity vs Sparsity (Optimal B)")

plt.legend()
plt.grid(False)
plt.tight_layout()

plt.tight_layout()
plt.savefig(
    "BCos_GCN_Fidelity_vs_Sparsity_OptimalB.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

