
import matplotlib.pyplot as plt

# Sparsity values (EXACT)
sparsity = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Fidelity values for different B (EXACT)
fidelity_B_1_0 = [1.00, 0.95, 0.92, 0.885, 0.85, 0.82, 0.77, 0.73, 0.685, 0.61]
fidelity_B_1_1 = [1.00, 0.955, 0.925, 0.89, 0.86, 0.83, 0.805, 0.77, 0.73, 0.63]
fidelity_B_1_2 = [1.00, 0.95, 0.92, 0.90, 0.87, 0.84, 0.82, 0.79, 0.745, 0.645]
fidelity_B_1_3 = [1.00, 0.945, 0.915, 0.88, 0.85, 0.83, 0.80, 0.77, 0.72, 0.64]
fidelity_B_1_5 = [1.00, 0.945, 0.92, 0.895, 0.87, 0.83, 0.795, 0.74, 0.665, 0.565]
fidelity_B_1_7 = [1.00, 0.93, 0.885, 0.855, 0.825, 0.795, 0.78, 0.75, 0.705, 0.66]
fidelity_B_2_0 = [1.00, 0.95, 0.92, 0.895, 0.87, 0.845, 0.82, 0.79, 0.765, 0.685]

# Increase ONLY font sizes (paper-ready)
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18
})

plt.figure(figsize=(10, 7))

plt.plot(sparsity, fidelity_B_1_0, marker='o', linewidth=2.5, label="B=1.0")
plt.plot(sparsity, fidelity_B_1_1, marker='o', linewidth=2.5, label="B=1.1")
plt.plot(sparsity, fidelity_B_1_2, marker='o', linewidth=2.5, label="B=1.2")
plt.plot(sparsity, fidelity_B_1_3, marker='o', linewidth=2.5, label="B=1.3")
plt.plot(sparsity, fidelity_B_1_5, marker='o', linewidth=2.5, label="B=1.5")
plt.plot(sparsity, fidelity_B_1_7, marker='o', linewidth=2.5, label="B=1.7")
plt.plot(sparsity, fidelity_B_2_0, marker='o', linewidth=2.5, label="B=2.0")

plt.xlabel("Sparsity", fontsize=24)
plt.ylabel("Fidelity", fontsize=24)
plt.title("Cora: Fidelity vs Sparsity (Multi-B)", fontsize=24)

plt.legend(loc="lower left")
plt.grid(False)
plt.tight_layout()
# 🔹 SAVE FIGURE (paper-ready)
plt.tight_layout()
plt.savefig(
    "Cora_Fidelity_vs_Sparsity_MultiB.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
