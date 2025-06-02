import pandas as pd

# Load all results
df = pd.read_csv("./results/results.csv")

# Group by number of workers
summary = df.groupby("num_workers").agg({
    "ARI": ["mean", "std"],
    "NMI": ["mean", "std"],
    "Silhouette": ["mean", "std"],
    "rounds": ["mean", "std"],
    "time_taken": ["mean", "std"],
    "comm_volume": ["mean", "std"]
})

# Flatten the multi-level columns
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

# Format for LaTeX
def format_latex(mean, std):
    return f"{mean:.3f} ± {std:.3f} &"

# Create a new DataFrame with LaTeX formatted strings
latex_ready = pd.DataFrame(index=summary.index)

for metric in ["ARI", "NMI", "Silhouette", "rounds", "time_taken", "comm_volume"]:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    latex_ready[metric] = summary.apply(lambda row: format_latex(row[mean_col], row[std_col]), axis=1)

# Print LaTeX-ready DataFrame
print(latex_ready)
