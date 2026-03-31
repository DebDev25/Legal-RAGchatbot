import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read metrics
df = pd.read_csv("benchmark_metrics.csv")

# Simplify questions for x-axis labels
short_questions = [f"Q{i+1}" for i in range(len(df))]

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Generation Time
sns.barplot(x=short_questions, y=df['Generation_Time_sec'], ax=axs[0], palette="Blues_d")
axs[0].set_title("NVIDIA CUDA Generation Latency (Seconds)", fontsize=14)
axs[0].set_ylabel("Time (s)", fontsize=12)
for i, v in enumerate(df['Generation_Time_sec']):
    axs[0].text(i, v + 1, f"{v:.1f}s", ha='center', fontweight='bold')

# Plot 2: Relevancy Distance
sns.barplot(x=short_questions, y=df['Avg_L2_Vector_Distance'], ax=axs[1], palette="Reds_d")
axs[1].set_title("ChromaDB Vector Distance (Lower = More Relevant)", fontsize=14)
axs[1].set_ylabel("L2 Distance Score", fontsize=12)
for i, v in enumerate(df['Avg_L2_Vector_Distance']):
    axs[1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("rag_performance_metrics.png", dpi=300)
print("Graph saved as rag_performance_metrics.png!")
