import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data with proper data types
df = pd.read_csv("data/ensembl/protein_coding_splice_sites.tsv", 
               sep="\t",
               dtype={"gene_id": str},
               low_memory=False)

# Count splice sites per gene
splice_counts = df.groupby("gene_id").size()

# Create figure
plt.figure(figsize=(12, 6))

# Set style
sns.set_style("whitegrid")

# Create histogram with log scale
plt.hist(splice_counts.values, 
        bins=np.logspace(np.log10(2), np.log10(splice_counts.max()), 50),
        color="#4169E1", 
        alpha=0.7)

# Set log scale for x-axis
plt.xscale("log")

# Customize plot
plt.title("Distribution of Splice Sites per Gene (Log Scale)", pad=20)
plt.xlabel("Number of Splice Sites (log scale)")
plt.ylabel("Number of Genes")

# Add text with detailed statistics
stats_text = f"Total Genes: {len(splice_counts):,}\n"
stats_text += f"Mean: {splice_counts.mean():.1f}\n"
stats_text += f"Median: {splice_counts.median():.1f}\n"
stats_text += f"Max: {splice_counts.max():,}\n"
stats_text += f"Top gene: {splice_counts.idxmax()}"

plt.text(0.95, 0.95, stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Save plot
plt.tight_layout()
plt.savefig("output/figures/splice_sites_histogram_log.png", dpi=300, bbox_inches="tight")

# Create a second plot with linear scale for comparison
plt.figure(figsize=(12, 6))
plt.hist(splice_counts.values, bins=50, color="#4169E1", alpha=0.7)
plt.title("Distribution of Splice Sites per Gene (Linear Scale)", pad=20)
plt.xlabel("Number of Splice Sites")
plt.ylabel("Number of Genes")
plt.text(0.95, 0.95, stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.tight_layout()
plt.savefig("output/figures/splice_sites_histogram_linear.png", dpi=300, bbox_inches="tight")
