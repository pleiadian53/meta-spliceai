import matplotlib.pyplot as plt
import pandas as pd


def plot_splice_predictions(predicted_donors, predicted_acceptors, exons, gene_length, gene_name):
    """
    Plot predicted donor and acceptor sites along with actual exon positions for a gene.

    Parameters:
    - predicted_donors (pd.DataFrame): DataFrame with 'position' and 'probability' columns for donor sites.
    - predicted_acceptors (pd.DataFrame): DataFrame with 'position' and 'probability' columns for acceptor sites.
    - exons (pd.DataFrame): DataFrame with 'start' and 'end' columns for exon positions.
    - gene_length (int): Length of the gene sequence.
    - gene_name (str): Name of the gene.
    """
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # Plot gene track
    ax.plot([0, gene_length], [0, 0], color='black', lw=2, label=f"{gene_name} Gene Track")

    # Plot predicted donor sites (green arrows)
    for _, row in predicted_donors.iterrows():
        ax.arrow(row['position'], 0, 0, 0.2, color='green', head_width=0.05 * gene_length, head_length=0.02, alpha=row['probability'], length_includes_head=True)

    # Plot predicted acceptor sites (red arrows)
    for _, row in predicted_acceptors.iterrows():
        ax.arrow(row['position'], 0, 0, -0.2, color='red', head_width=0.05 * gene_length, head_length=0.02, alpha=row['probability'], length_includes_head=True)

    # Plot exons (black boxes)
    for _, row in exons.iterrows():
        ax.add_patch(plt.Rectangle((row['start'], -0.1), row['end'] - row['start'], 0.2, color='black'))

    # Set axis labels and limits
    ax.set_xlim(0, gene_length)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Genomic Position')
    ax.set_ylabel('Predictions')
    ax.set_title(f"Predicted Donor and Acceptor Sites for {gene_name}")
    
    # Add legend
    ax.legend()
    
    # Display plot
    plt.show()

# Example Usage
# Example data for predicted donors, acceptors, and exons
predicted_donors = pd.DataFrame({'position': [1000, 5000, 15000], 'probability': [0.8, 0.9, 0.7]})
predicted_acceptors = pd.DataFrame({'position': [2000, 7000, 17000], 'probability': [0.85, 0.95, 0.75]})
exons = pd.DataFrame({'start': [1500, 8000], 'end': [3000, 10000]})
gene_length = 20000  # Length of the gene sequence
gene_name = "CFTR"

# Call the function to plot
plot_splice_predictions(predicted_donors, predicted_acceptors, exons, gene_length, gene_name)
