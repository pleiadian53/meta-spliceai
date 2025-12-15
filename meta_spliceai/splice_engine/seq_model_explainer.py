import os
import polars as pl
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from captum.attr import IntegratedGradients

import matplotlib.pyplot as plt

from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display, 
    display_dataframe_in_chunks
)

# Captum for Local Attributions


# Define a toy sequence classification model (CNN-based)
class DNASeqClassifier(nn.Module):
    def __init__(self):
        super(DNASeqClassifier, self).__init__()
        self.conv = nn.Conv1d(4, 16, kernel_size=5)  # Input: one-hot encoded (A, T, G, C)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(16 * 48, 2)  # Example assumes input sequence length of 100
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def demo_integrated_gradients():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import shap

    # Generate a toy one-hot encoded DNA sequence (length=100)
    sequence = np.random.randint(0, 4, 100)  # Random DNA sequence
    one_hot_sequence = np.eye(4)[sequence].T  # One-hot encoding (shape: [4, 100])

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(one_hot_sequence, dtype=torch.float32).unsqueeze(0)

    # Model
    model = DNASeqClassifier()
    model.eval()  # Use eval mode for Captum

    # Todo: Classification model

    # Captum Integrated Gradients
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(input_tensor, target=0, return_convergence_delta=True)

    # Get attributions as numpy array
    attributions_np = attributions.squeeze().detach().numpy()

    print_emphasized("Extract High-Attribution Regions")

    # Define threshold to select high-attribution positions
    threshold = np.percentile(attributions_np.sum(axis=0), 90)  # Top 10% of attributions

    # Identify positions and extract motifs
    high_attr_positions = np.where(attributions_np.sum(axis=0) >= threshold)[0]

    # Extract motifs (e.g., k-mers of length 6) from the sequence
    motif_length = 6
    motifs = [sequence[i:i+motif_length] for i in high_attr_positions if i+motif_length <= len(sequence)]
    print("Extracted motifs:", motifs)

    print_emphasized("SHAP/InterpretML for Global Motif Analysis")

    # Generate motif presence/absence as features across a dataset
    n_samples = 100
    motif_features = np.random.randint(0, 2, (n_samples, len(motifs)))  # Binary presence/absence
    labels = np.random.randint(0, 2, n_samples)  # Binary classification labels

    # Train RandomForest
    X_train, X_test, y_train, y_test = train_test_split(motif_features, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # SHAP explanation
    explainer = shap.Explainer(clf, X_train)
    shap_values = explainer(X_test)

    # Plot SHAP values for global importance of motifs
    shap.summary_plot(shap_values, X_test, feature_names=[f"Motif_{i}" for i in range(len(motifs))])

    print_emphasized("Combine Results for Interpretation")

    # Identify globally important motifs
    global_importance = np.abs(shap_values.values).mean(axis=0)
    important_motifs_indices = np.argsort(global_importance)[-5:]  # Top 5 motifs

    # Map back to high-attribution positions
    for idx in important_motifs_indices:
        motif = motifs[idx]
        print(f"Motif: {motif}, Locations: {np.where(np.isin(sequence, motif))[0]}")

    return attributions_np, delta


# Starting with pre-computed motifs (k-mers identified using SHAP or another method) and 
# using Captum to validate their alignment with regions of high attribution in the sequences. 
def demo_integrated_gradients_from_precomputed_motifs(): 
    import torch
    from captum.attr import IntegratedGradients

    print_emphasized("Setup and Pre-Computed k-mers")

    # Example pre-computed top k-mers identified from SHAP
    top_kmers = {"AA", "AT", "CG", "GCG", "TAT"}  # Example k-mers (k=2,3)

    # Toy DNA sequences (one-hot encoded for Captum input)
    sequences = [
        "AATCGGCGTATA",  # Sequence 1
        "CGATGCGTAAAT",  # Sequence 2
    ]  # Add more sequences as needed
    one_hot_sequences = []  # Convert sequences to one-hot encoding
    for seq in sequences:
        one_hot = np.zeros((4, len(seq)))  # Shape: [4 (A, T, G, C), sequence_length]
        for i, base in enumerate(seq):
            if base == "A":
                one_hot[0, i] = 1
            elif base == "T":
                one_hot[1, i] = 1
            elif base == "G":
                one_hot[2, i] = 1
            elif base == "C":
                one_hot[3, i] = 1
        one_hot_sequences.append(one_hot)

    # Convert to PyTorch tensors for Captum
    input_tensors = [torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0) for one_hot in one_hot_sequences]

    print_emphasized("Captum Attribution Analysis")

    # Define or load your trained sequence model (example toy model used here)
    class DNASeqClassifier(nn.Module):
        def __init__(self):
            super(DNASeqClassifier, self).__init__()
            self.conv = nn.Conv1d(4, 16, kernel_size=5)  # Input: one-hot encoded (A, T, G, C)
            self.pool = nn.MaxPool1d(2)
            self.fc = nn.Linear(16 * 4, 2)  # Example assumes input sequence length of 12
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = DNASeqClassifier()
    model.eval()  # Captum requires model in evaluation mode

    # Initialize Captum Integrated Gradients
    ig = IntegratedGradients(model)

    # Compute attribution for each sequence
    attribution_results = []
    for input_tensor in input_tensors:
        attributions, _ = ig.attribute(input_tensor, target=0, return_convergence_delta=True)
        attribution_results.append(attributions.squeeze().detach().numpy())

    print_emphasized("Match k-mers with High-Attribution Regions")

    # Threshold for high-attribution regions (e.g., top 10%)
    threshold = 0.9

    for i, (seq, attributions) in enumerate(zip(sequences, attribution_results)):
        print(f"\nSequence {i+1}: {seq}")
        summed_attributions = attributions.sum(axis=0)  # Sum over channels (A, T, G, C)
        high_attr_positions = np.where(summed_attributions >= np.percentile(summed_attributions, threshold * 100))[0]

        print(f"High-attribution positions: {high_attr_positions}")
        
        # Scan for pre-computed k-mers in high-attribution regions
        matches = []
        for pos in high_attr_positions:
            for k in range(2, 5):  # Check k-mers of length 2, 3, 4
                if pos + k <= len(seq):
                    kmer = seq[pos:pos+k]
                    if kmer in top_kmers:
                        matches.append((kmer, pos))
        
        print(f"Matched k-mers in high-attribution regions: {matches}")

    print_emphasized("Quantify and Visualize Overlap")

    # Compute the overlap fraction for validation
    total_high_attr_bases = sum(len(np.where(attr.sum(axis=0) >= np.percentile(attr.sum(axis=0), threshold * 100))[0])
                                for attr in attribution_results)
    matched_bases = sum(len(matches) for matches in attribution_results)  # Matches from the previous step

    overlap_fraction = matched_bases / total_high_attr_bases
    print(f"Fraction of high-attribution regions explained by k-mers: {overlap_fraction:.2f}")

    # Example visualization of attribution scores and k-mer matches
    import matplotlib.pyplot as plt
    for i, (seq, attributions) in enumerate(zip(sequences, attribution_results)):
        summed_attributions = attributions.sum(axis=0)
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(seq)), summed_attributions, label="Attribution Scores")
        plt.scatter([match[1] for match in matches], [summed_attributions[match[1]] for match in matches],
                    color='red', label="Matched k-mer positions")
        plt.title(f"Sequence {i+1}")
        plt.xlabel("Position")
        plt.ylabel("Attribution")
        plt.legend()
        plt.show()


def extract_analysis_sequences_with_probabilities_v0(
    sequence_df: pl.DataFrame,
    position_df: pl.DataFrame,
    predictions_df: pl.DataFrame,
    window_size: int = 500,
    include_empty_entries: bool = True,
    use_absolute_position: bool = True,
    **kargs
) -> pl.DataFrame:
    """
    Extract sequences and probability vectors for predicted splice sites, matching 
    the absolute coordinates from position_df with 'absolute_position' in predictions_df.

    This version fixes the mismatch: 'position' in predict_splice_sites_for_genes(...) 
    is local index, while 'absolute_position' is the real coordinate. So we filter 
    on predictions_df["absolute_position"].

    Parameters
    ----------
    sequence_df : pl.DataFrame
        Must have columns: ['gene_id', 'sequence', 'strand'].
        Negative‐strand genes' sequences are already reversed-complemented if needed.
    position_df : pl.DataFrame
        Must have at least ['gene_id', 'position'] plus any other metadata 
        columns like 'pred_type', 'score', 'splice_type', etc.
        -> Here 'position' is presumably an absolute or near-absolute coordinate 
           from Ensembl, or at least it matches the coordinate system we want 
           for the final 'window'.
    predictions_df : pl.DataFrame
        The tall "one row per base" predictions table from predict_splice_sites_for_genes(...),
        which must include ['gene_id','absolute_position','donor_prob','acceptor_prob','neither_prob'].
        The 'position' column is a local index, but we rely on 'absolute_position'.
    window_size : int
        Number of nucleotides to include on each side of the focal splice site position.
    include_empty_entries : bool
        If True, create a row even if the gene_id is missing from sequence_df 
        or if no probabilities are available in predictions_df (in which case 
        we store None or empty lists).
    use_absolute_position : bool
        If True, we interpret 'position' in position_df as an absolute coordinate
        and match it with predictions_df['absolute_position'].
    **kargs : dict
        Additional parameters (like verbose).

    Returns
    -------
    pl.DataFrame
        One row per row in position_df. In each row:
          - All columns from position_df
          - 'window_start' (0-based index within the gene's sequence)
          - 'window_end'
          - 'sequence' (the substring from window_start:window_end)
          - 'positions_list' (list of int, local positions for each base in the window if local coords)
          - 'donor_prob_list' (list of float)
          - 'acceptor_prob_list' (list of float)
          - 'neither_prob_list' (list of float)

    Example
    -------
        output_df = extract_analysis_sequences_with_probabilities(
            sequence_df, position_df, predictions_df, window_size=500
        )
    """
    import polars as pl
    from collections import defaultdict

    verbose = kargs.get("verbose", 1)

    # Build quick lookups for gene sequences and strand
    gene_seq_lookup = dict(zip(sequence_df["gene_id"], sequence_df["sequence"]))
    strand_lookup = dict(zip(sequence_df["gene_id"], sequence_df["strand"]))

    # We require predictions_df to have 'absolute_position'
    required_cols_pred = {"gene_id", "absolute_position", "donor_prob", "acceptor_prob", "neither_prob"}
    missing_pred = required_cols_pred - set(predictions_df.columns)
    if missing_pred:
        raise ValueError(f"predictions_df must contain columns {missing_pred} to match absolute coords.")

    # Also require position_df to have 'gene_id' and 'position'
    required_cols_pos = {"gene_id", "position"}
    missing_pos = required_cols_pos - set(position_df.columns)
    if missing_pos:
        raise ValueError(f"position_df must contain columns {missing_pos}.")

    # Build a dictionary for predictions keyed by gene_id
    # We'll keep them sorted by 'absolute_position' ascending so we can do range filtering.
    gene_pred_dict = {}
    for gid, sub in predictions_df.group_by("gene_id"):
        sub_sorted = sub.sort("absolute_position", descending=False)
        gene_pred_dict[gid] = sub_sorted

    # We'll store final results row by row
    results = []

    # For each row in position_df
    for row in position_df.iter_rows(named=True):
        gene_id = row["gene_id"]
        pos = row["position"]  # This is presumably an absolute coordinate 
                               # in the same reference as predictions_df["absolute_position"].

        # If gene not in sequence_df, skip or fill empty
        if gene_id not in gene_seq_lookup:
            if include_empty_entries:
                r = row.copy()
                r.update({
                    "window_start": None,
                    "window_end": None,
                    "sequence": None,
                    "positions_list": [],
                    "donor_prob_list": [],
                    "acceptor_prob_list": [],
                    "neither_prob_list": []
                })
                results.append(r)
            continue

        full_seq = gene_seq_lookup[gene_id]
        strand = strand_lookup.get(gene_id, "+")
        seq_len = len(full_seq)

        # We'll define a local window in 0-based indexing w.r.t. the gene's sequence
        # But we only have absolute coords. 
        # We need to convert absolute coords => local coords 
        #   if we want "window_start" etc. 
        # Because negative-strand was reversed-complemented, we must do:
        #   if strand == '+':
        #       local_pos = pos - gene_start
        #   else: 
        #       local_pos = gene_end - pos
        # But let's see if your data has 'gene_start'/'gene_end' in sequence_df 
        # to compute local. 
        # We'll assume we do. 
        # If not, we can't define local positions easily.

        # We also do the absolute window for filtering predictions
        gene_row = sequence_df.filter(pl.col("gene_id") == gene_id)
        if gene_row.is_empty():
            # no info for gene start/end
            if include_empty_entries:
                r = row.copy()
                r.update({
                    "window_start": None,
                    "window_end": None,
                    "sequence": None,
                    "positions_list": [],
                    "donor_prob_list": [],
                    "acceptor_prob_list": [],
                    "neither_prob_list": []
                })
                results.append(r)
            continue

        # Let's assume there's only 1 row in sequence_df per gene
        gene_start = gene_row.select("start").item()
        gene_end = gene_row.select("end").item()

        if strand == '+':
            local_pos = pos - gene_start
        else:
            local_pos = gene_end - pos

        # define local window in [0..seq_len)
        window_start = max(local_pos - window_size, 0)
        window_end = min(local_pos + window_size, seq_len)

        # Extract the substring
        extracted_seq = full_seq[window_start:window_end]

        # Build absolute window in the reference
        if strand == '+':
            abs_window_start = gene_start + window_start
            abs_window_end = gene_start + window_end
        else:
            # if strand == '-'
            # local index 0 => gene_end (the "start" in local sense)
            # local index seq_len-1 => gene_start
            # so absolute coordinates run from [gene_end-window_start.. gene_end-window_end]
            # We can define 
            abs_window_start = gene_start + (seq_len - window_end) if local_pos > 0 else (gene_end - window_end + 1)
            abs_window_end = gene_start + (seq_len - window_start) if local_pos > 0 else (gene_end - window_start + 1)

            # Actually simpler approach is just:
            #   if local_pos = (gene_end - pos),
            #   then absolute coord for local=window_start => ???

            # Maybe let's do a simpler approach: 
            # For filtering predictions, we can define min/max for the absolute coords 
            # if we know the absolute pos for local = window_start => 
            #   => pos_abs = gene_start + window_start if + strand
            #   => pos_abs = gene_end - window_start if - strand

            abs_window_start = gene_end - window_start
            abs_window_end = gene_end - window_end

            # If we want abs_window_start < abs_window_end, we can reorder them
            if abs_window_start > abs_window_end:
                abs_window_start, abs_window_end = abs_window_end, abs_window_start

        # Filter predictions in predictions_df[gene_id], which is sorted ascending in absolute_position
        if gene_id not in gene_pred_dict:
            if include_empty_entries:
                r = row.copy()
                r.update({
                    "window_start": window_start,
                    "window_end": window_end,
                    "sequence": extracted_seq,
                    "positions_list": [],
                    "donor_prob_list": [],
                    "acceptor_prob_list": [],
                    "neither_prob_list": []
                })
                results.append(r)
            continue

        sub_pred_df = gene_pred_dict[gene_id]
        # We'll define numeric filter:
        window_pred = sub_pred_df.filter(
            (pl.col("absolute_position") >= abs_window_start) & (pl.col("absolute_position") < abs_window_end)
        )

        if window_pred.is_empty():
            positions_list = []
            donor_probs = []
            acceptor_probs = []
            neither_probs = []
        else:
            # local positions 
            # e.g. if strand == '+', local = absolute_position - gene_start 
            # or if strand == '-', local = gene_end - absolute_position
            # Then we subtract window_start from that local. 
            if strand == '+':
                local_positions = (window_pred["absolute_position"] - gene_start).to_list()
                positions_list = [lp - window_start for lp in local_positions]
            else:
                # local_pos = gene_end - absolute_position
                local_positions = (gene_end - window_pred["absolute_position"]).to_list()
                positions_list = [lp - window_start for lp in local_positions]

            donor_probs = window_pred["donor_prob"].to_list()
            acceptor_probs = window_pred["acceptor_prob"].to_list()
            neither_probs = window_pred["neither_prob"].to_list()

        # Build final row
        row_dict = row.copy()
        row_dict.update({
            "window_start": window_start,
            "window_end": window_end,
            "sequence": extracted_seq,
            "positions_list": positions_list,
            "donor_prob_list": donor_probs,
            "acceptor_prob_list": acceptor_probs,
            "neither_prob_list": neither_probs
        })
        results.append(row_dict)

    output_df = pl.DataFrame(results)

    # reorder columns if desired
    first_cols = ["gene_id", "transcript_id"]
    new_cols = [
        "window_start", "window_end", "sequence",
        "positions_list", "donor_prob_list", "acceptor_prob_list", "neither_prob_list"
    ]
    original_cols = list(position_df.columns)
    middle_cols = [c for c in output_df.columns if (c in original_cols and c not in first_cols)]
    ordered_cols = [c for c in first_cols if c in output_df.columns] + middle_cols + new_cols
    output_df = output_df.select([c for c in ordered_cols if c in output_df.columns])

    if verbose:
        n_rows = output_df.shape[0]
        print(f"[info] Created an output DF with {n_rows} rows.")
        if n_rows > 0:
            print("[info] Example row:\n", output_df.head(1))

    return output_df


def extract_analysis_sequences_with_probabilities(
    sequence_df: pl.DataFrame,
    position_df: pl.DataFrame,
    predictions_df: pl.DataFrame,
    window_size: int = 500,
    include_empty_entries: bool = True,
    verbose: int = 1
) -> pl.DataFrame:
    """
    Extract sequences and corresponding splice probability vectors for predicted splice sites.

    Parameters
    ----------
    sequence_df : pl.DataFrame
        Columns: ['gene_id', 'sequence', 'strand'] (5' to 3').

    position_df : pl.DataFrame
        Columns: ['gene_id', 'position'] (relative positions).

    predictions_df : pl.DataFrame
        Columns: ['gene_id', 'position' (absolute), 'gene_start', 'gene_end',
                  'strand', 'donor_prob', 'acceptor_prob', 'neither_prob'].

    Returns
    -------
    pl.DataFrame
        Extracted sequences with corresponding splice probability vectors.
    """

    gene_seq_lookup = dict(zip(sequence_df["gene_id"], sequence_df["sequence"]))
    gene_strand_lookup = dict(zip(sequence_df["gene_id"], sequence_df["strand"]))

    pred_dict = {}
    for (gene_id,), sub_pred in predictions_df.group_by("gene_id"):
        gene_strand = sub_pred["strand"][0]  # Assuming consistent strand per gene
        
        if gene_strand == '+':
            sorted_pred = sub_pred.sort("absolute_position")  # ascending order
        elif gene_strand == '-':
            sorted_pred = sub_pred.sort("absolute_position", descending=True)  # descending order
        else:
            raise ValueError(f"Invalid strand '{gene_strand}' for gene {gene_id}")
        
        pred_dict[gene_id] = sorted_pred

    print(f"[debug] Predictions for {len(pred_dict)} genes.")
    print(f"[debug] genes in pred_dict: {list(pred_dict.keys())}")

    extracted_results = []

    if verbose >= 1:
        print(f"[info] Starting extraction for {position_df.shape[0]} positions.")

    for idx, row in enumerate(position_df.iter_rows(named=True)):
        gene_id = row['gene_id']
        rel_pos = row['position']

        if verbose >= 2:
            print(f"[debug] Processing row {idx}: gene_id={gene_id}, rel_pos={rel_pos}")

        if gene_id not in gene_seq_lookup:
            if verbose >= 2:
                print(f"[debug] Gene {gene_id} not found in sequence_df.")
            if include_empty_entries:
                extracted_results.append({**row, 'sequence': None, 'donor_prob_list': [],
                                          'acceptor_prob_list': [], 'neither_prob_list': []})
            continue

        seq = gene_seq_lookup[gene_id]
        strand = gene_strand_lookup[gene_id]

        seq_len = len(seq)
        window_start = max(rel_pos - window_size, 0)
        window_end = min(rel_pos + window_size, seq_len)
        extracted_seq = seq[window_start:window_end]

        if verbose >= 2:
            print(f"[debug] gene_id={gene_id}, strand={strand}, seq_len={seq_len}, window=({window_start}, {window_end})")

        if gene_id not in pred_dict:
            if verbose >= 2:
                print(f"[debug] Gene {gene_id} not found in predictions_df.")
            if include_empty_entries:
                extracted_results.append({**row, 'sequence': extracted_seq, 'donor_prob_list': [],
                                          'acceptor_prob_list': [], 'neither_prob_list': []})
            continue

        pred_sub_df = pred_dict[gene_id]
        gene_start = pred_sub_df['gene_start'][0]
        gene_end = pred_sub_df['gene_end'][0]

        if strand == '+':
            abs_window_start = gene_start + window_start
            abs_window_end = gene_start + window_end
        elif strand == '-':
            abs_window_end = gene_end - window_start
            abs_window_start = gene_end - window_end
        else:
            raise ValueError(f"Invalid strand: {strand}")

        window_pred = pred_sub_df.filter(
            (pl.col("position") >= abs_window_start) & (pl.col("position") < abs_window_end)
        ).sort("position")

        if verbose >= 2:
            print(f"[debug] abs_window=({abs_window_start}, {abs_window_end}), predictions found={window_pred.shape[0]}")

        if window_pred.is_empty():
            if include_empty_entries:
                extracted_results.append({**row, 'sequence': extracted_seq, 'donor_prob_list': [],
                                          'acceptor_prob_list': [], 'neither_prob_list': []})
            continue

        donor_probs = window_pred["donor_prob"].to_list()
        acceptor_probs = window_pred["acceptor_prob"].to_list()
        neither_probs = window_pred["neither_prob"].to_list()

        if verbose >= 2:
            print(f"[debug] Length check: seq={len(extracted_seq)}, probs={len(donor_probs)}")

        if len(extracted_seq) != len(donor_probs):
            if verbose >= 2:
                print(f"[warning] Length mismatch for gene {gene_id}: sequence length {len(extracted_seq)}, probability length {len(donor_probs)}")

        result_row = row.copy()
        result_row.update({
            'sequence': extracted_seq,
            'donor_prob_list': donor_probs,
            'acceptor_prob_list': acceptor_probs,
            'neither_prob_list': neither_probs
        })
        extracted_results.append(result_row)

    output_df = pl.DataFrame(extracted_results)

    if verbose >= 1:
        print(f"[info] Extracted {len(output_df)} sequences with probabilities.")

    return output_df


def plot_alignment_with_probabilities(
    original_sequence,
    prob_vector=None,
    splice_site_index=None,
    annotation=None,
    output_path=None,
    color_map="plasma",
    figsize=(20, 6),
    show_tokens=False,
    tokenized_sequence=None,
    attention_weights=None,
    top_k=10,
    **kwargs
):
    """
    Visualize DNA sequence with corresponding splice-site probabilities clearly marked.

    Parameters
    ----------
    original_sequence : str
        The contextual DNA sequence (5'→3').
    prob_vector : array-like of float, optional
        Probability scores per nucleotide (length = len(original_sequence)).
    splice_site_index : int, optional
        Index in `original_sequence` marking the splice site.
    annotation : dict, optional
        Contains window_start, window_end, position, strand, etc.
    output_path : str, optional
        Path to save the figure.
    color_map : str
        Colormap for tokens (if used).
    figsize : tuple
        Figure size.
    show_tokens : bool
        Whether to visualize tokens (attention/IG).
    tokenized_sequence : list of str, optional
        Tokenized representation for transformer-based visualization.
    attention_weights : np.array, optional
        Attention or IG scores per token.
    top_k : int
        Number of top tokens to highlight.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)
    
    seq_len = len(original_sequence)
    
    if annotation is None:
        annotation = {}
        
    gene_id = annotation.get("gene_id", "N/A")
    strand = annotation.get("strand", "?")
    position = annotation.get("position")
    window_start = annotation.get("window_start", 0)

    title_str = f"Gene: {gene_id}, Strand: {strand}"
    ax.set_title(title_str, fontsize=12)

    # Infer splice_site_index from annotation if not provided
    if splice_site_index is None and position is not None:
        splice_site_index = position - window_start

    # Ensure splice_site_index is within bounds
    if splice_site_index is not None and not (0 <= splice_site_index < seq_len):
        raise ValueError(f"splice_site_index ({splice_site_index}) out of bounds for sequence length {seq_len}")

    # Plot Probability Vector clearly (probabilities in blue)
    if prob_vector is not None:
        if len(prob_vector) != seq_len:
            raise ValueError(f"prob_vector length {len(prob_vector)} != seq_len {seq_len}")
        x_vals = np.arange(seq_len)
        ax.plot(x_vals, prob_vector, color="blue", linewidth=1.5, label="Splice Probability")
        ax.fill_between(x_vals, 0, prob_vector, color="blue", alpha=0.2)

        # Clearly show probability axis up to slightly more than max probability
        ax.set_ylim(-0.1, min(1.2, max(prob_vector) + 0.2))
    else:
        ax.set_ylim(-0.1, 1.2)

    # Mark predicted splice site clearly with dashed red line
    if splice_site_index is not None:
        ax.axvline(splice_site_index, color="red", linestyle="--", linewidth=1.2, label="Predicted Splice Site")
        ax.text(splice_site_index + 0.5, 1.05, "Splice Site", color="red",
                fontsize=10, ha="left", va="bottom", rotation=90)

    # Plot DNA sequence clearly just above the x-axis at y=-0.05
    base_y = -0.05
    for i, base in enumerate(original_sequence):
        ax.text(i + 0.5, base_y, base, ha="center", va="center", fontsize=8)

    # Label 5' and 3' ends with actual coordinates
    window_end = annotation.get("window_end", seq_len)
    if strand == '+':
        left_coord, right_coord = window_start, window_end
    else:
        left_coord, right_coord = window_end, window_start

    ax.text(0, base_y - 0.1, f"5' end: {left_coord}", ha="left", va="top", fontsize=9, color="black")
    ax.text(seq_len, base_y - 0.1, f"3' end: {right_coord}", ha="right", va="top", fontsize=9, color="black")

    # If requested, show token alignments above the probability curve
    if show_tokens and tokenized_sequence is not None and attention_weights is not None:
        import matplotlib.patches as patches
        cmap = plt.get_cmap(color_map)

        if attention_weights.ndim == 2:
            attention_weights = attention_weights.sum(axis=0)
        norm_attn = (attention_weights - attention_weights.min()) / (attention_weights.ptp() + 1e-8)

        token_starts = np.cumsum([0] + [len(tk) for tk in tokenized_sequence[:-1]])
        top_indices = np.argsort(norm_attn)[-top_k:][::-1]

        token_y = 1.05
        for idx in top_indices:
            tk = tokenized_sequence[idx]
            start_pos = token_starts[idx]
            tk_len = len(tk)
            attn_score = norm_attn[idx]

            rect_height = 0.1 + 0.3 * attn_score
            rect = patches.Rectangle(
                (start_pos, token_y),
                tk_len, rect_height,
                color=cmap(attn_score), alpha=0.7
            )
            ax.add_patch(rect)
            ax.text(start_pos + tk_len / 2, token_y + rect_height / 2,
                    tk, ha="center", va="center", fontsize=8, color="white")

            token_y += rect_height + 0.05

        ax.set_ylim(-0.3, token_y + 0.2)

    ax.set_xlim(0, seq_len)
    ax.set_xlabel("Nucleotide Position (5'→3')")
    ax.set_ylabel("Probability" if prob_vector is not None else "")

    ax.legend(loc="upper right")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[info] Saved plot to: {output_path}")
    else:
        plt.show()


def plot_alignment_with_probabilities_v0(
    original_sequence,
    prob_vector=None,
    splice_site_index=None,
    annotation=None,
    output_path=None,
    color_map="plasma",
    figsize=(20, 6),
    show_tokens=False,
    tokenized_sequence=None,
    attention_weights=None,
    top_k=10,
    prob_label="Probability",
    **kwargs
):
    """
    Visualize splice site probabilities and contextual sequences.
    """
    seq_len = len(original_sequence)

    # Check inputs
    if prob_vector is None or len(prob_vector) != seq_len:
        raise ValueError("prob_vector must be provided and match original_sequence length.")

    if splice_site_index is None and annotation is not None:
        window_start = annotation.get("window_start")
        window_end = annotation.get("window_end")
        splice_position = annotation.get("position")

        if window_start is not None and window_end is not None and splice_position is not None:
            splice_site_index = splice_position - window_start
        else:
            # fallback
            splice_site_index = len(original_sequence) // 2

    if splice_site_index is None or not (0 <= splice_site_index < seq_len):
        raise ValueError("splice_site_index must be a valid index within the sequence length.")

    # Extract annotation
    annotation = annotation or {}
    gene_id = annotation.get('gene_id', 'N/A')
    tx_id = annotation.get('transcript_id', 'N/A')
    splice_type = annotation.get('splice_type', 'N/A')
    pred_type = annotation.get('pred_type', 'N/A')
    strand = annotation.get('strand', '+')

    # Setup figure with two panels
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=figsize,
                                            gridspec_kw={'height_ratios': [4, 1]},
                                            sharex=True)

    # Top panel: Probability plot
    positions = np.arange(seq_len)
    cmap = plt.get_cmap(color_map)
    colors = cmap(prob_vector / np.max(prob_vector))

    ax_top.bar(positions, prob_vector, color=colors, edgecolor="none")
    ax_top.axvline(splice_site_index, color='red', linestyle='--', linewidth=1.5, label='Predicted Splice Site')

    ax_top.set_ylabel(prob_label, fontsize=12)
    ax_top.set_title(f"Splice Site Probability Alignment\nGene: {gene_id}, Transcript: {tx_id}, Splice type: {splice_type}, Pred type: {pred_type}, Strand: {strand}")
    ax_top.legend(loc='upper right')

    # Bottom panel: Raw DNA sequence
    for i, base in enumerate(original_sequence):
        ax_bottom.text(i, 0, base, fontsize=8, ha='center', va='center')

    ax_bottom.set_ylim(-0.5, 0.5)
    ax_bottom.axis('off')

    ax_bottom.text(-0.5, 0, "5'", fontsize=10, ha='right', va='center', color='blue')
    ax_bottom.text(seq_len - 0.5, 0, "3'", fontsize=10, ha='left', va='center', color='blue')

    # Adjust layout
    plt.tight_layout(h_pad=1.0)

    # Optional IG/token overlay
    if show_tokens and tokenized_sequence and attention_weights is not None:
        if attention_weights.ndim == 2:
            attention_weights = attention_weights.sum(axis=0)
        norm_weights = attention_weights / attention_weights.max()

        # Highlight top-k tokens
        top_k_idx = np.argsort(norm_weights)[-top_k:][::-1]

        for idx in top_k_idx:
            token = tokenized_sequence[idx]
            weight = norm_weights[idx]
            color = cmap(weight)
            ax_top.text(idx, prob_vector[idx] + 0.02, token, rotation=90,
                        ha='center', va='bottom', fontsize=8, color=color)

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
