import os, sys
import re
import pandas as pd
import polars as pl

import shap
import random
import math
import numpy as np

from .utils_bio import (
    normalize_strand
)

from .utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display, 
    display_dataframe_in_chunks, 
    display_dataframe
)

from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.extract_genomic_features import FeatureAnalyzer, SpliceAnalyzer

import pyBigWig  # pip install pyBigWig
from tqdm import tqdm

# Refactor to utils_plot
import seaborn as sns
import matplotlib.pyplot as plt


class VisualAnalyzer(FeatureAnalyzer):
    def __init__(self, data_dir=None, *, source='ensembl', version=None, gtf_file=None, visual_param=None, **kargs):
        super().__init__(data_dir=data_dir, source=source, version=version, gtf_file=gtf_file)
        self.visual_param = visual_param
        self.separator = kargs.get('separator', '\t')

    def path_to_bedgraph(self, track_name="SpliceAI_Predictions"): 
        return f"{self.analysis_dir}/{track_name}.bedGraph"

    def create_bedgraph(self, track_name="SpliceAI_Predictions", source='splice_positions', **kargs):
        verbose = kargs.get('verbose', 1)
        pred_type = kargs.get('pred_type', None)

        if verbose: 
            print_emphasized(f"[i/o] Reading gene features ...")
        gene_feature_df = self.retrieve_gene_features()
        display_dataframe(gene_feature_df.head(5), title="Gene features")

        # if pred_type is not None: 
        #     track_name = f"{track_name}_{pred_type}"  # Adjust track name for filtered data
        # NOTE: Track name adjusted at create_bedgraph_from_positions()

        if source.startswith('splice_position'):
            mefd = ModelEvaluationFileHandler(self.eval_dir, separator=self.separator)

            if verbose: 
                print_emphasized(f"[i/o] Reading splice positions ...")
            splice_pos_df = mefd.load_splice_positions(aggregated=True)

            if verbose: 
                display_dataframe(splice_pos_df.head(5), title="Splice positions") 

            output_file = self.path_to_bedgraph(track_name=track_name)
            create_bedgraph_from_positions(
                splice_pos_df, 
                gene_feature_df, 
                output_file, 
                track_name=track_name, pred_type=pred_type)

            if verbose: 
                print_emphasized(f"Created bedGraph file from source={source} data ...")
                print_with_indent(f"> Output: {self.path_to_bedgraph(track_name=track_name)}", indent_level=1)

        return output_file

    def load_bedgraph(self, track_name="SpliceAI_Predictions"): 
        bedgraph_file = self.path_to_bedgraph(track_name=track_name)
        return self.load_dataframe(bedgraph_file, sep=self.separator)


def extract_genes_from_gtf(gtf_file, verbose=1, output_file="ensembl_only_genes.gtf"):
    """
    Extract gene features from a GTF file.

    Alternatively, use the awk command:
    
    .utils.extract_gene_features_from_gtf_by_awk_command(gtf_file, output_file)
    """
    from .utils_bio import extract_genes_from_gtf

    gene_df = extract_genes_from_gtf(gtf_file, verbose=verbose)  # returns a Polars dataframe by default

    # Save the filtered gene features to a new GTF file
    gene_df.write_csv(output_file, separator='\t', has_header=False)

    return 


def create_error_bigwig(error_df, output_path, chrom_sizes_path, verbose=1):
    """
    Create a bigWig file to visualize splice site prediction errors.

    Parameters:
    - error_df (pd.DataFrame or pl.DataFrame): DataFrame with error data.
      Required columns: ['chrom', 'position', 'error_type'].
    - output_path (str): Path to save the generated bigWig file.
    - chrom_sizes_path (str): Path to a file containing chromosome sizes (required for bigWig creation).
    - verbose (int): If > 0, print progress messages.
    """
    # import pandas as pd

    # Ensure DataFrame is in Pandas format
    if isinstance(error_df, pl.DataFrame):
        error_df = error_df.to_pandas()

    # Validate required columns
    required_columns = {"chrom", "position", "error_type"}
    if not required_columns.issubset(error_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Map error_type to signal values
    error_type_to_signal = {"FP": 1, "FN": -1}
    error_df["signal"] = error_df["error_type"].map(error_type_to_signal)

    # Group data by chromosome for processing
    chrom_groups = error_df.groupby("chrom")

    # Initialize bigWig file
    bw = pyBigWig.open(output_path, "w")

    # Load chromosome sizes
    chrom_sizes = []
    with open(chrom_sizes_path) as f:
        for line in f:
            chrom, size = line.strip().split()
            chrom_sizes.append((chrom, int(size)))

    # Add header with chromosome sizes
    bw.addHeader(chrom_sizes)

    # Add entries to bigWig
    for chrom, group in chrom_groups:
        if verbose:
            print(f"Processing chromosome {chrom} with {len(group)} entries.")
        for _, row in group.iterrows():
            chrom = row["chrom"]
            position = int(row["position"])
            signal = float(row["signal"])
            bw.addEntries(chrom, position, position + 1, values=signal)

    # Close the bigWig file
    bw.close()

    if verbose:
        print(f"Finished creating bigWig file: {output_path}")


def create_bedgraph_from_positions(
    position_df, gene_feature_df, output_file, track_name="SpliceAI_Predictions", pred_type=None
):
    """
    Generate a bedGraph file from a position dataframe for visualization in genome browsers.

    Parameters:
    - position_df (pd.DataFrame): DataFrame containing splice site information.
      Required columns: ['gene_id', 'position', 'score', 'pred_type', 'splice_type'].
    - gene_feature_df (pd.DataFrame): DataFrame containing gene-to-chromosome mapping.
      Required columns: ['gene_id', 'chrom'].
    - output_file (str): Path to save the bedGraph file.
    - track_name (str): Name of the track for visualization (default: "SpliceAI_Predictions").
    - pred_type (str or None): If specified, filters the bedGraph to a specific prediction type (e.g., 'TP', 'FP', 'FN').

    Returns:
    - None: Outputs a bedGraph file.
    """

    # Ensure gene_feature_df has the necessary columns
    required_gene_cols = ['gene_id', 'chrom']
    for col in required_gene_cols:
        if col not in gene_feature_df.columns:
            raise ValueError(f"Column '{col}' is missing from gene feature dataframe.")

    # Convert Polars dataframes to Pandas if necessary
    if isinstance(position_df, pl.DataFrame):
        position_df = position_df.to_pandas()
    if isinstance(gene_feature_df, pl.DataFrame):
        gene_feature_df = gene_feature_df.to_pandas()

    # Merge position_df with gene_feature_df to get chromosome information
    if 'chrom' not in position_df.columns:
        position_df = position_df.merge(gene_feature_df[['gene_id', 'chrom']], on='gene_id', how='left')

    # Check required columns after merge
    required_cols = ['chrom', 'position', 'score', 'pred_type', 'splice_type']
    for col in required_cols:
        if col not in position_df.columns:
            raise ValueError(f"Column '{col}' is missing from position dataframe after merge.")

    # Filter by prediction type if specified
    if pred_type:
        if pred_type not in position_df['pred_type'].unique():
            raise ValueError(f"Specified pred_type '{pred_type}' not found in position_df.")
        position_df = position_df[position_df['pred_type'] == pred_type]
        track_name = f"{track_name}_{pred_type}"  # Adjust track name for filtered data

    # UCSC track header
    track_header = (
        f"track type=bedGraph name=\"{track_name}\" description=\"SpliceAI {pred_type or 'Predictions'}\" "
        f"visibility=full color=0,0,255 maxHeightPixels=128:64:32\n"
    )

    # Convert to bedGraph format
    bedgraph_lines = []
    for _, row in position_df.iterrows():
        chrom = row['chrom']
        start = row['position'] - 1  # Convert to 0-based start
        end = row['position']  # End = start + 1 for single-nucleotide data
        score = row['score']
        bedgraph_lines.append(f"{chrom}\t{start}\t{end}\t{score}")

    # Write to file
    with open(output_file, 'w') as f:
        f.write(track_header)
        f.write("\n".join(bedgraph_lines))
    print(f"BedGraph file saved to {output_file}.")


######################################################


def plot_feature_distributions(
    X, 
    y, 
    feature_list, 
    label_col="label",
    # Re-label classes in final plots
    label_text_0="TP",   # Instead of "Class 0"
    label_text_1="FP",   # Instead of "Class 1"
    plot_type="box", 
    n_cols=3, 
    figsize=None, 
    title="Feature Distributions",
    output_path=None, 
    show_plot=False, 
    verbose=1,
    top_k_motifs=None,        # New: Only plot top N motif features (k-mers) if provided
    kmer_pattern=r'^\d+mer_.*'  # Regex to identify motif features
):
    """
    Create plots for features based on their type, using feature type classification.
    For numeric features, produces box/violin/histograms.
    For categorical features, produces countplots.
    Optionally, if top_k_motifs is provided, restricts motif features to the top N
    based on the absolute difference in mean values between classes.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.array
        Binary labels corresponding to X's rows.
    feature_list : list
        List of column names in X to plot.
    label_col : str
        Name to use for the label column when classifying features.
    label_text_0 : str
        Text label for class 0 (default: "TP").
    label_text_1 : str
        Text label for class 1 (default: "FP").
    plot_type : str
        "box", "violin", or fallback "histplot" for numeric features.
    n_cols : int
        Number of columns in the subplot grid.
    figsize : tuple
        (width, height) in inches. If None, auto-calculated.
    title : str
        The overall figure title.
    output_path : str or None
        If given, save the figure to this path.
    show_plot : bool
        Whether to display the plot.
    verbose : int
        Level of verbosity.
    top_k_motifs : int or None
        If provided, only the top N motif features (matching kmer_pattern) will be plotted.
    kmer_pattern : str
        Regular expression to identify motif (k-mer) feature names.
    
    Returns
    -------
    None

    Updates
    -------
    - 2025-03-11: Added support for derived categorical features.
    - 2025-03-11: Added support for top_k_motifs.
    """
    from .analysis_utils import classify_features

    # Combine X and y into a temporary DataFrame for feature classification
    temp_df = X.copy()
    temp_df[label_col] = y
    feature_categories = classify_features(temp_df, label_col=label_col)
    
    # Retrieve classified feature lists
    categorical_vars = feature_categories.get("categorical_features", []) + feature_categories.get("derived_categorical_features", [])
    numerical_vars = feature_categories.get("numerical_features", [])
    motif_vars = feature_categories.get("motif_features", [])
    
    # If top_k_motifs is provided, select top N motif features based on absolute difference in means.
    if top_k_motifs is not None:
        # Filter out motif features from the provided feature_list
        motif_in_list = [feat for feat in feature_list if re.match(kmer_pattern, feat)]
        if len(motif_in_list) > 0:
            # Merge X and y for computing group means.
            df_for_calc = X.copy()
            df_for_calc[label_col] = y
            # Group by label: assume error class is 1 and correct class is 0.
            # To be generic, we compute means for each motif feature for each class.
            diff_dict = {}
            for feat in motif_in_list:
                group_means = df_for_calc.groupby(label_col)[feat].mean()
                # If either class is missing, skip the feature.
                if group_means.shape[0] < 2:
                    continue
                diff = abs(group_means.iloc[0] - group_means.iloc[1])
                diff_dict[feat] = diff
            # Sort motif features by difference, descending, and select top N.
            top_motifs = sorted(diff_dict, key=lambda k: diff_dict[k], reverse=True)[:top_k_motifs]
            if verbose:
                print(f"[info] Selected top {top_k_motifs} motif features based on mean difference: {top_motifs}")
            # Replace motif features in feature_list with the top_motifs.
            non_motif = [feat for feat in feature_list if not re.match(kmer_pattern, feat)]
            feature_list = non_motif + top_motifs

    # Determine number of features to plot and grid layout.
    n_features = len(feature_list)
    n_rows = math.ceil(n_features / n_cols)
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Convert y to a pandas Series ensuring alignment.
    y_series = pd.Series(y, index=X.index, name=label_col)

    for i, feat in enumerate(feature_list):
        ax = axes[i]
        if feat not in X.columns:
            if verbose:
                print(f"[warning] Feature '{feat}' not found in X; skipping.")
            ax.set_visible(False)
            continue

        plot_df = pd.DataFrame({feat: X[feat], label_col: y_series})
        dtype = X[feat].dtype

        if feat in numerical_vars or feat in motif_vars:
            # Numeric features (and motif features treated as numeric counts).
            if plot_type == "box":
                sns.boxplot(data=plot_df, x=label_col, y=feat, ax=ax)
                ax.set_xticks([0, 1])
                ax.set_xticklabels([label_text_0, label_text_1], fontsize=9)
            elif plot_type == "violin":
                sns.violinplot(data=plot_df, x=label_col, y=feat, ax=ax)
                ax.set_xticks([0, 1])
                ax.set_xticklabels([label_text_0, label_text_1], fontsize=9)
            else:
                sns.histplot(data=plot_df, x=feat, hue=label_col, kde=True, ax=ax)
            ax.set_title(feat, fontsize=10)
        elif feat in categorical_vars:
            # Categorical features (includes binary features).
            sns.countplot(data=plot_df, x=feat, hue=label_col, ax=ax)
            ax.set_title(feat, fontsize=10)
            ax.tick_params(axis='x', labelrotation=45)
        else:
            if verbose:
                print(f"[warning] Skipping feature '{feat}' as its type is not handled.")
            ax.set_visible(False)

    # Hide any leftover subplots.
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is not None:
        if verbose:
            print(f"[output] Saving feature distribution plot to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_feature_distributions_v0(
    X, y, 
    feature_list, 
    pos_label=1, 
    neg_label=0, 
    plot_type="box", 
    n_cols=3, 
    figsize=None, 
    title="Feature Distributions",
    output_path=None, 
    show_plot=False, 
    verbose=1
):
    """
    Create boxplots or violin plots for each feature in feature_list,
    grouping by y (pos_label vs. neg_label).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.array
        Binary labels corresponding to X's rows.
    feature_list : list
        List of column names in X to plot.
    pos_label : int
        Value of the positive class in y.
    neg_label : int
        Value of the negative class in y.
    plot_type : str
        "box" or "violin" or fallback=histplot.
    n_cols : int
        Number of columns in the subplot grid.
    figsize : tuple
        (width, height) in inches.
    title : str
        The overall figure title.
    output_path : str or None
        If given, save the figure to this path (file extension e.g. .pdf).
    show_plot : bool
        If True, display the plot via plt.show().
    verbose : int
        Level of verbosity.

    Returns
    -------
    None
    """
    n_features = len(feature_list)
    n_rows = math.ceil(n_features / n_cols)

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)  # or tweak these multipliers to taste
        # 4–5 inches for each column, 3–4 inches for each row.

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, feat in enumerate(feature_list):
        ax = axes[i]
        plot_df = X[[feat]].copy()
        plot_df["label"] = y.values

        if plot_type == "box":
            sns.boxplot(data=plot_df, x="label", y=feat, ax=ax)
        elif plot_type == "violin":
            sns.violinplot(data=plot_df, x="label", y=feat, ax=ax)
        else:
            sns.histplot(data=plot_df, x=feat, hue="label", ax=ax, kde=True)

        ax.set_title(feat, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Class {neg_label}", f"Class {pos_label}"], fontsize=9)

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    # Save the plot if output_path is specified
    if output_path is not None:
        if verbose:
            print(f"[output] Saving feature distribution plot to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    # Show or close
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pairwise_scatter(
    X, 
    y, 
    features, 
    pos_label=1, 
    neg_label=0, 
    figsize=(8,6),
    output_path=None,
    show_plot=False,
    verbose=1, 
    **kargs
):
    """
    Create a pairwise scatter (pairplot) for the given features, 
    coloring points by the label.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.array
        Binary labels.
    features : list
        List of feature names (columns in X).
    pos_label : int
        Positive label in y.
    neg_label : int
        Negative label in y.
    figsize : tuple
        Overall figure dimension. 
        Note: pairplot internally sets subplots. 
    output_path : str or None
        Path to save the figure. If None, no saving.
    show_plot : bool
        Whether to show the plot with plt.show().
    verbose : int
        Level of verbosity.

    Returns
    -------
    None
    """
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    plot_df = X[features].copy()
    plot_df["label"] = y.values

    use_inf_as_na = kargs.get("use_inf_as_na", True)

    # Convert infinite values to NaN to avoid warnings
    if use_inf_as_na:
        plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Seaborn pairplot doesn't directly respect figsize from the user;
    # Instead we can approximate with 'height' param.
    n_features = len(features)
    # E.g. height=2.5 => total width ~ 2.5*n_features
    # We'll just rely on user to specify
    
    g = sns.pairplot(
        plot_df, 
        hue="label", 
        corner=True, 
        diag_kind="hist",
        height=min(figsize[0]/n_features, figsize[1]/n_features) if n_features>1 else figsize[0], 
        plot_kws={"alpha": 0.7}
    )
    g.fig.suptitle("Pairwise Scatter", y=1.02)

    # Save if output_path
    if output_path is not None:
        if verbose:
            print(f"[output] Saving pairwise scatter to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    # Show or close
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_shap_beeswarm(
    shap_values, 
    X, 
    max_display=20, 
    output_path=None,
    show_plot=False,
    verbose=1
):
    """
    Plot a SHAP beeswarm summary (shap.summary_plot with plot_type='dot')
    for a given set of shap_values and corresponding data X.

    Parameters
    ----------
    shap_values : np.array or shap.Explanation
        The SHAP values for each sample, shape (n_samples, n_features).
    X : pd.DataFrame or np.array
        The data used to compute shap_values. 
        If pd.DataFrame, the columns are used as feature names.
    max_display : int
        Number of top features to display in the plot.
    output_path : str or None
        Where to save the figure. e.g. "shap_beeswarm.pdf"
    show_plot : bool
        If True, display the figure.
    verbose : int
        Verbosity level.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    # shap.summary_plot is the typical method for "beeswarm"
    # We'll intercept the show by shap.plots._force_matplotlib_show or show=False
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        # fallback
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # By default shap.summary_plot tries to show plot immediately
    # We can force show=False, then do a manual save
    shap.summary_plot(
        shap_values, 
        features=X, 
        feature_names=feature_names, 
        plot_type="dot",  # dot is the typical beeswarm style
        max_display=max_display, 
        show=False
    )

    if output_path is not None:
        if verbose:
            print(f"[output] Saving SHAP beeswarm plot to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


######################################################


def analyze_error_features(
    genomic_feature_df, 
    position_df, 
    feature_column='gene_type',
    error_type='FP',
    top_n=None,
    save_plot=True, 
    plot_file='error_feature_fraction.pdf',
    horizontal=False,
    verbose=True
):
    """
    Analyze the fraction of genes with prediction errors for each genomic feature (e.g., gene_type or transcript_length).

    Parameters:
    - genomic_feature_df (pd.DataFrame): DataFrame with gene or transcript features, including 'gene_id'.
    - position_df (pd.DataFrame): DataFrame with splice position predictions, including 'gene_id', 'pred_type', etc.
    - feature_column (str): Feature to analyze, e.g., 'gene_type' or 'transcript_length' (default is 'gene_type').
    - error_type (str): Type of error to analyze ('FP' or 'FN') (default is 'FP').
    - top_n (int): Number of top categories to show based on fraction of errors (optional).
    - save_plot (bool): Whether to save the plot as a file (default is True).
    - plot_file (str): File name to save the plot (default is 'error_feature_fraction.pdf').
    - horizontal (bool): Whether to display the barplot horizontally (default is False).
    - verbose (bool): Whether to print progress updates (default is True).

    Returns:
    - None
    """
    # import pandas as pd
    # import matplotlib.pyplot as plt
    import seaborn as sns

    if isinstance(genomic_feature_df, pl.DataFrame):
        genomic_feature_df = genomic_feature_df.to_pandas()
    if isinstance(position_df, pl.DataFrame):
        position_df = position_df.to_pandas()

    # Step 1: Validate inputs
    if feature_column not in genomic_feature_df.columns:
        raise ValueError(f"'{feature_column}' is not in the genomic_feature_df columns.")
    if 'gene_id' not in genomic_feature_df.columns or 'gene_id' not in position_df.columns:
        raise ValueError("Both input dataframes must contain 'gene_id' column.")
    
    # Step 2: Filter for the desired error type
    if verbose:
        print(f"Filtering positions for error type: {error_type}...")
    error_positions = position_df[position_df['pred_type'] == error_type]
    if error_positions.empty:
        raise ValueError(f"No rows found with pred_type = '{error_type}'. Check your input data.")
    
    # Step 3: Count total genes per feature category
    total_genes_per_feature = genomic_feature_df.groupby(feature_column)['gene_id'].nunique()
    total_genes_per_feature = total_genes_per_feature.reset_index(name='total_gene_count')
    
    # Step 4: Count genes with errors per feature category
    error_genes = error_positions[['gene_id']].drop_duplicates()
    merged_errors = pd.merge(
        error_genes, 
        genomic_feature_df[['gene_id', feature_column]],
        on='gene_id', 
        how='left'
    )
    error_genes_per_feature = merged_errors.groupby(feature_column)['gene_id'].nunique()
    error_genes_per_feature = error_genes_per_feature.reset_index(name='error_gene_count')
    
    # Step 5: Merge and calculate the fraction of genes with errors
    feature_error_summary = pd.merge(
        total_genes_per_feature, 
        error_genes_per_feature, 
        on=feature_column, 
        how='left'
    )
    feature_error_summary['error_gene_count'].fillna(0, inplace=True)
    feature_error_summary['fraction_with_errors'] = (
        feature_error_summary['error_gene_count'] / feature_error_summary['total_gene_count']
    ) * 100  # Convert to percentage

    # Step 6: Detect if feature_column is numerical
    if pd.api.types.is_numeric_dtype(feature_error_summary[feature_column]):
        # Scatter plot with regression line
        plt.figure(figsize=(10, 6))
        sns.regplot(
            x=feature_column, 
            y='fraction_with_errors', 
            data=feature_error_summary, 
            scatter_kws={'alpha': 0.6}, 
            line_kws={'color': 'red'}
        )
        plt.title(f"Relationship Between {feature_column.capitalize()} and {error_type} Error Fractions")
        plt.xlabel(feature_column.capitalize())
        plt.ylabel(f"Fraction of Genes with {error_type} Errors (%)")
        plt.tight_layout()
    else:
        # Sort data and focus on top N categories if applicable
        feature_error_summary = feature_error_summary.sort_values(by='fraction_with_errors', ascending=False)
        if top_n:
            feature_error_summary = feature_error_summary.head(top_n)
        
        # Bar plot for categorical data
        plt.figure(figsize=(12, 8))
        if horizontal:
            ax = sns.barplot(
                y=feature_column, 
                x='fraction_with_errors', 
                data=feature_error_summary,
                hue=feature_column, 
                palette='viridis', 
                legend=False
            )
            plt.xlabel(f"Fraction of Genes with {error_type} Errors (%)")
            plt.ylabel(feature_column.capitalize())

            # Adjust margins to prevent cutoff of long y-axis labels
            plt.subplots_adjust(left=0.3)  # Increase the left margin
        else:
            ax = sns.barplot(
                x=feature_column, 
                y='fraction_with_errors', 
                data=feature_error_summary,
                hue=feature_column, 
                palette='viridis', 
                legend=False
            )
            plt.ylabel(f"Fraction of Genes with {error_type} Errors (%)")
            plt.xlabel(feature_column.capitalize())

        plt.title(f"Fraction of Genes with {error_type} Errors by {feature_column.capitalize()}")
        plt.xticks(rotation=45, ha='right')

        # Annotate bars with counts
        for p, count in zip(ax.patches, feature_error_summary['error_gene_count']):
            if horizontal:
                ax.annotate(f"{int(count)}", 
                            (p.get_width(), p.get_y() + p.get_height() / 2), 
                            ha='left', va='center', fontsize=10, color='black')
            else:
                ax.annotate(f"{int(count)}", 
                            (p.get_x() + p.get_width() / 2, p.get_height()), 
                            ha='center', va='bottom', fontsize=10, color='black')

    # Save or display the plot
    if save_plot:
        plt.savefig(plot_file)
        if verbose:
            print(f"Plot saved as {plot_file}")
    plt.close()


def analyze_error_features_v0(
        genomic_feature_df, 
        position_df, 
        feature_column='gene_type',
        error_type='FP',
        top_n=None,
        save_plot=True, 
        plot_file='error_feature_fraction.pdf',
        horizontal=False,
        verbose=True
    ):
    """
    Analyze the fraction of genes with prediction errors for each genomic feature (e.g., gene_type or transcript_length).

    Parameters:
    - genomic_feature_df (pd.DataFrame): DataFrame with gene or transcript features, including 'gene_id'.
    - position_df (pd.DataFrame): DataFrame with splice position predictions, including 'gene_id', 'pred_type', etc.
    - feature_column (str): Feature to analyze, e.g., 'gene_type' or 'transcript_length' (default is 'gene_type').
    - error_type (str): Type of error to analyze ('FP' or 'FN') (default is 'FP').
    - top_n (int): Number of top categories to show based on fraction of errors (optional).
    - save_plot (bool): Whether to save the plot as a file (default is True).
    - plot_file (str): File name to save the plot (default is 'error_feature_fraction.pdf').
    - horizontal (bool): Whether to display the barplot horizontally (default is False).
    - verbose (bool): Whether to print progress updates (default is True).

    Returns:
    - None
    """
    # import pandas as pd
    # import matplotlib.pyplot as plt
    import seaborn as sns

    # Convert Polars dataframes to Pandas if necessary
    if isinstance(genomic_feature_df, pl.DataFrame):
        genomic_feature_df = genomic_feature_df.to_pandas()
    if isinstance(position_df, pl.DataFrame):
        position_df = position_df.to_pandas()

    # Step 1: Validate inputs
    if feature_column not in genomic_feature_df.columns:
        raise ValueError(f"'{feature_column}' is not in the genomic_feature_df columns.")
    if 'gene_id' not in genomic_feature_df.columns or 'gene_id' not in position_df.columns:
        raise ValueError("Both input dataframes must contain 'gene_id' column.")
    
    # Step 2: Filter for the desired error type
    if verbose:
        print(f"Filtering positions for error type: {error_type}...")
    error_positions = position_df[position_df['pred_type'] == error_type]
    if error_positions.empty:
        raise ValueError(f"No rows found with pred_type = '{error_type}'. Check your input data.")
    
    # Step 3: Count total genes per feature category
    total_genes_per_feature = genomic_feature_df.groupby(feature_column)['gene_id'].nunique()
    total_genes_per_feature = total_genes_per_feature.reset_index(name='total_gene_count')
    
    # Step 4: Count genes with errors per feature category
    error_genes = error_positions[['gene_id']].drop_duplicates()
    merged_errors = pd.merge(
        error_genes, 
        genomic_feature_df[['gene_id', feature_column]],
        on='gene_id', 
        how='left'
    )
    error_genes_per_feature = merged_errors.groupby(feature_column)['gene_id'].nunique()
    error_genes_per_feature = error_genes_per_feature.reset_index(name='error_gene_count')
    
    # Step 5: Merge and calculate the fraction of genes with errors
    feature_error_summary = pd.merge(
        total_genes_per_feature, 
        error_genes_per_feature, 
        on=feature_column, 
        how='left'
    )
    feature_error_summary['error_gene_count'].fillna(0, inplace=True)
    feature_error_summary['fraction_with_errors'] = (
        feature_error_summary['error_gene_count'] / feature_error_summary['total_gene_count']
    )
    
    # Step 6: Sort data by fraction_with_errors
    feature_error_summary = feature_error_summary.sort_values(by='fraction_with_errors', ascending=False)

    # Step 7: Optionally focus on top N categories
    if top_n:
        feature_error_summary = feature_error_summary.head(top_n)
    
    if verbose:
        print(f"\nFraction of Genes with {error_type} Errors by '{feature_column}':")
        print(feature_error_summary)

    # Step 8: Plot the fraction of genes with errors
    plt.figure(figsize=(12, 8))
    if horizontal:
        ax = sns.barplot(
            y=feature_column, 
            x='fraction_with_errors', 
            hue=feature_column,  # Assign x to hue
            data=feature_error_summary,
            palette='viridis'
        )
        plt.xlabel(f"Fraction of Genes with {error_type} Errors")
        plt.ylabel(feature_column.capitalize())
    else:
        ax = sns.barplot(
            x=feature_column, 
            y='fraction_with_errors', 
            hue=feature_column, 
            data=feature_error_summary,
            palette='viridis'
        )
        plt.ylabel(f"Fraction of Genes with {error_type} Errors")
        plt.xlabel(feature_column.capitalize())

    plt.title(f"Fraction of Genes with {error_type} Errors by {feature_column.capitalize()}")
    plt.xticks(rotation=45, ha='right')

    # Annotate bars with counts
    for p, count in zip(ax.patches, feature_error_summary['error_gene_count']):
        if horizontal:
            ax.annotate(f"{int(count)}", 
                        (p.get_width(), p.get_y() + p.get_height() / 2), 
                        ha='left', va='center', fontsize=10, color='black')
        else:
            ax.annotate(f"{int(count)}", 
                        (p.get_x() + p.get_width() / 2, p.get_height()), 
                        ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_file)
        if verbose:
            print(f"Plot saved as {plot_file}")
    plt.close()  # Prevent interactive display on remote machines

######################################################
# Plotting Functions


def plot_splice_sites(bed_file, threshold=0.5, output_file=None, verbose=1):
    """
    Plots the donor and acceptor splice sites along with the exons for a given transcript.
    
    Parameters:
    - bed_file (str): Path to the BED file with junctions and splice site predictions.
    - threshold (float): Probability threshold to filter donor and acceptor sites.
    - output_file (str, optional): Path to save the plot. If None, the plot will be displayed.

    Example Usage: 
        
        plot_splice_sites('splice_junctions.bed', threshold=0.9)

    """
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches

    # Load the BED file into a DataFrame
    columns = ['seqname', 'start', 'end', 'name', 'score', 'strand', 'donor_prob', 'acceptor_prob']
    df = pd.read_csv(bed_file, sep='\t', header=None, names=columns)
    
    # Determine the range for plotting
    plot_start = df['start'].min()
    plot_end = df['end'].max()
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot the transcript line
    ax.plot([plot_start, plot_end], [0, 0], color='black', lw=2)
    
    # Plot exons as black boxes
    for _, row in df.iterrows():
        exon_start = row['start']
        exon_end = row['end']
        rect = patches.Rectangle((exon_start, -0.1), exon_end - exon_start, 0.2, linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(rect)
    
    # Plot donor (green arrows pointing up) and acceptor (red arrows pointing down) sites
    for _, row in df.iterrows():
        donor_pos = row['start']
        acceptor_pos = row['end']
        
        # Plot donor site if probability is above the threshold
        if row['donor_prob'] >= threshold:
            ax.arrow(donor_pos, -0.05, 0, 0.1, head_width=0.3, head_length=1000, fc='green', ec='green')
        
        # Plot acceptor site if probability is above the threshold
        if row['acceptor_prob'] >= threshold:
            ax.arrow(acceptor_pos, 0.05, 0, -0.1, head_width=0.3, head_length=1000, fc='red', ec='red')
    
    # Set plot limits
    ax.set_xlim([plot_start - 1000, plot_end + 1000])
    ax.set_ylim([-0.3, 0.3])
    ax.axis('off')  # Hide axes

    # Display the plot
    plt.title("SpliceAI-10k score")

    # Display or save the plot
    if output_file:
        if verbose: 
            print("[i/o] Saving plot to", output_file)
        plt.savefig(output_file)
    else:
        plt.show()

    # Close the plot to free memory
    plt.close(fig)


# Todo: Refactor to visual_analyzer
def plot_splice_probabilities_vs_annotation(
    predictions_df,
    annotated_sites,
    output_dir=".",
    show_plot=False,
    save_plot=True,
    plot_format="pdf",
    verbose=1
):
    """
    For each gene in predictions_df, create a plot of position (x-axis) vs 
    SpliceAI probability (y-axis). Overlay known donor/acceptor sites.
    """
    import matplotlib.pyplot as plt
    import os

    is_polars = False
    if isinstance(predictions_df, pl.DataFrame):
        predictions_df = predictions_df.to_pandas()
        is_polars = True

    os.makedirs(output_dir, exist_ok=True)
    all_genes = predictions_df["gene_name"].unique()

    for gene in all_genes:
        sub_df = predictions_df[predictions_df["gene_name"] == gene].copy()
        if sub_df.empty:
            continue

        sub_df.sort_values("position", inplace=True)
        positions = sub_df["position"]
        donor_prob = sub_df["donor_prob"]
        acceptor_prob = sub_df["acceptor_prob"]

        plt.figure(figsize=(12, 6))
        plt.plot(positions, donor_prob, label="Donor Prob", color='blue')
        plt.plot(positions, acceptor_prob, label="Acceptor Prob", color='red')

        # If we have annotated data for this gene:
        if gene in annotated_sites:
            donors = annotated_sites[gene]["donor_positions"]
            acceptors = annotated_sites[gene]["acceptor_positions"]

            # Mark donor sites (vertical dashed lines)
            for dpos in donors:
                plt.axvline(x=dpos, color='blue', linestyle='--', alpha=0.4)
            # Mark acceptor sites (vertical dotted lines)
            for apos in acceptors:
                plt.axvline(x=apos, color='red', linestyle=':', alpha=0.4)

            plt.legend(["Donor Prob","Acceptor Prob","Annotated Donors","Annotated Acceptors"])
        else:
            plt.legend()

        plt.xlabel("Genomic Position (absolute or gene-relative)")
        plt.ylabel("Probability")
        plt.title(f"SpliceAI Probability vs. Annotated Sites: {gene}")

        if save_plot:
            out_path = os.path.join(output_dir, f"{gene}_splice_prob.{plot_format}")
            plt.savefig(out_path, bbox_inches='tight')
            if verbose:
                print(f"[i/o] Saved plot for gene '{gene}' to {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_splice_probabilities_vs_annotation_enhanced(
    predictions_df,
    annotated_sites,
    output_dir=".",
    show_plot=False,
    save_plot=True,
    plot_format="pdf",
    # Color scheme adjustments
    donor_prob_color="dodgerblue",    # bright-ish blue for predicted donor
    donor_site_color="navy",          # darker blue for true donor sites
    acceptor_prob_color="tomato",     # bright-ish red/orange for predicted acceptor
    acceptor_site_color="firebrick",  # darker red for true acceptor sites
    verbose=1
):
    """
    For each gene in predictions_df, create a 2×2 grid of plots:
      - Left column for DONOR
      - Right column for ACCEPTOR
      - Top row for PREDICTED probabilities
      - Bottom row for TRUE annotated sites

    The top-left axes: predicted donor probabilities
    The bottom-left axes: vertical lines for annotated donor sites
    The top-right axes: predicted acceptor probabilities
    The bottom-right axes: vertical lines for annotated acceptor sites

    Parameters
    ----------
    predictions_df : pd.DataFrame or pl.DataFrame
        Must have columns ["gene_name", "position", "donor_prob", "acceptor_prob"].
        If polars, it will be converted to pandas for plotting syntax.
    annotated_sites : dict
        A dict keyed by gene_name with structure:
          annotated_sites[gene]["donor_positions"] = [...]
          annotated_sites[gene]["acceptor_positions"] = [...]
        Possibly from load_splice_sites_to_dict() or a similar approach.
    output_dir : str
        Where to save the plots (if save_plot=True).
    show_plot : bool
        If True, calls plt.show() after each gene's figure.
    save_plot : bool
        If True, saves each gene's figure in output_dir with format plot_format.
    plot_format : str
        File extension, e.g. "pdf", "png", etc.

    donor_prob_color : str
        Color for the predicted donor probability line (e.g. "dodgerblue").
    donor_site_color : str
        Color for the annotated donor sites (vertical dashed lines). Typically darker than donor_prob_color (e.g. "navy").
    acceptor_prob_color : str
        Color for the predicted acceptor probability line (e.g. "tomato").
    acceptor_site_color : str
        Color for the annotated acceptor sites (vertical dashed lines). Typically darker (e.g. "firebrick").

    verbose : int
        Print progress messages if > 0.
    """
    # import polars as pl

    if isinstance(predictions_df, pl.DataFrame):
        predictions_df = predictions_df.to_pandas()

    os.makedirs(output_dir, exist_ok=True)

    # Get unique genes
    all_genes = predictions_df["gene_name"].unique()
    for gene in all_genes:
        sub_df = predictions_df[predictions_df["gene_name"] == gene].copy()
        if sub_df.empty:
            continue

        # Sort by position
        sub_df.sort_values("position", inplace=True)

        # Extract numeric arrays
        positions = sub_df["position"].values
        donor_prob = sub_df["donor_prob"].values
        acceptor_prob = sub_df["acceptor_prob"].values

        # Make the figure: 2 rows × 2 columns
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)
        # axes[0,0] = donor probability
        # axes[1,0] = donor annotation lines
        # axes[0,1] = acceptor probability
        # axes[1,1] = acceptor annotation lines

        # ---------- TOP ROW: predicted probabilities ----------
        # Donor prob (top-left)
        ax_donor_pred = axes[0, 0]
        ax_donor_pred.plot(positions, donor_prob, color=donor_prob_color, label="Donor Prob")
        ax_donor_pred.set_ylabel("Probability")
        ax_donor_pred.set_title("Donor Probability (Predicted)")
        ax_donor_pred.grid(True)

        # Acceptor prob (top-right)
        ax_acceptor_pred = axes[0, 1]
        ax_acceptor_pred.plot(positions, acceptor_prob, color=acceptor_prob_color, label="Acceptor Prob")
        ax_acceptor_pred.set_title("Acceptor Probability (Predicted)")
        ax_acceptor_pred.grid(True)

        # ---------- BOTTOM ROW: annotated sites ----------
        # Donor sites (bottom-left)
        ax_donor_annot = axes[1, 0]
        ax_donor_annot.set_title("Donor Sites (Annotated)")
        ax_donor_annot.set_xlabel("Position")
        ax_donor_annot.set_ylabel("Indicator")
        ax_donor_annot.set_yticks([0, 1])
        ax_donor_annot.set_yticklabels(["", ""])  # Hide y labels but keep 0..1 range
        ax_donor_annot.set_ylim(-0.1, 1.1)
        ax_donor_annot.grid(True)

        # Acceptor sites (bottom-right)
        ax_acceptor_annot = axes[1, 1]
        ax_acceptor_annot.set_title("Acceptor Sites (Annotated)")
        ax_acceptor_annot.set_xlabel("Position")
        ax_acceptor_annot.set_ylabel("Indicator")
        ax_acceptor_annot.set_yticks([0, 1])
        ax_acceptor_annot.set_yticklabels(["", ""])
        ax_acceptor_annot.set_ylim(-0.1, 1.1)
        ax_acceptor_annot.grid(True)

        # If annotated data available for this gene, mark them
        if gene in annotated_sites:
            donors_anno = annotated_sites[gene].get("donor_positions", [])
            acceptors_anno = annotated_sites[gene].get("acceptor_positions", [])

            # Mark donor sites with vertical lines
            for dpos in donors_anno:
                ax_donor_annot.axvline(x=dpos, color=donor_site_color, linestyle='--', alpha=0.7)

            # Mark acceptor sites with vertical lines
            for apos in acceptors_anno:
                ax_acceptor_annot.axvline(x=apos, color=acceptor_site_color, linestyle='--', alpha=0.7)

        # Add a main suptitle
        fig.suptitle(f"Gene: {gene}", fontsize=14)

        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for suptitle

        # Save or show
        if save_plot:
            out_path = os.path.join(output_dir, f"{gene}_splice_prob.{plot_format}")
            fig.savefig(out_path, bbox_inches='tight')
            if verbose:
                print(f"[i/o] Saved plot for gene '{gene}' to {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def plot_splice_probabilities_vs_annotation_4x1_v0(
    predictions_df,
    annotated_sites,
    output_dir=".",
    show_plot=False,
    save_plot=True,
    plot_format="pdf",
    # Color scheme adjustments
    donor_prob_color="dodgerblue",    # bright-ish blue for predicted donor
    donor_site_color="navy",          # darker blue for true donor sites
    acceptor_prob_color="tomato",     # bright-ish red/orange for predicted acceptor
    acceptor_site_color="firebrick",  # darker red for true acceptor sites
    figsize=(10, 12),
    verbose=1
):
    """
    For each gene in predictions_df, create a 4×1 grid of plots (4 rows, 1 column):

        Row 1: Donor Probability (Predicted)
        Row 2: Donor Sites (Annotated)
        Row 3: Acceptor Probability (Predicted)
        Row 4: Acceptor Sites (Annotated)

    This layout can help when a gene has many splice sites, giving more vertical room
    for each subplot compared to a 2×2 or 1×4 layout.

    Parameters
    ----------
    predictions_df : pd.DataFrame or pl.DataFrame
        Must have columns ["gene_name", "position", "donor_prob", "acceptor_prob"].
        If polars, it will be converted to pandas for Matplotlib plotting.
    annotated_sites : dict
        A dict keyed by gene_name with structure:
          annotated_sites[gene]["donor_positions"] = [...]
          annotated_sites[gene]["acceptor_positions"] = [...]
    output_dir : str
        Where to save the plots (if save_plot=True).
    show_plot : bool
        If True, calls plt.show() after each gene's figure.
    save_plot : bool
        If True, saves each gene's figure in output_dir with format plot_format.
    plot_format : str
        File extension, e.g. "pdf", "png", etc.
    donor_prob_color : str
        Color for the predicted donor probability line (e.g. "dodgerblue").
    donor_site_color : str
        Color for the annotated donor sites (vertical dashed lines).
    acceptor_prob_color : str
        Color for the predicted acceptor probability line (e.g. "tomato").
    acceptor_site_color : str
        Color for the annotated acceptor sites (vertical dashed lines).
    figsize : tuple
        Figure size, default (10, 12) for a tall 4×1 layout.
    verbose : int
        Print progress messages if > 0.

    Returns
    -------
    None
    """
    import polars as pl
    import os
    import matplotlib.pyplot as plt

    # If the input is a Polars DataFrame, convert to pandas
    if isinstance(predictions_df, pl.DataFrame):
        predictions_df = predictions_df.to_pandas()

    os.makedirs(output_dir, exist_ok=True)

    # For each gene_name, create a figure
    all_genes = predictions_df["gene_name"].unique()
    for gene in all_genes:
        sub_df = predictions_df[predictions_df["gene_name"] == gene].copy()
        if sub_df.empty:
            continue

        # Sort by position
        sub_df.sort_values("position", inplace=True)

        # Extract arrays
        positions = sub_df["position"].values
        donor_prob = sub_df["donor_prob"].values
        acceptor_prob = sub_df["acceptor_prob"].values

        # Create a 4×1 layout
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize, sharex=True)

        # Row 1: Donor Probability
        ax_donor_prob = axes[0]
        ax_donor_prob.plot(positions, donor_prob, color=donor_prob_color, label="Donor Prob")
        ax_donor_prob.set_title("Donor Probability (Predicted)")
        ax_donor_prob.set_ylabel("Probability")
        ax_donor_prob.grid(True)

        # Row 2: Donor Sites (Annotated)
        ax_donor_sites = axes[1]
        ax_donor_sites.set_title("Donor Sites (Annotated)")
        ax_donor_sites.set_ylabel("Indicator")
        ax_donor_sites.set_yticks([0, 1])
        ax_donor_sites.set_yticklabels(["", ""])
        ax_donor_sites.set_ylim(-0.1, 1.1)
        ax_donor_sites.grid(True)

        # Row 3: Acceptor Probability
        ax_acceptor_prob = axes[2]
        ax_acceptor_prob.plot(positions, acceptor_prob, color=acceptor_prob_color, label="Acceptor Prob")
        ax_acceptor_prob.set_title("Acceptor Probability (Predicted)")
        ax_acceptor_prob.set_ylabel("Probability")
        ax_acceptor_prob.grid(True)

        # Row 4: Acceptor Sites (Annotated)
        ax_acceptor_sites = axes[3]
        ax_acceptor_sites.set_title("Acceptor Sites (Annotated)")
        ax_acceptor_sites.set_ylabel("Indicator")
        ax_acceptor_sites.set_yticks([0, 1])
        ax_acceptor_sites.set_yticklabels(["", ""])
        ax_acceptor_sites.set_ylim(-0.1, 1.1)
        ax_acceptor_sites.grid(True)

        # Set x-label only on the bottom axis
        ax_acceptor_sites.set_xlabel("Position")

        # If annotated data for this gene, plot them
        if gene in annotated_sites:
            donors_anno = annotated_sites[gene].get("donor_positions", [])
            acceptors_anno = annotated_sites[gene].get("acceptor_positions", [])

            for dpos in donors_anno:
                ax_donor_sites.axvline(x=dpos, color=donor_site_color,
                                       linestyle='--', alpha=0.7)
            for apos in acceptors_anno:
                ax_acceptor_sites.axvline(x=apos, color=acceptor_site_color,
                                          linestyle='--', alpha=0.7)

        # Main suptitle
        fig.suptitle(f"Gene: {gene}", fontsize=14)

        # Adjust layout so suptitle has some space above subplots
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        # Save or show
        if save_plot:
            out_path = os.path.join(output_dir, f"{gene}_splice_prob_4x1.{plot_format}")
            fig.savefig(out_path, bbox_inches='tight')
            if verbose:
                print(f"[i/o] Saved plot for gene '{gene}' => {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def plot_splice_probabilities_vs_annotation_4x1(
    predictions_df,
    annotated_sites,
    output_dir=".",
    show_plot=False,
    save_plot=True,
    plot_format="pdf",
    # Color scheme adjustments
    donor_prob_color="dodgerblue",     # bright-ish blue for predicted donor
    donor_site_color="navy",           # darker blue for true donor sites
    acceptor_prob_color="tomato",      # bright-ish red/orange for predicted acceptor
    acceptor_site_color="firebrick",   # darker red for true acceptor sites
    figsize=(10, 12),
    verbose=1
):
    """
    For each gene in predictions_df, create a 4×1 grid of plots (4 rows, 1 column):
        Row 1: Donor Probability (Predicted)  [larger height]
        Row 2: Donor Sites (Annotated)        [smaller height]
        Row 3: Acceptor Probability (Predicted)  [larger height]
        Row 4: Acceptor Sites (Annotated)        [smaller height]

    By giving rows 2 and 4 a smaller height ratio, we reduce vertical space
    for annotated sites (which only need a 0–1 indicator axis). This leaves
    more vertical room for the predicted probability rows (1 and 3),
    which can improve readability when a gene has many splice sites.

    Parameters
    ----------
    predictions_df : pd.DataFrame or pl.DataFrame
        Must have columns: ["gene_name", "position", "donor_prob", "acceptor_prob"].
        If polars, it will be converted to pandas for Matplotlib plotting.
    annotated_sites : dict
        A dict keyed by gene_name with structure:
          annotated_sites[gene]["donor_positions"] = [...]
          annotated_sites[gene]["acceptor_positions"] = [...]
    output_dir : str
        Directory to save the plots (if save_plot=True).
    show_plot : bool
        If True, calls plt.show() after each gene's figure.
    save_plot : bool
        If True, saves each gene's figure in output_dir with format plot_format.
    plot_format : str
        File extension, e.g. "pdf", "png", etc.

    donor_prob_color : str
        Color for the predicted donor probability line.
    donor_site_color : str
        Color for the annotated donor sites (vertical dashed lines).
    acceptor_prob_color : str
        Color for the predicted acceptor probability line.
    acceptor_site_color : str
        Color for the annotated acceptor sites (vertical dashed lines).

    figsize : tuple
        Default (10, 12) for the overall figure size (width, height).
        The actual row heights are further controlled by 'height_ratios' below.

    verbose : int
        Print progress messages if > 0.

    Returns
    -------
    None
    """
    # import polars as pl
    # import os
    # import matplotlib.pyplot as plt

    # If polars, convert to pandas
    if isinstance(predictions_df, pl.DataFrame):
        predictions_df = predictions_df.to_pandas()

    os.makedirs(output_dir, exist_ok=True)

    all_genes = predictions_df["gene_name"].unique()
    for gene in all_genes:
        sub_df = predictions_df[predictions_df["gene_name"] == gene].copy()
        if sub_df.empty:
            continue

        sub_df.sort_values("position", inplace=True)

        positions = sub_df["position"].values
        donor_prob = sub_df["donor_prob"].values
        acceptor_prob = sub_df["acceptor_prob"].values

        # We create a 4×1 figure using GridSpec with custom height_ratios:
        # e.g. [3, 1, 3, 1] => the annotated site rows are 1/3 the height of the prob rows
        fig, axes = plt.subplots(
            nrows=4, ncols=1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1, 4, 1]}  # Adjust as desired
        )

        # Row 1: Donor Probability
        ax_donor_prob = axes[0]
        ax_donor_prob.plot(positions, donor_prob, color=donor_prob_color)
        ax_donor_prob.set_title("Donor Probability (Predicted)", fontsize=12)
        ax_donor_prob.set_ylabel("Probability")
        ax_donor_prob.grid(True)

        # Row 2: Donor Sites (Annotated)
        ax_donor_sites = axes[1]
        ax_donor_sites.set_title("Donor Sites (Annotated)", fontsize=12)
        ax_donor_sites.set_ylabel("Indicator")
        ax_donor_sites.set_yticks([0, 1])
        ax_donor_sites.set_yticklabels(["", ""])  # minimal y-labels
        ax_donor_sites.set_ylim(-0.1, 1.1)
        ax_donor_sites.grid(True)

        # Row 3: Acceptor Probability
        ax_acceptor_prob = axes[2]
        ax_acceptor_prob.plot(positions, acceptor_prob, color=acceptor_prob_color)
        ax_acceptor_prob.set_title("Acceptor Probability (Predicted)", fontsize=12)
        ax_acceptor_prob.set_ylabel("Probability")
        ax_acceptor_prob.grid(True)

        # Row 4: Acceptor Sites (Annotated)
        ax_acceptor_sites = axes[3]
        ax_acceptor_sites.set_title("Acceptor Sites (Annotated)", fontsize=12)
        ax_acceptor_sites.set_ylabel("Indicator")
        ax_acceptor_sites.set_yticks([0, 1])
        ax_acceptor_sites.set_yticklabels(["", ""])
        ax_acceptor_sites.set_ylim(-0.1, 1.1)
        ax_acceptor_sites.grid(True)

        # Label the x-axis only at the bottom
        ax_acceptor_sites.set_xlabel("Position")

        # If annotated data exist for this gene, mark them
        if gene in annotated_sites:
            donors_anno = annotated_sites[gene].get("donor_positions", [])
            acceptors_anno = annotated_sites[gene].get("acceptor_positions", [])

            for dpos in donors_anno:
                ax_donor_sites.axvline(dpos, color=donor_site_color,
                                       linestyle='--', alpha=0.7)
            for apos in acceptors_anno:
                ax_acceptor_sites.axvline(apos, color=acceptor_site_color,
                                          linestyle='--', alpha=0.7)

        # Main suptitle
        fig.suptitle(f"Gene: {gene}", fontsize=14)

        # Adjust layout so suptitle doesn't overlap subplots
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        # Save or show
        out_path = os.path.join(output_dir, f"{gene}_splice_prob_4x1.{plot_format}")
        if save_plot:
            fig.savefig(out_path, bbox_inches='tight')
            if verbose:
                print(f"[i/o] Saved 4x1 plot for gene='{gene}' => {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def plot_splice_probabilities_4x1_dynamic_width(
    predictions_df,
    annotated_sites,
    output_dir=".",
    show_plot=False,
    save_plot=True,
    plot_format="pdf",
    donor_prob_color="dodgerblue",     
    donor_site_color="navy",           
    acceptor_prob_color="tomato",      
    acceptor_site_color="firebrick",   
    base_height=12,
    min_width=10,
    scale_factor=10000,
    verbose=1
):
    """
    A variation of the 4×1 plotting function that estimates figure width
    based on the genomic (or gene-relative) position range for each gene.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must have ["gene_name","position","donor_prob","acceptor_prob"].
    annotated_sites : dict
        e.g. annotated_sites[gene]["donor_positions"], ["acceptor_positions"].
    output_dir : str
        Where to save the plots if save_plot=True.
    show_plot : bool
        If True, show via plt.show().
    save_plot : bool
        If True, save the figure in output_dir as plot_format.
    plot_format : str
        e.g. "pdf", "png", ...
    donor_prob_color : str
        color for donor probability line
    donor_site_color : str
        color for donor site vertical lines
    acceptor_prob_color : str
        color for acceptor probability line
    acceptor_site_color : str
        color for acceptor site vertical lines
    base_height : float
        The total height in inches for the 4×1 layout (default=12).
    min_width : float
        The minimum figure width in inches (default=10).
    scale_factor : float
        The factor to scale position range into inches for dynamic width.
        E.g. a larger scale_factor means less width per unit range => narrower plot.

    Returns
    -------
    None
    """
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import os
    if isinstance(predictions_df, pl.DataFrame):
        predictions_df = predictions_df.to_pandas()

    os.makedirs(output_dir, exist_ok=True)

    # For each gene
    all_genes = predictions_df["gene_name"].unique()
    for gene in all_genes:
        sub_df = predictions_df.loc[predictions_df["gene_name"] == gene].copy()
        if sub_df.empty:
            continue

        sub_df.sort_values("position", inplace=True)
        positions = sub_df["position"].values
        donor_prob = sub_df["donor_prob"].values
        acceptor_prob = sub_df["acceptor_prob"].values

        if len(positions) == 0:
            if verbose:
                print(f"[warning] No positions for gene={gene}. Skipping.")
            continue

        # 1) Compute the range of positions
        pos_min, pos_max = positions.min(), positions.max()
        pos_range = pos_max - pos_min

        # 2) Estimate figure width
        # e.g. each 'scale_factor' units in positions -> 1 inch
        # Then clamp at some minimum.
        width_in = max(min_width, pos_range / scale_factor)

        # 3) Make the figure
        fig, axes = plt.subplots(nrows=4, ncols=1,
                                 figsize=(width_in, base_height),
                                 sharex=True)

        # Row 1: Donor Probability
        ax_donor_prob = axes[0]
        ax_donor_prob.plot(positions, donor_prob, color=donor_prob_color)
        ax_donor_prob.set_title("Predicted Donor Probabilities")
        ax_donor_prob.set_ylabel("Probability")
        ax_donor_prob.grid(True)

        # Row 2: Donor Sites
        ax_donor_sites = axes[1]
        ax_donor_sites.set_title("Annotated Donor Sites")
        ax_donor_sites.set_ylabel("Indicator")
        ax_donor_sites.set_yticks([0,1])
        ax_donor_sites.set_yticklabels(["",""])
        ax_donor_sites.set_ylim(-0.1,1.1)
        ax_donor_sites.grid(True)

        # Row 3: Acceptor Probability
        ax_acceptor_prob = axes[2]
        ax_acceptor_prob.plot(positions, acceptor_prob, color=acceptor_prob_color)
        ax_acceptor_prob.set_title("Predicted Acceptor Probabilities")
        ax_acceptor_prob.set_ylabel("Probability")
        ax_acceptor_prob.grid(True)

        # Row 4: Acceptor Sites
        ax_acceptor_sites = axes[3]
        ax_acceptor_sites.set_title("Annotated Acceptor Sites")
        ax_acceptor_sites.set_ylabel("Indicator")
        ax_acceptor_sites.set_yticks([0,1])
        ax_acceptor_sites.set_yticklabels(["",""])
        ax_acceptor_sites.set_ylim(-0.1,1.1)
        ax_acceptor_sites.grid(True)
        ax_acceptor_sites.set_xlabel("Position")

        # If annotated data exist for this gene, draw vertical lines
        if gene in annotated_sites:
            donors_anno = annotated_sites[gene].get("donor_positions", [])
            acceptors_anno = annotated_sites[gene].get("acceptor_positions", [])

            for dpos in donors_anno:
                ax_donor_sites.axvline(dpos, color=donor_site_color,
                                       linestyle='--', alpha=0.7)
            for apos in acceptors_anno:
                ax_acceptor_sites.axvline(apos, color=acceptor_site_color,
                                          linestyle='--', alpha=0.7)

        # Add a global suptitle
        fig.suptitle(f"Gene: {gene} (pos_range={pos_range})", fontsize=14)

        # Tweak layout
        fig.tight_layout(rect=[0,0,1,0.94])

        # Save or show
        out_path = os.path.join(output_dir, f"{gene}_splice_prob_4x1_dyn_width.{plot_format}")
        if save_plot:
            fig.savefig(out_path, bbox_inches='tight')
            if verbose:
                print(f"[i/o] Saved 4x1 plot for gene={gene} => {out_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def plot_splice_probabilities_vs_annotation_1x4(
    predictions_df,
    annotated_sites,
    output_dir=".",
    show_plot=False,
    save_plot=True,
    plot_format="pdf",
    # Color scheme adjustments
    donor_prob_color="dodgerblue",    # bright-ish blue for predicted donor
    donor_site_color="navy",          # darker blue for true donor sites
    acceptor_prob_color="tomato",     # bright-ish red/orange for predicted acceptor
    acceptor_site_color="firebrick",  # darker red for true acceptor sites
    figsize=(16, 5),
    verbose=1
):
    """
    For each gene in predictions_df, create a 1×4 grid of plots:

        Column 1: Donor Probability (Predicted)
        Column 2: Donor Sites (Annotated)
        Column 3: Acceptor Probability (Predicted)
        Column 4: Acceptor Sites (Annotated)

    This layout can reduce clutter in cases where a gene has many splice sites.

    Parameters
    ----------
    predictions_df : pd.DataFrame or pl.DataFrame
        Must have columns ["gene_name", "position", "donor_prob", "acceptor_prob"].
        If polars, it will be converted to pandas for Matplotlib plotting.
    annotated_sites : dict
        A dict keyed by gene_name with structure:
          annotated_sites[gene]["donor_positions"] = [...]
          annotated_sites[gene]["acceptor_positions"] = [...]
    output_dir : str
        Where to save the plots (if save_plot=True).
    show_plot : bool
        If True, calls plt.show() after each gene's figure.
    save_plot : bool
        If True, saves each gene's figure in output_dir with format plot_format.
    plot_format : str
        File extension, e.g. "pdf", "png", etc.

    donor_prob_color : str
        Color for the predicted donor probability line (e.g. "dodgerblue").
    donor_site_color : str
        Color for the annotated donor sites (vertical dashed lines). 
    acceptor_prob_color : str
        Color for the predicted acceptor probability line (e.g. "tomato").
    acceptor_site_color : str
        Color for the annotated acceptor sites (vertical dashed lines).
    figsize : tuple
        Figure size, default (16, 5) for a wide 1×4 layout.
    verbose : int
        Print progress messages if > 0.

    Returns
    -------
    None
    """

    # If polars, convert to pandas
    import polars as pl
    if isinstance(predictions_df, pl.DataFrame):
        predictions_df = predictions_df.to_pandas()

    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)

    # For each gene_name, create a figure
    all_genes = predictions_df["gene_name"].unique()
    for gene in all_genes:
        sub_df = predictions_df[predictions_df["gene_name"] == gene].copy()
        if sub_df.empty:
            continue

        # Sort by position
        sub_df.sort_values("position", inplace=True)

        # Extract arrays
        positions = sub_df["position"].values
        donor_prob = sub_df["donor_prob"].values
        acceptor_prob = sub_df["acceptor_prob"].values

        # 1×4 layout
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize, sharex=True)

        # Columns
        # (1) Donor Probability
        ax_donor_pred = axes[0]
        ax_donor_pred.plot(positions, donor_prob, color=donor_prob_color, label="Donor Prob")
        ax_donor_pred.set_title("Donor Probability (Predicted)")
        ax_donor_pred.set_ylabel("Probability")
        ax_donor_pred.grid(True)

        # (2) Donor Sites (Annotated)
        ax_donor_sites = axes[1]
        ax_donor_sites.set_title("Donor Sites (Annotated)")
        ax_donor_sites.set_yticks([0, 1])
        ax_donor_sites.set_yticklabels(["", ""])
        ax_donor_sites.set_ylim(-0.1, 1.1)
        ax_donor_sites.grid(True)

        # (3) Acceptor Probability
        ax_acceptor_pred = axes[2]
        ax_acceptor_pred.plot(positions, acceptor_prob, color=acceptor_prob_color, label="Acceptor Prob")
        ax_acceptor_pred.set_title("Acceptor Probability (Predicted)")
        ax_acceptor_pred.grid(True)

        # (4) Acceptor Sites (Annotated)
        ax_acceptor_sites = axes[3]
        ax_acceptor_sites.set_title("Acceptor Sites (Annotated)")
        ax_acceptor_sites.set_yticks([0, 1])
        ax_acceptor_sites.set_yticklabels(["", ""])
        ax_acceptor_sites.set_ylim(-0.1, 1.1)
        ax_acceptor_sites.grid(True)

        # Label the x-axis only for columns 2 & 4, or you can do it for each if you prefer
        for ax_idx in [1, 3]:
            axes[ax_idx].set_xlabel("Position")

        # If annotated data for this gene, plot them
        if gene in annotated_sites:
            donors_anno = annotated_sites[gene].get("donor_positions", [])
            acceptors_anno = annotated_sites[gene].get("acceptor_positions", [])

            for dpos in donors_anno:
                ax_donor_sites.axvline(x=dpos, color=donor_site_color, linestyle='--', alpha=0.7)
            for apos in acceptors_anno:
                ax_acceptor_sites.axvline(x=apos, color=acceptor_site_color, linestyle='--', alpha=0.7)

        # Global title
        fig.suptitle(f"Gene: {gene}", fontsize=14)

        # Adjust layout: enough top margin for suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        # Save or show
        if save_plot:
            out_path = os.path.join(output_dir, f"{gene}_splice_prob_1x4.{plot_format}")
            fig.savefig(out_path, bbox_inches='tight')
            if verbose:
                print(f"[i/o] Saved plot for gene '{gene}' => {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


######################################################

def demo_analyze_error_features(**kargs):
    from .splice_error_analyzer import retrieve_splice_site_analysis_data, ErrorAnalyzer
    import itertools

    gtf_file = kargs.get('gtf_file', None)  # Set to None to use default GTF file
    result_set = retrieve_splice_site_analysis_data(gtf_file)
    error_df = result_set['error_df']  # window_start, window_end
    position_df = result_set['position_df']
    splice_sites_df = result_set['splice_sites_df']
    transcript_feature_df = result_set['transcript_feature_df']
    gene_feature_df = result_set['gene_feature_df']

    # Rename columns to avoid conflicts
    transcript_feature_df = transcript_feature_df.rename({'start': 'transcript_start', 'end': 'transcript_end'})
    transcript_feature_df = transcript_feature_df.drop(['strand', 'chrom'])

    gene_feature_df = gene_feature_df.rename({'start': 'gene_start', 'end': 'gene_end'})

    # Join the dataframes on 'gene_id'
    merged_df = gene_feature_df.join(transcript_feature_df, on='gene_id', how='inner')

    # Your code to process the merged dataframe
    display_dataframe_in_chunks(merged_df.head(5))
    print(f"[info] Merged dataframe columns:\n{list(merged_df.columns)}\n")
    # NOTE: 
    # ['gene_start', 'gene_end', 'score', 'strand', 'gene_id', 'gene_name', 'gene_type', 'gene_length', 
    #  'chrom', 'transcript_start', 'transcript_end', 'transcript_id', 'transcript_name', 'transcript_type', 'transcript_length']

    feature_columns = ['gene_type', 'transcript_type', 'transcript_length', ]
    error_types = ['FP', 'FN']

    for feature_column, error_type in itertools.product(feature_columns, error_types):
        print_emphasized(f"Processing combination: {feature_column} vs {error_type} ...")
        plot_file_path = os.path.join(ErrorAnalyzer.analysis_dir, f'plot_relation_{feature_column}_vs_{error_type}.pdf')

        analyze_error_features(
            merged_df, 
            position_df, 
            feature_column=feature_column, 
            error_type=error_type, 
            top_n=10,
            horizontal=True,
            save_plot=True, 
            plot_file=plot_file_path, 
            verbose=1)


def demo_create_bedgraph(**kargs): 
    from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
    # from meta_spliceai.splice_engine.extract_genomic_features import FeatureAnalyzer

    use_mock_data = False
    eval_dir = kargs.get('eval_dir', VisualAnalyzer.eval_dir)
    separator = kargs.get('separator', kargs.get('sep', '\t'))

    if use_mock_data: 
        # Example position dataframe
        position_df = pd.DataFrame({
            'gene_id': ['ENSG00000224079', 'ENSG00000224079', 'ENSG00000233082'],
            'position': [1763, 324, 199],
            'pred_type': ['FN', 'TP', 'FP'],
            'score': [1.76e-7, 0.68324, 0.79099],
            'splice_type': ['donor', 'donor', 'donor']
        })

        # Example gene feature dataframe (derived from GTF)
        gene_feature_df = pd.DataFrame({
            'gene_id': ['ENSG00000224079', 'ENSG00000233082', 'ENSG00000236708'],
            'chrom': ['1', '1', '2']
        })

        output_path = kargs.get('output_path', 'output.bedGraph')
        create_bedgraph_from_positions(position_df, gene_feature_df, output_path)

    else: 
        # mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)

        # error_df = mefd.load_error_analysis_df(aggregated=True)
        va = VisualAnalyzer()

        # splice_pos_df = mefd.load_splice_positions(aggregated=True)
        # display_dataframe(splice_pos_df.head(5), title="Splice positions") 

        # gene_feature_df = va.retrieve_gene_features()
        # display_dataframe(gene_feature_df.head(5), title="Gene features")

        # Create bedGraph file
        # create_bedgraph_from_positions(splice_pos_df, gene_feature_df, "output.bedGraph")

        bedgraph_path = va.create_bedgraph(track_name="SpliceAI_Predictions", source='splice_positions', verbose=1)

        pred_types = ['TP', 'FP', 'FN']
        for pred_type in pred_types: 
            print_emphasized(f"Creating bedGraph for pred_type={pred_type} ...")
            bedgraph_path = va.create_bedgraph(track_name="SpliceAI_Predictions", source='splice_positions', pred_type=pred_type, verbose=1)



def demo(): 

    # demo_create_bedgraph()

    demo_analyze_error_features()


if __name__ == "__main__":
    demo()

