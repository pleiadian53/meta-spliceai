import numpy as np
import pandas as pd
import polars as pl
import math
from collections import defaultdict
import matplotlib.pyplot as plt


def aggregate_shap_values_v0(shap_values, aggregation='mean', verbose=0):
    """
    Robustly aggregate SHAP values across classes into a single global importance vector,
    handling binary, multiclass, and regression cases, including different SHAP output formats.

    Parameters
    ----------
    shap_values : numpy array or list of numpy arrays
        - Binary classification or regression: array (n_samples, n_features)
        - Multiclass: 
          - list of arrays [(n_samples, n_features), ...], OR
          - single 3D array (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
    aggregation : str, default='mean'
        Method to aggregate SHAP values across classes ('mean' or 'sum').

    Returns
    -------
    aggregated_shap : numpy array (n_samples, n_features)
        Aggregated SHAP values.

    Memo
    ----
    - This function additionally (relative to v0) takes care of the case 
      where the SHAP values are a 3D array (possible but rare).
    """
    # import numpy as np

    if verbose:
        if isinstance(shap_values, list):
            print(f"[debug] Multiclass case: shap_values is a list of length {len(shap_values)}. ")
            print(f"[debug] Each class shap shape: {shap_values[0].shape}")
        elif isinstance(shap_values, np.ndarray):
            print(f"[debug] shap_values ndarray shape: {shap_values.shape}, ndim={shap_values.ndim}")
        else:
            print(f"[debug] Unrecognized shap_values type: {type(shap_values)}")

    # Case 1: list of arrays (typical multiclass)
    if isinstance(shap_values, list):
        # print(f"[debug] Detected multiclass with {len(shap_values)} classes.")
        # print(f"[debug] Each class shap shape: {shap_values[0].shape}")
        
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)  # (n_classes, n_samples, n_features)
        axis_to_agg = 0  # aggregate over classes

    # Case 2: single numpy array
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 2:
            # Binary classification or regression: already 2D
            return np.abs(shap_values)
        elif shap_values.ndim == 3:
            # Multiclass tensor (rare, but possible)
            # Determine class axis: typically either 0 or 2
            if shap_values.shape[0] <= shap_values.shape[-1]:
                # (n_classes, n_samples, n_features)
                axis_to_agg = 0
            else:
                # (n_samples, n_features, n_classes)
                axis_to_agg = -1
            stacked = np.abs(shap_values)
        else:
            raise ValueError(f"Unexpected shap_values array with {shap_values.ndim} dimensions.")
    else:
        raise TypeError("shap_values must be a numpy array or a list of numpy arrays.")

    # Perform aggregation across identified axis
    if aggregation == 'mean':
        aggregated_shap = stacked.mean(axis=axis_to_agg)
    elif aggregation == 'sum':
        aggregated_shap = stacked.sum(axis=axis_to_agg)
    else:
        raise ValueError("aggregation must be 'mean' or 'sum'")

    return aggregated_shap


def aggregate_shap_values(shap_values, aggregation='mean', verbose=0):
    """
    Aggregate SHAP values to a consistent 2D format (n_samples, n_features),
    handling binary, multiclass, and regression cases.

    Parameters
    ----------
    shap_values : array or list
        SHAP values returned by shap.Explainer:
          - Binary classification or regression: ndarray of shape (n_samples, n_features)
          - Multiclass: either
              - list of arrays [(n_samples, n_features), ...] OR
              - single 3D array of shape (n_samples, n_features, n_classes) or
                (n_classes, n_samples, n_features)

    aggregation : {'mean', 'sum'}, default='mean'
        Method used to aggregate across classes in multiclass scenarios.

    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    aggregated_shap : ndarray of shape (n_samples, n_features)
        Aggregated SHAP values suitable for global analysis.
    """
    if isinstance(shap_values, list):
        # Typical multiclass scenario: list of (n_samples, n_features)
        if verbose:
            print(f"[aggregate_shap_values] Multiclass list detected with {len(shap_values)} classes.")
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)  # shape: (n_classes, n_samples, n_features)
        axis_to_agg = 0

    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 2:
            # Binary classification or regression scenario; already 2D
            if verbose:
                print("[aggregate_shap_values] Binary/regression 2D array detected.")
            return shap_values  # Already the desired shape

        elif shap_values.ndim == 3:
            # 3D tensor for multiclass: infer the classes axis
            if verbose:
                print(f"[aggregate_shap_values] 3D array detected with shape: {shap_values.shape}")

            if shap_values.shape[0] in (shap_values.shape[1], shap_values.shape[2]):
                axis_to_agg = 0
            else:
                axis_to_agg = -1  # Typically (n_samples, n_features, n_classes)

            stacked = np.abs(shap_values)
        else:
            raise ValueError(f"Unexpected shap_values dimension: {shap_values.ndim}")

    else:
        raise TypeError(f"Unrecognized shap_values type: {type(shap_values)}")

    # Aggregate along the identified axis
    if aggregation == 'mean':
        aggregated_shap = stacked.mean(axis=axis_to_agg)
    elif aggregation == 'sum':
        aggregated_shap = stacked.sum(axis=axis_to_agg)
    else:
        raise ValueError("aggregation must be 'mean' or 'sum'")

    return aggregated_shap

#######################################################

def compare_feature_importance_ranks(
    importance_dfs, 
    top_k=20,
    primary_method='SHAP',
    output_path=None, 
    verbose=1, 
    save=True,
    plot_style="percentile_rank",
    ensure_method_representation=True  # New parameter to ensure all methods are represented
):
    """
    Compare feature importance across multiple methods.

    Purpose
    -------
    The function helps data scientists and ML practitioners understand how different feature 
    importance techniques rank the same set of features. This is valuable because:

    - Different methods can produce different rankings
    - Features that rank highly across multiple methods are likely more robust/reliable
    - It provides a standardized way to compare importance metrics that might have different scales

    Parameters
    ----------
    importance_dfs : dict
        Dictionary where keys are method names (e.g., 'SHAP', 'XGBoost', 'Hypothesis')
        and values are DataFrames with columns ['feature', 'importance_score'].
    top_k : int
        Number of top features to compare (based on the primary method).
    primary_method : str
        The key in importance_dfs used to select the top_k features.
    output_path : str or None
        Where to save the figure. Defaults to "feature_ranking_comparison.pdf" if None.
    verbose : int
        Verbosity level.
    save : bool
        If True, saves the plot; otherwise, displays it.
    plot_style : {"rank", "reverse_rank", "raw", "percentile_rank"}
        Representation style for importance scores.
    ensure_method_representation : bool, default=True
        If True, ensures that all methods have at least some of their top features represented.

    Returns
    -------
    None
    """

    # Merge all dataframes on 'feature'
    merged_df = None
    for method, df in importance_dfs.items():
        df_renamed = df.rename(columns={'importance_score': method})
        if merged_df is None:
            merged_df = df_renamed
        else:
            merged_df = merged_df.merge(df_renamed, on='feature', how='outer')

    # Compute ranks
    rank_cols = {}
    for method in importance_dfs.keys():
        rank_col = f"{method}_rank"
        merged_df[rank_col] = merged_df[method].rank(ascending=False, method="min")
        rank_cols[method] = rank_col

        # Explicitly fill NaN ranks with the next available rank
        max_rank = merged_df[rank_col].max()
        merged_df[rank_col] = merged_df[rank_col].fillna(max_rank + 1)
    # NOTE: 
    #   - Standardization via ranks: 
    #     Different methods might produce scores in different scales 
    #     (e.g., SHAP values vs. hypothesis p-values). Converting them to ranks makes them 
    #   directly comparable.
    #   - method="min" means if two features have the same importance score, 
    #     they receive the same (minimal) rank. The next rank is adjusted accordingly.
    #   - By default, Pandas' .rank() assigns NaN ranks to NaN values.

    # Explicitly use 'SHAP' or provided primary method for selecting top_k features
    primary_rank_col = rank_cols.get(f"{primary_method}_rank", next(iter(rank_cols.values())))
    
    # Select top features based on primary method
    if ensure_method_representation and len(importance_dfs) > 1 and top_k >= len(importance_dfs):
        # If we want to ensure all methods are represented, take some top features from each method
        features_per_method = max(1, top_k // len(importance_dfs))
        top_features_set = set()
        
        # Get top features from each method
        print(f"[info] Ensuring balanced representation - selecting ~{features_per_method} features from each method")
        for method, rank_col in rank_cols.items():
            method_top_features = merged_df.nsmallest(features_per_method, rank_col)['feature'].tolist()
            top_features_set.update(method_top_features)
        # NOTE: Why .nsmallest() instead of .nlargest()?
        # - .nsmallest() is more intuitive for this context: 
        #   We want the smallest ranks (i.e., highest importance scores) from each method.
        # - .nlargest() would give us the highest ranks (i.e., lowest importance scores) from each method.
        # - Pandas .nsmallest() ignores NaN ranks.

        # If we haven't reached top_k features yet, fill remainder from primary method
        remaining_slots = top_k - len(top_features_set)
        if remaining_slots > 0:
            primary_features = merged_df.nsmallest(top_k, primary_rank_col)['feature'].tolist()
            for feature in primary_features:
                if feature not in top_features_set and remaining_slots > 0:
                    top_features_set.add(feature)
                    remaining_slots -= 1
        
        # Get the final top features dataframe
        top_features = merged_df[merged_df['feature'].isin(top_features_set)]
    else:
        # Default behavior - just take top_k from primary method
        top_features = merged_df.nsmallest(top_k, primary_rank_col)

    # Prepare plot columns based on plot_style
    plot_cols = []

    if plot_style == "rank":
        plot_cols = list(rank_cols.values())
        ylabel = "Rank (lower is better)"

    elif plot_style == "reverse_rank":
        for method, rank_col in rank_cols.items():
            max_rank = top_features[rank_col].max()
            top_features.loc[:, f"{method}_inv"] = (max_rank + 1) - top_features[rank_col]
            plot_cols.append(f"{method}_inv")
        ylabel = "Inverted Rank (higher is better)"

    elif plot_style == "raw":
        top_features = top_features.fillna(0)
        plot_cols = list(importance_dfs.keys())
        ylabel = "Raw Importance Score"

    elif plot_style == "percentile_rank":
        for method, rank_col in rank_cols.items():
            max_rank = top_features[rank_col].max()
            top_features.loc[:, f"{method}_pct"] = (max_rank - top_features[rank_col] + 1) / max_rank
            plot_cols.append(f"{method}_pct")
        ylabel = "Percentile Rank (1 = top feature)"
    else:
        raise ValueError("Invalid plot_style. Choose from 'rank', 'reverse_rank', 'raw', or 'percentile_rank'.")

    # Sort by rank and feature name for better visualization
    top_features = top_features.sort_values(by=[primary_rank_col, 'feature'])

    # Plot
    fig = plt.figure(figsize=(12, max(8, 0.4 * len(top_features))))
    ax = top_features.plot(
        x="feature",
        y=plot_cols,
        kind="barh",
        alpha=0.85,
        edgecolor="black",
        figsize=(12, max(8, 0.4 * len(top_features)))
    )

    ax.set_xlabel(ylabel)
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance Ranking Comparison")
    ax.invert_yaxis()

    # Move legend outside clearly
    ax.legend(
        [col.replace("_rank", "").replace("_inv", "").replace("_pct", "") for col in plot_cols],
        bbox_to_anchor=(1.01, 0.5),
        loc="center left"
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save or show
    if save:
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "feature_ranking_comparison.pdf")
        if verbose:
            print(f"[output] Saving plot to: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def compare_feature_importance_ranks_v0(
    xgb_df, 
    shap_df, 
    hypo_df, 
    top_k=20, 
    output_path=None, 
    verbose=1, 
    save=True,
    plot_style="percentile_rank"
):
    """
    Compare feature importance across SHAP, XGBoost, and Hypothesis Testing.
    
    This function merges feature-importance data from three sources:
      - SHAP (shap_df)
      - XGBoost (xgb_df)
      - Hypothesis Testing (hypo_df)
    
    It then ranks each feature (in descending order so that the most important
    features receive rank 1) and converts these ranks into a standardized scale (if requested)
    so that the relative ordering can be compared visually.
    
    Parameters
    ----------
    xgb_df : pd.DataFrame
        DataFrame with columns ['feature', 'importance_score'] from XGBoost feature importance.
    shap_df : pd.DataFrame
        DataFrame with columns ['feature', 'importance_score'] from SHAP analysis.
    hypo_df : pd.DataFrame
        DataFrame with columns ['feature', 'importance_score'] from hypothesis testing.
    top_k : int
        Number of top features to compare (based on SHAP ranking).
    output_path : str or None
        Where to save the figure. Defaults to "feature_ranking_comparison.pdf" if None.
    verbose : int
        Print progress if > 0.
    save : bool
        If True, save the plot to output_path; else show with plt.show().
    plot_style : {"rank", "reverse_rank", "raw", "percentile_rank"}
        Determines how to represent the bars:
          - "rank": (original) lower numeric value means higher importance.
          - "reverse_rank": inverts ranks so that a higher numeric value means higher importance.
          - "raw": plot the raw importance scores.
          - "percentile_rank": convert each method's rank into a percentile score (0–1 scale, where 1 = best).

    Returns
    -------
    None
        Displays or saves a horizontal bar chart comparing the three methods.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1) Merge the three dataframes based on 'feature'
    merged_df = (
        shap_df.rename(columns={'importance_score': 'SHAP'})
        .merge(xgb_df.rename(columns={'importance_score': 'XGBoost'}), on='feature', how='outer')
        .merge(hypo_df.rename(columns={'importance_score': 'Hypothesis'}), on='feature', how='outer')
    )

    # 2) Compute ranks (lower rank = better)
    merged_df["SHAP_rank"] = merged_df["SHAP"].rank(ascending=False, method="min")
    merged_df["XGBoost_rank"] = merged_df["XGBoost"].rank(ascending=False, method="min")
    merged_df["Hypothesis_rank"] = merged_df["Hypothesis"].rank(ascending=False, method="min")

    # 3) Select top_k features based on SHAP rank (you could choose another method)
    top_features = merged_df.nsmallest(top_k, "SHAP_rank")

    # 4) Process plot_style
    if plot_style == "rank":
        plot_cols = ["SHAP_rank", "XGBoost_rank", "Hypothesis_rank"]
        ylabel = "Feature Rank (lower is better)"
    elif plot_style == "reverse_rank":
        max_shap = top_features["SHAP_rank"].max()
        max_xgb = top_features["XGBoost_rank"].max()
        max_hypo = top_features["Hypothesis_rank"].max()

        top_features["SHAP_rank_inv"] = (max_shap + 1) - top_features["SHAP_rank"]
        top_features["XGBoost_rank_inv"] = (max_xgb + 1) - top_features["XGBoost_rank"]
        top_features["Hypothesis_rank_inv"] = (max_hypo + 1) - top_features["Hypothesis_rank"]

        plot_cols = ["SHAP_rank_inv", "XGBoost_rank_inv", "Hypothesis_rank_inv"]
        ylabel = "Inverted Rank (higher is better)"

    elif plot_style == "raw":
        top_features[["SHAP", "XGBoost", "Hypothesis"]] = top_features[["SHAP", "XGBoost", "Hypothesis"]].fillna(0)
        plot_cols = ["SHAP", "XGBoost", "Hypothesis"]
        ylabel = "Raw Importance Score"

    elif plot_style == "percentile_rank":
        max_shap = top_features["SHAP_rank"].max()
        max_xgb = top_features["XGBoost_rank"].max()
        max_hypo = top_features["Hypothesis_rank"].max()

        top_features["SHAP_pct"] = (max_shap - top_features["SHAP_rank"] + 1) / max_shap
        top_features["XGBoost_pct"] = (max_xgb - top_features["XGBoost_rank"] + 1) / max_xgb
        top_features["Hypothesis_pct"] = (max_hypo - top_features["Hypothesis_rank"] + 1) / max_hypo

        plot_cols = ["SHAP_pct", "XGBoost_pct", "Hypothesis_pct"]
        ylabel = "Percentile Rank (1 = top feature)"
    else:
        raise ValueError("Invalid plot_style. Choose from 'rank', 'reverse_rank', 'raw', or 'percentile_rank'.")

    # 5) Sort features for plotting.
    if plot_style in ["rank"]:
        top_features_sorted = top_features.sort_values(plot_cols[0], ascending=True)
    else:
        top_features_sorted = top_features.sort_values(plot_cols[0], ascending=False)

    # 6) Plot horizontal bar chart.
    fig = plt.figure(figsize=(12, 8))
    ax = top_features_sorted.plot(
        x="feature",
        y=plot_cols,
        kind="barh",
        alpha=0.85,
        edgecolor="black"
    )
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance Ranking Comparison")
    ax.invert_yaxis()  # Top feature at top.

    # Adjust legend: place it completely to the right by using bbox_to_anchor
    ax.legend(["SHAP", "XGBoost", "Hypothesis Testing"],
              loc="center left", bbox_to_anchor=(1.02, 0.5))

    # Extend the right margin so legend doesn't cover any bars.
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # 7) Save or display
    if save:
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "feature_ranking_comparison.pdf")
        if verbose:
            print(f"[output] Saving feature ranking comparison to: {output_path}")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def compare_feature_rankings(
    importance_df_list,
    method_names=None,
    top_k=20,
    verbose=True
):
    """
    Compare feature rankings from multiple importance DataFrames. 
    Each df has columns ["feature","importance_score"] sorted descending.
    Returns a summary of top-k overlaps and pairwise rank correlations.
    """
    from scipy.stats import spearmanr, kendalltau

    if method_names is None:
        method_names = [f"Method{i+1}" for i in range(len(importance_df_list))]

    # Ensure each DF is sorted descending by "importance_score"
    sorted_dfs = []
    for df in importance_df_list:
        df_sorted = df.sort_values(by="importance_score", ascending=False).reset_index(drop=True)
        sorted_dfs.append(df_sorted)
    
    # Summarize top-k overlap
    n_methods = len(sorted_dfs)
    overlap_results = {}
    for i in range(n_methods):
        for j in range(i+1, n_methods):
            df_i = sorted_dfs[i]
            df_j = sorted_dfs[j]
            top_i = set(df_i["feature"].iloc[:top_k])
            top_j = set(df_j["feature"].iloc[:top_k])
            overlap = len(top_i & top_j)
            overlap_ratio = overlap / top_k
            if verbose:
                print(f"Overlap between {method_names[i]} and {method_names[j]} in top {top_k}: "
                      f"{overlap} / {top_k} = {overlap_ratio:.2f}")
            overlap_results[(i,j)] = overlap_ratio
    
    # Summarize rank correlation (Spearman or Kendall)
    # 1) build rank dict for each method
    rank_dicts = []
    all_features = set()
    for df in sorted_dfs:
        rank_map = {}
        for idx, row in df.iterrows():
            rank_map[row["feature"]] = idx+1  # rank = 1..N
        rank_dicts.append(rank_map)
        all_features |= set(rank_map.keys())
    
    # intersection if you only want to measure correlation for features that appear in all methods
    # or union if you want to consider all. Typically intersection is better.
    common_features = set.intersection(*[set(rd.keys()) for rd in rank_dicts])
    
    correlation_results = {}
    for i in range(n_methods):
        for j in range(i+1, n_methods):
            # build two rank lists
            ranks_i, ranks_j = [], []
            for feat in common_features:
                ranks_i.append(rank_dicts[i][feat])
                ranks_j.append(rank_dicts[j][feat])
            # compute spearman or kendall
            spearman_corr, _ = spearmanr(ranks_i, ranks_j)
            kendall_corr, _ = kendalltau(ranks_i, ranks_j)
            if verbose:
                print(f"Rank correlation between {method_names[i]} and {method_names[j]} over {len(common_features)} feats: "
                      f"Spearman={spearman_corr:.3f}, Kendall={kendall_corr:.3f}")
            correlation_results[(i,j)] = (spearman_corr, kendall_corr)

    return overlap_results, correlation_results

####################################################################

def build_token_frequency_comparison(
    top_token_counts,
    num_examples,
    error_label="FP",
    correct_label="TP", 
    verbose=0
):
    """
    Compare token frequencies between error and correct examples.

    Parameters:
    - top_token_counts (dict): A dictionary with token counts for error and correct examples.
    - num_examples (dict): A dictionary with the number of examples for error and correct labels.
    - error_label (str): The label for error examples (default is "FP").
    - correct_label (str): The label for correct examples (default is "TP").

    Returns:
    - list: A list of tuples (token, error_freq, correct_freq, diff) sorted by absolute difference.
        - token (str): The token.
        - error_freq (float): Frequency of the token in error examples.
        - correct_freq (float): Frequency of the token in correct examples.
        - diff (float): Difference between error_freq and correct_freq.
    """
    # import math
    # from collections import defaultdict

    # gather all tokens from both sets
    all_tokens = set(top_token_counts[error_label].keys()) | set(top_token_counts[correct_label].keys())

    result_list = []
    e_count = num_examples[error_label]
    c_count = num_examples[correct_label]

    for tk in all_tokens:
        e_freq = 0.0
        c_freq = 0.0
        if e_count > 0:
            e_freq = top_token_counts[error_label][tk] / e_count
        if c_count > 0:
            c_freq = top_token_counts[correct_label][tk] / c_count
        diff = e_freq - c_freq
        result_list.append((tk, e_freq, c_freq, diff))

    # sort by absolute difference
    result_list.sort(key=lambda x: abs(x[3]), reverse=True)

    if verbose > 0:
        # Verify the size of the output
        expected_size = len(all_tokens)
        actual_size = len(result_list)
        if actual_size != expected_size:
            print(f"[warning] Expected {expected_size} tokens, but got {actual_size}.")
        else:
            print(f"[info] Output size is correct: {actual_size} tokens.")

    return result_list


def bar_chart_freq(result_list, error_label="FP", correct_label="TP", top_n=20, output_path=None, token_type=None, **kargs):
    """
    Plot a bar chart comparing token frequencies between error and correct examples.

    Use cases: 
    - Local-Frequency bar chart 
    - Per-gene bar chart
    - Global-Frequency bar chart

    See splice_engine/error_sequence_model.py for more details.
    """
    #import matplotlib.pyplot as plt
    # import numpy as np

    y_label_text = kargs.get("y_label", None)

    # slice top_n
    data = result_list[:top_n]
    tokens = [x[0] for x in data]
    efreqs = [x[1] for x in data]
    cfreqs = [x[2] for x in data]
    diffs  = [x[3] for x in data]

    x = np.arange(len(data))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12,6))
    rects_e = ax.bar(x - width/2, efreqs, width, label=error_label)
    rects_c = ax.bar(x + width/2, cfreqs, width, label=correct_label)

    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha="right")

    if y_label_text is None:
        if token_type is None:
            ax.set_ylabel("Frequency in Top-K tokens")
        else:
            ax.set_ylabel(f"Frequency in Top-K {token_type} tokens")  # E.g. IG tokens
    else:
        ax.set_ylabel(y_label_text)

    ax.set_title(f"Comparison: {error_label} vs {correct_label} (top {top_n} tokens by abs diff)")
    ax.legend()

    # optional numeric annotations
    for rect in rects_e:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    for rect in rects_c:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[info] Saved token frequency barchart to: {output_path}")
        plt.close(fig)
    else:
        plt.show()


def bar_chart_local_feature_importance(result_list, error_label="FP", correct_label="TP", top_n=20, output_path=None, **kargs):
    """
    Plot a bar chart comparing top local feature frequencies between error and correct examples.
    
    This plot helps identify which features consistently differentiate error cases 
    (FP/FN) from correct cases (TP/TN) by summarizing local SHAP feature importance 
    across all predicted splice sites.

    Use cases: 
    - Local-Frequency bar chart 
    - Per-gene bar chart
    - Global-Frequency bar chart

    See splice_engine/error_sequence_model.py for more details.
    """
    # slice top_n
    data = result_list[:top_n]
    tokens = [x[0] for x in data]  # tokens really mean local features here
    efreqs = [x[1] for x in data]
    cfreqs = [x[2] for x in data]
    diffs  = [x[3] for x in data]

    x = np.arange(len(data))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12,6))
    rects_e = ax.bar(x - width/2, efreqs, width, label=error_label)
    rects_c = ax.bar(x + width/2, cfreqs, width, label=correct_label)

    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha="right")

    y_label_text = kargs.get("y_label", None)

    if y_label_text is None:
        ax.set_ylabel("Frequency in Top Locally Important Features")
    else:
        ax.set_ylabel(y_label_text)

    ax.set_title(f"Comparison: {error_label} vs {correct_label} (top {top_n} local features by abs diff)")
    ax.legend()

    # optional numeric annotations
    for rect in rects_e:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    for rect in rects_c:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[info] Saved local feature frequency barchart to: {output_path}")
        plt.close(fig)
    else:
        plt.show()


def get_unique_labels(y):
    """
    Get unique values from a list, numpy array, Series, or single-column DataFrame.

    Parameters:
    - y: list, numpy array, pandas/polars Series, or single-column DataFrame.

    Returns:
    - A set of unique values.
    """
    if isinstance(y, (pd.DataFrame, pl.DataFrame)):
        # Ensure it's a single-column DataFrame
        if y.shape[1] != 1:
            raise ValueError("DataFrame must have only one column.")
        # Polars doesn't support iloc; use appropriate indexing
        unique_values = y[:, 0].unique() if isinstance(y, pl.DataFrame) else y.iloc[:, 0].unique()
    elif isinstance(y, (pd.Series, pl.Series)):
        unique_values = y.unique()
    elif isinstance(y, np.ndarray):
        unique_values = np.unique(y)
    elif isinstance(y, list):
        unique_values = set(y)
    else:
        raise TypeError("Input must be a list, numpy array, Series, or single-column DataFrame.")

    # Ensure consistent return type
    return set(unique_values)


def subsample_data(X, y, max_samples=20000, random_state=42, verbose=1):
    """
    Subsample data to keep dataset size manageable for SHAP analysis.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature dataset.
    y : pd.Series, pd.DataFrame, or np.ndarray
        Target values.
    max_samples : int, default=20000
        Maximum number of samples allowed.
    random_state : int, default=42
        Seed for reproducibility.
    verbose : int, default=1
        Verbosity level.
    
    Returns
    -------
    X_subsampled, y_subsampled
    """
    n_samples = X.shape[0]
    if n_samples <= max_samples:
        if verbose:
            print(f"[info] No subsampling required (n_samples={n_samples}).")
        return X, y

    if verbose:
        print(f"[info] Subsampling data from {n_samples} to {max_samples} samples.")

    np.random.seed(random_state)
    indices = np.random.choice(n_samples, size=max_samples, replace=False)

    if isinstance(X, pd.DataFrame):
        X_sub = X.iloc[indices]
    else:
        X_sub = X[indices]

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_sub = y.iloc[indices]
    else:
        y_sub = y[indices]

    return X_sub, y_sub
