import os, sys
import re
import numpy as np
from collections import defaultdict
from typing import Union, List, Set

import pandas as pd
import polars as pl
from tabulate import tabulate

from .utils_bio import (
    normalize_strand
)

from .utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display, 
    display_dataframe_in_chunks
)

from .visual_analyzer import (
    create_error_bigwig
)

from .model_utils import (
    aggregate_shap_values,
    get_unique_labels, 
    # bar_chart_freq, 
    compare_feature_rankings,
    compare_feature_importance_ranks,
    bar_chart_local_feature_importance,
    build_token_frequency_comparison, 
    subsample_data
)

from meta_spliceai.mllib import ModelTracker
from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.analyzer import Analyzer

from tqdm import tqdm

# Refactor to utils_plot
import matplotlib.pyplot as plt 
import seaborn as sns

from meta_spliceai.mllib.model_trainer import (
    mini_ml_pipeline,
    get_dummies_and_verify, 
    to_xy, 
)

from .performance_analyzer import (
    plot_single_roc_curve, 
    plot_single_pr_curve,
    plot_cv_roc_curve,
    plot_cv_pr_curve,
)

from .splice_error_analyzer import (
    ErrorAnalyzer, 
)

from .analysis_utils import (
    analyze_data_labels, 
    classify_features, 
    filter_kmer_features, 
    plot_feature_distributions, 
    plot_feature_distributions_v1,
    plot_pairwise_scatter, 
    plot_shap_beeswarm, 
    plot_shap_global_importance, 
    plot_feature_importance
)

from .visual_analyzer import (
    create_error_bigwig,
    # plot_feature_distributions, 
    # plot_pairwise_scatter, 
    # plot_shap_beeswarm
)

from .sequence_featurizer import (
    display_feature_set
)

# ML Libraries
from meta_spliceai.mllib.plot_performance_curve import (
    plot_roc_curve_cv,  # plot_roc_curve_cv(model, X, y, n_folds=10, figsize=None, **kargs)
    plot_prc_curve_cv   # plot_prc_curve_cv(model, X, y, n_folds=10, figsize=None, **kargs)
)

import xgboost as xgb
# from xgboost import XGBClassifier
# from sklearn.base import ClassifierMixin

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Run the following to install the required packages:
# - pip install --upgrade xgboost
# - pip install --upgrade scikit-learn


def train_error_classifier(
    pred_type='FP', 
    remove_strong_predictors=False, 
    strong_predictors=None, 
    splice_type=None,              # <- New param for "donor" or "acceptor" or None
    **kargs
): 
    """
    Train an XGBoost-based error classifier for splice-site errors (e.g. FP vs TP),
    using a featurized dataset that may include k-mer features, gene/transcript-level
    features, and SpliceAI prediction scores.

    Parameters
    ----------
    pred_type : str
        Which error analysis type to handle, e.g. "FP" or "FN" vs. "TP".
    remove_strong_predictors : bool
        If True, we exclude known "strong" features (e.g. "score") that might overshadow
        the motif-based features. 
    strong_predictors : list of str or None
        List of column names to drop if remove_strong_predictors is True. 
        If None, defaults to ["score"].

    splice_type : str or None
        If "donor", subset data where df["splice_type"]=="donor".
        If "acceptor", subset data where df["splice_type"]=="acceptor".
        If None, use all data (both donor & acceptor).

    **kargs : 
        - col_label : str, default 'label'
        - n_splits : int, default 5
        - top_k : int, default 20
        - output_dir : str, directory for saving results
        - Additional parameters for XGBoost or pipeline

    Returns
    -------
    model : XGBoost model
    result_set : dict
        Contains various outputs from xgboost_pipeline, such as feature_importance, etc.

    Memo: 
    ----
    1. Load pre-computed featurized dataset
       
       Run python -m splice_engine.splice_error_analyzer to generate the featurized dataset
    """
    # from .analysis_utils import filter_kmer_features
    # from .model_utils import get_unique_labels

    verbose = kargs.get('verbose', 1)
    col_label = kargs.get('col_label', 'label')

    n_splits = kargs.get('n_splits', 5)
    top_k = kargs.get('top_k', 20)
    top_k_motifs = kargs.get('top_k_motifs', 20)
    error_label = kargs.get("error_label", pred_type)
    correct_label = kargs.get("correct_label", "TP")
    model_type = kargs.get('model_type', 'xgboost')

    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')

    # By default, assume "score" is the strong predictor we might want to exclude
    if strong_predictors is None:
        strong_predictors = ["score"]  # Could add others, e.g. "spliceai_prob"

    # 1) Load pre-computed featurized dataset [1]
    df_trainset = mefd.load_featurized_dataset(
        aggregated=True, 
        error_label=error_label, 
        correct_label=correct_label,
        splice_type=splice_type
    )
    
    if verbose > 0: 
        print_emphasized(f"[action] Training error classifier that compares {error_label} vs. {correct_label} ...")
        print_with_indent(f"Training set: {df_trainset.shape}", indent_level=1)
        print_with_indent(f"Columns: {display_feature_set(df_trainset, max_kmers=100)}", indent_level=1)
        analysis_result = \
            analyze_data_labels(df_trainset, label_col=col_label, verbose=verbose, handle_missing=None)

    # Convert polars DF to pandas if needed
    if isinstance(df_trainset, pl.DataFrame):
        df_trainset = df_trainset.to_pandas()

    # 1a) Optionally remove strong predictors if requested
    if remove_strong_predictors:
        for col_to_drop in strong_predictors:
            if col_to_drop in df_trainset.columns:
                df_trainset.drop(columns=col_to_drop, inplace=True, errors='ignore')
                if verbose > 0:
                    print_with_indent(f"[info] Dropped strong predictor: {col_to_drop}", indent_level=1)
            else:
                if verbose > 0:
                    print_with_indent(f"[warning] Column '{col_to_drop}' not found in the dataset. Skipping.", indent_level=1)

    # 2) Convert DF to X, y
    X, y = to_xy(df_trainset, dummify=True, verbose=1)

    print_with_indent("\nFeatures (X):", indent_level=1)
    display_dataframe_in_chunks(filter_kmer_features(X, keep_kmers=20).head())
    print_with_indent(f"Columns: {display_feature_set(X, max_kmers=100)}", indent_level=1)
    print_with_indent("\nLabels (y):", indent_level=1)
    # print(y.head())
    print_with_indent(f"Unique labels: {get_unique_labels(y)}", indent_level=1) 

    # 3) XGBoost pipeline
    output_dir = kargs.get('output_dir', ErrorAnalyzer.analysis_dir)  # depends on `experiment`
    os.makedirs(output_dir, exist_ok=True)

    # Prepare output subject => incorporate splice_type if provided
    if splice_type is None:
        st_str = "any"
    else:
        st_str = splice_type.lower()

    if st_str in ("donor", "acceptor", "neither"):
        print_with_indent(f"Adjusting the subject name to include splice_type={splice_type} ...", indent_level=1)
        subject = kargs.get('subject', f"{st_str}_{error_label.lower()}_vs_{correct_label.lower()}")
        print_with_indent(f"New subject name: {subject}", indent_level=2)

        # print_with_indent(f"Adjusting the output directory to include splice_type={splice_type} ...", indent_level=1)
        # output_dir = os.path.join(output_dir, st_str)
        # os.makedirs(output_dir, exist_ok=True)
        # print_with_indent(f"New output directory: {output_dir}", indent_level=2)
    else:
        subject = kargs.get('subject', f"{error_label.lower()}_vs_{correct_label.lower()}")

    if remove_strong_predictors:
        subject = f"{subject}-strong-vars-filtered"

    print('[test] Shape of X, y prior to xgboost pipeline', X.shape, y.shape)
    model, result_set = \
        xgboost_pipeline(
            X, y,
            output_dir=output_dir, 
            top_k=top_k, 
            top_k_motifs=top_k_motifs,
            n_splits=n_splits, 
            subject=subject)

    print_section_separator()
    print_emphasized(f"[result] Completed training error classifier for {pred_type}.")
    # Outputs 
    # feature_importance = result_set.get('feature_importance', None)
    # top_features = result_set.get('top_features', None)
    # top_motifs = result_set.get('top_motifs', None)

    # Save model 
    # tracker = ModelTracker(output_dir=output_dir)

    # 4) Additional follow-up analysis
    print("[action] Running follow-up analyses on feature importance  ...")
    # top_k = kargs.get('top_k', 20)

    # Get top features from each method
    shap_features = result_set['importance_df_shap'].head(top_k)['feature'].tolist()
    xgb_features = result_set['importance_df_xgboost'].head(top_k)['feature'].tolist()
    sig_features = result_set['importance_df_hypotest'].head(top_k)['feature'].tolist()

    # Keep features that appear in at least two rankings
    final_features = list(set(shap_features) & set(xgb_features) | set(sig_features))
    print_emphasized(f"[info] Final consolidated feature set: {final_features}")

    importance_df_xgboost = result_set['importance_df_xgboost']
    importance_df_shap = result_set['importance_df_shap']
    importance_df_hypotest = result_set['importance_df_hypotest']

    # Test 
    print_with_indent(f"[test] Data shape XGBoost importance: {importance_df_xgboost.shape}", indent_level=1)
    print_with_indent(f"[test] Data shape SHAP importance: {importance_df_shap.shape}", indent_level=1)
    print_with_indent(f"[test] Data shape HypoTest importance: {importance_df_hypotest.shape}", indent_level=1)

    print_emphasized("[action] Feature importance comparison #1: ")
    overlap, correlation = compare_feature_rankings(
        [  
            importance_df_xgboost,   # from xgboost
            importance_df_shap,      # from shap
            importance_df_hypotest   # from hypothesis testing
        ],
        method_names=["XGBoost", "SHAP", "HypoTest"],
        top_k=top_k
    )

    # Investigating Differences in Feature Values for FPs vs. TPs
    
    print_emphasized("[action] Feature importance comparison #2: ")
    output_path = os.path.join(output_dir, f"{subject}-feature-importance-comparison.pdf")

    compare_feature_importance_ranks(
        importance_dfs={
            'XGBoost': importance_df_xgboost,
            'SHAP': importance_df_shap,
            'HypoTest': importance_df_hypotest
        }, 
        top_k=top_k,
        primary_method='SHAP',
        output_path=output_path, 
        verbose=1, 
        # save=True,
        # plot_style="percentile_rank"
    )

    print_emphasized("[action] Comparing feature value distributions ...")
    top_features = importance_df_shap["feature"].head(top_k).tolist()
    summary_df = summarize_feature_distributions(
        X, y, top_features, pos_label=1, neg_label=0, n_bins=30)
    display_dataframe_in_chunks(summary_df)
    # This yields a table showing how the means/medians differ between FPs and TPs. 
    # Todo: 
    # You might want to incorporate 95% confidence intervals or boxplots to see where 
    # the difference is largest.

    # Plot feature distributions
    print_emphasized("[action] Plotting feature distributions ...")
    output_path = os.path.join(output_dir, f"{subject}-feature-distributions.pdf")
    plot_feature_distributions(
        X, y, 
        top_features,  # plot top N features based on SHAP importance
        # Re-label classes in final plots
        label_text_0=correct_label,   # Instead of "Class 0"
        label_text_1=error_label,   # Instead of "Class 1"
        plot_type="box", 
        n_cols=3, 
        figsize=None, 
        title="Feature Distributions",
        output_path=output_path, 
        show_plot=False, 
        verbose=1, 
        top_k_motifs=top_k_motifs,        # New: Only plot top N motif features (k-mers) if provided
        kmer_pattern=r'^\d+mer_.*',  # Regex to identify motif features
        use_swarm_for_motifs=True
    )

    # Plot feature distributions v1
    print_emphasized("[action] Plotting feature distributions v1 ...")
    output_path = os.path.join(output_dir, f"{subject}-feature-distributions-v1.pdf")
    plot_feature_distributions_v1(
        X, y, 
        top_features,  # plot top N features based on SHAP importance
        # Re-label classes in final plots
        label_text_0=correct_label,   # Instead of "Class 0"
        label_text_1=error_label,   # Instead of "Class 1"
        plot_type="box", 
        n_cols=2, 
        figsize=None, 
        title="Feature Distributions",
        output_path=output_path, 
        show_plot=False, 
        verbose=1, 
        top_k_motifs=top_k_motifs,        # New: Only plot top N motif features (k-mers) if provided
        kmer_pattern=r'^\d+mer_.*',  # Regex to identify motif features
        use_swarm_for_motifs=True, 
        show_feature_stats=True,   # New parameter: show mean/median for each class
        annotate_sparse=True      # New parameter: add annotations for sparse features
    )
  
    # Plot pairwise scatter
    # output_path = os.path.join(output_dir, f"{subject}-pairwise-scatter.pdf")
    # plot_pairwise_scatter(
    #     X, y, 
    #     top_features[:5], 
    #     pos_label=1, 
    #     neg_label=0, 
    #     figsize=(8,6),
    #     output_path=output_path,
    #     show_plot=False,
    #     verbose=1
    # )

    # ------------------------------------------------------------
    # Meta-Analysis
    # ------------------------------------------------------------
    print_section_separator()
    local_top_k = kargs.get('local_top_k', 25)
    global_top_k = kargs.get('global_top_k', 50)
    plot_top_k = kargs.get('plot_top_k', local_top_k)
     
    print_emphasized("[action] Running SHAP meta-analysis ...")

    # 1) We have:
    #    - model: your trained XGBoost model
    #    - X_explain, y_explain
    #    - output_dir: somewhere to save outputs
    #    - error_label=1, correct_label=0
    #    - local_top_k=10, global_top_k=20

    top_global_features, global_shap_df, local_freq_df = \
        shap_analysis_with_local_agg_plots(
            model=model,
            X_test=X,
            y_test=y,
            output_dir=output_dir,
            # error_label=1,           # e.g. “error” = 1, already encoded in y_explain
            # correct_label=0,         # e.g. “correct”= 0, already encoded in y_explain
            local_top_k=local_top_k,
            global_top_k=global_top_k,
            plot_top_k=plot_top_k,
            subject=subject,
            suffix="-meta",   # suffix for output files; use "-meta" for meta-analysis
            verbose=1,
            return_all=True  # we want global_shap_df & local_freq_df
        )

    # global_shap_df has the following columns:
    # [ "feature",
    #   "importance_score"  # mean(|SHAP|) across all samples
    # ]

    # local_freq_df has the following columns:
    # [ "feature", 
    #   "freq_error",    # fraction of error samples in which feature was local-top-k
    #   "freq_correct",  # fraction of correct samples in which feature was local-top-k
    #   "freq_diff",     # large +ve => more prevalent in errors
    #   "count_error", 
    #   "count_correct"
    # ]
    # - If freq_diff is positive (and large), that feature’s local top-k usage is more prevalent 
    #   in error samples than correct, suggesting it might be a “decoy” or mis-leading feature.

    # A. Per-gene bar chart (Todo: Implement)
    
    # B. Whole-Dataset “Local-Frequency” as a Quick Bar Plot
    # - This shows the fraction of error samples in which each feature was in the local top-k.
    # - This is a quick way to check if any features are over-represented in errors.
    # - If a feature has a high freq_error, it might be a “decoy” or mis-leading feature.

    # 1) Inspect the "local_freq_df"
    #    columns => [feature, freq_error, freq_correct, freq_diff, count_error, count_correct]

    # bar_chart_freq(...) expects a list of tuples like:
    # [
    #   (token/feature, error_freq, correct_freq, diff),
    #   ...
    # ]

    # 2) Build the list of (feature, error_freq, correct_freq, diff) 
    local_freq_list = []
    for row in local_freq_df.itertuples(index=False):
        local_freq_list.append((row.feature, row.freq_error, row.freq_correct, row.freq_diff))

    # Sort them by abs diff or however you like
    local_freq_list.sort(key=lambda x: abs(x[3]), reverse=True)

    # 3) Plot “global frequency bar chart” from local aggregator
    suffix = "-meta"
    output_file = f"{subject}-local-shap-frequency-comparison{suffix}.pdf"
    output_path = os.path.join(output_dir, output_file)
    bar_chart_local_feature_importance(
        local_freq_list,
        error_label=error_label,
        correct_label=correct_label,
        top_n=plot_top_k,
        output_path=output_path, 
        token_type=None
    )

    return model, result_set


######################################################


def train_error_classifier_incremental(
    batch_data_loader,
    error_label="FP",
    correct_label="TP",
    splice_type=None,
    n_batches=10,
    random_state=42,
    **kwargs
):
    """
    Train an incremental XGBoost error classifier using random subsets (batches) of genes.

    Parameters:
    - batch_data_loader: callable function or generator
        Function or generator that yields batches (X_batch, y_batch).
    - error_label, correct_label: Labels defining positive and negative classes.
    - splice_type: Optionally filter by donor/acceptor.
    - n_batches: Number of batches/subsets.
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    model = xgb.XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=random_state,
        verbosity=kwargs.get('verbose', 1)
    )

    for batch_idx, (X_batch, y_batch) in enumerate(batch_data_loader(n_batches=n_batches, 
                                                                     error_label=error_label, 
                                                                     correct_label=correct_label,
                                                                     splice_type=splice_type)):
        if batch_idx == 0:
            # Initial training
            model.fit(X_batch, y_batch)
        else:
            # Incremental training (updating the existing model)
            model.fit(X_batch, y_batch, xgb_model=model.get_booster())
        
        # Optional: Evaluate after each batch
        y_pred_batch = model.predict(X_batch)
        batch_auc = roc_auc_score(y_batch, model.predict_proba(X_batch)[:, 1])
        print(f"Batch {batch+1}/{n_batches} | ROC AUC: {batch_auc:.4f}")

    return model


def batch_data_loader(full_data, batch_size=500, error_label='FP', correct_label='TP', random_state=42):
    """
    Generator function yielding batches from the full data.
    
    Parameters:
    - full_data (pd.DataFrame): entire dataset
    - batch_size (int): Number of genes per batch.
    
    Yields:
    -------
    X_batch, y_batch : DataFrame, Series
    """
    np.random.seed(random_state)
    gene_ids = full_data['gene_id'].unique()
    np.random.shuffle(gene_ids)

    num_batches = int(np.ceil(len(gene_ids) / batch_size))

    for i in range(num_batches):
        batch_genes = gene_ids[i * batch_size:(i + 1) * batch_size]
        batch_df = full_data[full_data['gene_id'].isin(batch_genes)]

        # Filter for error and correct labels
        batch_df = batch_df[batch_df['pred_type'].isin([error_label, correct_label])]
        
        # Convert to X, y format
        X_batch, y_batch = to_xy(batch_df, label_col='pred_type', dummify=True, verbose=0)
        
        # Map labels to binary
        y_batch_binary = y_batch.apply(lambda x: 1 if x == error_label else 0)

        yield X_batch, y_batch_binary


def train_error_classifier_ensemble(
    n_subsamples=3,
    sample_fraction=0.5,
    random_seed=42,
    ensemble_method="average",
    feature_importance_aggregation="mean",
    **train_classifier_kwargs
):
    """
    Train multiple XGBoost models using train_error_classifier() on subsampled data,
    then combine or ensemble the models.

    Parameters
    ----------
    n_subsamples : int
        Number of subsets (subsamples) to train on.
    sample_fraction : float
        Fraction of the dataset to retain for each subsample (e.g. 0.5 => 50%).
        This is used if your dataset is large, or if you want bagging-like approach.
    random_seed : int
        Random seed for reproducibility.
    ensemble_method : str
        How to combine the predictions from each sub-model.
        Options might include "average" or "vote" for classification.
    feature_importance_aggregation : str
        How to consolidate or unify feature importance across the multiple models:
         - "mean" => average importance
         - "median" => median importance
         - "union" => keep all features across models
         - "intersection" => only keep features that appear in every model
    **train_classifier_kwargs :
        Any additional kwargs to pass into train_error_classifier().

    Returns
    -------
    ensemble_models : list
        List of fitted XGBoost models from each subsample.
    aggregated_result : dict
        A dictionary that can store aggregated results, such as:
         - "ensemble_method": str
         - "feature_importance": pd.DataFrame or None
         - "subsample_results": [result_set_1, result_set_2, ...]
         - "meta_predictions_func": a function or partial that can be used for ensemble predictions

    Memo: 
    ----
    This function is a wrapper around train_error_classifier() that trains multiple models on subsampled data.
    It then combines the models using an ensemble method and returns the aggregated results.

    Handles Large Data: Each run only needs to load a fraction (e.g. 30% or 50%) of the full dataset.
    - Improves Generalization: Using multiple subsets in a “bagging” style approach typically reduces variance.
    - Interpretability: Summarizing feature importances across runs provides a more robust sense of which features remain consistently influential.
    """
    import random
    # import numpy as np
    
    # For reproducibility across multiple subsamples
    rng = np.random.default_rng(random_seed)
    
    # 1. Load the *full* dataset if it’s not too big or partially load it:
    #    -> We rely on train_error_classifier() to do the loading internally.
    #    BUT we want to override the loaded data with a subsample if it’s still huge.
    #    One approach: we let train_error_classifier() load the entire dataset, then 
    #    we do an in-memory subsample. Alternatively, we might do the sampling at 
    #    the file-IO level if your loading function supports it. 
    # 
    # For clarity, let's do the in-memory approach:
    # We'll call train_error_classifier() multiple times, each time instructing it 
    # to do a random subsample. We'll define a custom param e.g. "subsample_fraction".

    ensemble_models = []
    subsample_results = []
    
    for i in range(n_subsamples):
        print_emphasized(f"\n=== Training sub-model #{i+1}/{n_subsamples} ===")

        # We pass a new random_seed each time or pass sample_fraction so that 
        # train_error_classifier() or mefd.load_featurized_dataset() can do the subsampling inside.
        local_seed = random_seed + i
        local_kwargs = dict(train_classifier_kwargs)
        local_kwargs["random_seed"] = local_seed
        local_kwargs["subsample_fraction"] = sample_fraction
        
        # 2. Train the model on the subsample
        model_i, result_i = train_error_classifier(**local_kwargs)
        
        # collect results
        ensemble_models.append(model_i)
        subsample_results.append(result_i)

    # 3. Combine or average predictions => define a helper function:
    def ensemble_predict_proba(X):
        """
        Return the averaged probability (class=1) across the n_subsamples.
        """
        import numpy as np
        proba_list = []
        for mdl in ensemble_models:
            # XGBoost API => model.predict_proba(X) => shape (n, 2) for binary
            p = mdl.predict_proba(X)[:, 1]  # probability for class=1
            proba_list.append(p)
        # average (or could be e.g. median)
        # shape (n, )
        proba_avg = np.mean(np.array(proba_list), axis=0)
        return proba_avg

    # 4. Aggregate feature importance across all sub-models
    #    e.g., we want a single data structure that merges XGBoost + SHAP results from each run
    #    For demonstration, let’s do a straightforward “mean” of each feature’s importance 
    #    across submodels for XGBoost. Similarly for SHAP if present.

    # The result_set from xgboost_pipeline typically has:
    #   result_set["importance_df_xgboost"] => DataFrame with columns ["feature", "importance_score"]
    #   result_set["importance_df_shap"] => DataFrame with columns ["feature", "importance_score"]
    #   result_set["importance_df_hypotest"] => DataFrame with columns [ "feature", "importance_score"]
    # We can do something like “union” or “intersection” or “mean”.

    aggregated_importance_xgb = None
    aggregated_importance_shap = None
    aggregated_importance_hypo = None

    # Helper to unify + average
    def unify_and_aggregate(df_list, method="mean"):
        """
        Given a list of importance DataFrames => unify them by "feature" 
        and compute a single aggregated importance_score.
        Each DF must have columns ["feature", "importance_score"].
        """
        import pandas as pd
        if not df_list:
            return None
        # concat
        combined = pd.concat(df_list, axis=0, ignore_index=True)
        # group by feature
        grouped = combined.groupby("feature")["importance_score"]
        if method=="mean":
            agg_series = grouped.mean().sort_values(ascending=False)
        elif method=="median":
            agg_series = grouped.median().sort_values(ascending=False)
        else:
            # fallback => mean
            agg_series = grouped.mean().sort_values(ascending=False)
        # build DataFrame
        df_agg = agg_series.reset_index()
        df_agg.columns = ["feature","importance_score"]
        return df_agg

    # gather importance from each model
    xgb_list = []
    shap_list= []
    hypo_list= []

    for i, rset in enumerate(subsample_results):
        if "importance_df_xgboost" in rset:
            xgb_list.append(rset["importance_df_xgboost"])
        if "importance_df_shap" in rset:
            shap_list.append(rset["importance_df_shap"])
        if "importance_df_hypotest" in rset:
            hypo_list.append(rset["importance_df_hypotest"])

    aggregated_importance_xgb = unify_and_aggregate(xgb_list, method=feature_importance_aggregation)
    aggregated_importance_shap= unify_and_aggregate(shap_list, method=feature_importance_aggregation)
    aggregated_importance_hypo= unify_and_aggregate(hypo_list, method=feature_importance_aggregation)

    # finalize
    aggregated_result = {
        "ensemble_method": ensemble_method,
        "feature_importance_aggregation": feature_importance_aggregation,
        "ensemble_models": ensemble_models,
        "subsample_results": subsample_results,
        # final aggregated importance
        "xgb_importance": aggregated_importance_xgb,
        "shap_importance": aggregated_importance_shap,
        "hypotest_importance": aggregated_importance_hypo,
    }
    
    def predict_ensemble_proba(X):
        return ensemble_predict_proba(X)
    
    aggregated_result["predict_ensemble_proba"] = predict_ensemble_proba
    
    return ensemble_models, aggregated_result


######################################################


def summarize_feature_distributions(
    X, y, feature_list, pos_label=1, neg_label=0, n_bins=30, 
    output_dir=None, save=False
):
    """
    For each feature in feature_list, show distribution or stats for
    the pos_label vs. neg_label classes. Optionally plot histograms/boxplots.
    """
    pos_df = X.loc[y==pos_label, feature_list]
    neg_df = X.loc[y==neg_label, feature_list]

    summary = []
    for feat in feature_list:
        pos_vals = pos_df[feat].dropna()
        neg_vals = neg_df[feat].dropna()
        # compute mean, median, etc.
        pos_mean, neg_mean = pos_vals.mean(), neg_vals.mean()
        pos_median, neg_median = pos_vals.median(), neg_vals.median()
        row_info = {
            "feature": feat,
            "pos_mean": pos_mean, "neg_mean": neg_mean,
            "pos_median": pos_median, "neg_median": neg_median
        }
        summary.append(row_info)

        # optionally plot a small histogram or boxplot
        # or store data for external plotting
        # if save: 
        #   ... code for plt.figure() ...
        #   pos_vals.plot(kind='hist', alpha=0.5, label='FP', bins=n_bins)
        #   neg_vals.plot(kind='hist', alpha=0.5, label='TP', bins=n_bins)
        #   ...
    summary_df = pd.DataFrame(summary).sort_values(by="pos_mean", ascending=False)
    return summary_df



######################################################
# from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# def plot_single_roc_curve(y_test, y_proba, subject="fp_vs_tp", output_path=None, save=True):
#     raise NotImplementedError("Refactored to performance_analyzer.py")


# def plot_single_pr_curve(y_test, y_proba, subject="fp_vs_tp", output_path=None, save=True):
#     raise NotImplementedError("Refactored to performance_analyzer.py")

    
######################################################

def format_subject_for_title(subject):
    """
    Format the subject parameter value to be a valid part of the plot title.

    Parameters:
    - subject (str): The subject parameter value (e.g., "fp_vs_tp").

    Returns:
    - str: The formatted subject for the plot title (e.g., "FP vs TP").
    """
    return subject.replace("_", " ").title()


def xgboost_pipeline(
    X, y, top_k=15, n_splits=5, random_state=42, 
    output_dir=None, save=True, subject="fp_vs_tp", importance_type="weight", **kargs
):
    """
    Train and evaluate an XGBoost model, generate performance metrics, and visualize feature importance.

    Parameters:
    ----------
    X : pd.DataFrame
        Feature DataFrame.
    y : pd.Series
        Label column (binary).
    top_k : int
        Number of top features to identify.
    n_splits : int
        Number of cross-validation folds.

    use_full_data_for_explanation:
        determines if you use the entire dataset or just the test set for your SHAP-based feature importance.

    random_state : int
        Seed for reproducibility.

    Returns:
    -------
    None

    Memo: 
    ----
    This workflow performs the following steps:

    1. Data Preparation:
    - A single train/test split is performed on the dataset.
    - Cross-validation predictions are generated using cross_val_predict to 
       get out-of-fold predictions for the full dataset (X, y).

    2. Model Training and Evaluation:
    - The model is trained on the training set.
    - The model is evaluated on the test set.
    - ROC and PR curves are generated to assess the model's performance.

    3. Feature Importance Analysis:
    - XGBoost-based feature importance is computed to identify the most important features.
    - SHAP-based feature importance is computed to provide a more detailed explanation of feature contributions.
    - A motif subset analysis is conducted to understand the impact of specific motifs on the model's predictions.
    - Hypothesis testing is performed to validate the significance of the identified features.

    4. Results Saving:
    - The results of the analyses, including feature importance scores and hypothesis test results, are saved to the specified output directory.

    This workflow aims to provide a comprehensive analysis of the model's performance and the importance of different features in the dataset.

    """
    # import xgboost as xgb
    # from .model_utils import (
    #     bar_chart_freq, 
    #     subsample_data
    # )

    verbose = kargs.get('verbose', 1)
    use_full_data_for_explanation = kargs.get('use_full_data_for_explanation', True) 
    test_size = kargs.get('test_size', 0.2)
    plot_format = kargs.get('plot_format', 'pdf')
    model_type = 'xgboost'

    result_set = {}  # A dictionary to store outputs from each step. 

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Ensure X, y indices are reset to avoid indexing mismatches downstream
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.reset_index(drop=True)

    # Split data
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    # A standard 80/20 split, by default, with stratification.

    # Initialize XGBoost classifier
    print_emphasized(f"[info] Training XGBoost model for subject={subject} ...")
    model = xgb.XGBClassifier(
        # use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=random_state, 
        verbosity=verbose
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_proba_cv = cross_val_predict(model, X, y, cv=skf, method="predict_proba")
    y_pred_cv = np.argmax(y_proba_cv, axis=1)

    # Train final model on full training set
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f1_cv = f1_score(y, y_pred_cv)
    roc_auc = roc_auc_score(y_test, y_proba)
    specificity = recall_score(y_test, y_pred, pos_label=0)  # Specificity
    fpr = 1 - specificity  # False Positive Rate
    fnr = 1 - recall

    print_emphasized(f"[info] Performance Metrics (subject={subject}):")
    print_with_indent(f"Precision: {precision:.4f}", indent_level=1)
    print_with_indent(f"Recall: {recall:.4f}", indent_level=1)
    print_with_indent(f"Specificity: {specificity:.4f}", indent_level=1)
    print_with_indent(f"F1-Score: {f1:.4f}", indent_level=1)
    print_with_indent(f"F1-Score (CV): {f1_cv:.4f}", indent_level=1)
    print_with_indent(f"ROC AUC: {roc_auc:.4f}", indent_level=1)
    print_with_indent(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}", indent_level=1)

    ######################################################
    # Plot ROC and PR curves
    ext = plot_format
    output_file = kargs.get('output_file', f"{subject}-{model_type}-roc.{ext}") 
    output_path = os.path.join(output_dir, output_file)

    plot_single_roc_curve(
        y_test, y_proba, 
        subject=subject, 
        output_path=output_path, 
        save=save)

    output_file = kargs.get('output_file', f"{subject}-{model_type}-prc.{ext}") 
    output_path = os.path.join(output_dir, output_file)

    plot_single_pr_curve(
        y_test, y_proba, 
        subject=subject, 
        output_path=output_path, save=save)

    ######################################################
    formatted_subject = format_subject_for_title(subject)

    # We do a new model instance or clone (since we'll refit per fold)
    # or define model outside the loop
    xgboost_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

    print_emphasized("[info] Generating CV-driven ROCAUC plot, showing only mean curve in the output ...")
    plot_cv_roc_curve(
        xgboost_model,
        X, y,
        cv_splits=n_splits,
        random_state=42,
        title=f"{formatted_subject} Cross-Validated ROC",
        output_path=os.path.join(output_dir, f"{subject}-xgboost-ROC-CV.{ext}"),
        show_std=True,
        plot_folds=False
    )
    # This only shows mean ROC curve with std. dev. shading

    print_emphasized("[info] Generating CV-driven PRC plot, showing only mean curve in the output...")
    plot_cv_pr_curve(
        xgboost_model,
        X, y,
        cv_splits=n_splits,
        random_state=42,
        title=f"{formatted_subject} Cross-Validated PRC",
        output_path=os.path.join(output_dir, f"{subject}-xgboost-PRC-CV.{ext}"),
        show_std=True, 
        plot_folds=False
    )
    # This only shows mean PR curve with std. dev. shading

    print_emphasized("[info] Generating CV-driven ROCAUC plot, showing all curves in the output ...")
    ext = 'pdf'
    fig_text = f"{formatted_subject} Cross-Validated ROC"
    title_text = f"{formatted_subject} Error Analysis Model"
    # plot_roc_curve_cv(
    #     xgboost_model, 
    #     X, y, 
    #     n_folds=n_splits, 
    #     standardize=False,
    #     figsize=(10, 10), 
    #     output_dir=output_dir, 
    #     output_file=f"{subject}-xgboost-ROC-CV-{n_splits}folds-v2.{ext}",
    #     fig_text=fig_text, 
    #     title=title_text,
    #     save=True
    # )
    plot_cv_roc_curve(
        xgboost_model,
        X, y,
        cv_splits=n_splits,
        random_state=42,
        title=title_text,
        output_path=os.path.join(output_dir, f"{subject}-xgboost-ROC-CV-{n_splits}folds.{ext}"),
        show_std=True,
        plot_folds=True
    )

    print_emphasized("[info] Generating CV-driven PRC plot, showing all curves in the output ...")
    fig_text = f"{formatted_subject} Cross-Validated PRC"
    title_text = f"{formatted_subject} Error Analysis Model"
    # plot_prc_curve_cv(
    #     xgboost_model, 
    #     X, y, 
    #     n_folds=n_splits, 
    #     standardize=False,
    #     figsize=(10, 10), 
    #     output_dir=output_dir, 
    #     output_file=f"{subject}-xgboost-PRC-CV-{n_splits}folds-v2.{ext}",
    #     fig_text=fig_text, 
    #     title=title_text,
    #     save=True
    # )  
    plot_cv_pr_curve(
        xgboost_model,
        X, y,
        cv_splits=n_splits,
        random_state=42,
        title=title_text,
        output_path=os.path.join(output_dir, f"{subject}-xgboost-PRC-CV-{n_splits}folds.{ext}"),
        show_std=True, 
        plot_folds=True  # <--- This shows all the PR curves in CV
    )  


    ######################################################

    # Retrain the model on the entire datase
    # - A common practice to get the “best possible final model” now that we’ve validated it.
    print_emphasized("[info] Retraining model on the full dataset ...")
    model.fit(X, y)

    ######################################################
    # Explainability

    max_shap_samples = kargs.get('max_shap_samples', 25000)  # adjust as needed based on memory constraints

    if use_full_data_for_explanation:
        print("[info] Using full dataset for SHAP analysis with subsampling...")
        X_explain, y_explain = subsample_data(X, y, max_samples=max_shap_samples)
    else:
        X_explain, y_explain = X_test, y_test

    # Todo: Subsample for SHAP analysis
    # SHAP can be expensive for large datasets   

    # Feature Importance
    print_emphasized("[info] Analyzing feature importance and generating related plots ...")
    print_with_indent(f"[info] SHAP analysis will use {X_explain.shape[0]} samples.", indent_level=1)
    
    importance_type = 'weight'
    output_file = f"{subject}-xgboost-importance-{importance_type}-barplot.pdf"
    output_path = os.path.join(output_dir, output_file)
    print_with_indent(f"Feature importance via XGBoost (mode={importance_type})...", indent_level=1)

    plt.clf()  # Ensure clean figure before plotting
    top_features_xgboost, importance_df_xgboost = \
        quantify_feature_importance_via_xgboost(
            model, 
            X_explain, 
            output_path=output_path, 
            top_k=top_k, 
            importance_type=importance_type,
            # color_map="viridis", 
            save=True)
    # result_set['top_features_xgboost'] = top_features_xgboost
    # result_set['importance_df_xgboost'] = importance_df_xgboost

    print_emphasized(f"Top {top_k} important features (XGBoost)")
    display_dataframe_in_chunks(top_features_xgboost, num_rows=top_k)

    # --------------------------------------------------------------------

    importance_type = "total_gain"
    output_file = f"{subject}-xgboost-importance-{importance_type}-barplot.pdf"
    output_path = os.path.join(output_dir, output_file)
    print_with_indent(f"Feature importance via XGBoost (mode={importance_type})...", indent_level=1)

    plt.clf()  # Ensure clean figure before plotting
    top_features_xgboost, importance_df_xgboost = \
        quantify_feature_importance_via_xgboost(
            model, 
            X_explain, 
            output_path=output_path, 
            top_k=top_k, 
            importance_type=importance_type,
            # color_map="viridis", 
            save=True)

    result_set['top_features_xgboost'] = top_features_xgboost
    result_set['importance_df_xgboost'] = importance_df_xgboost

    print_emphasized(f"Top {top_k} important features (XGBoost)")
    display_dataframe_in_chunks(top_features_xgboost, num_rows=top_k)

    # --------------------------------------------------------------------
    print_section_separator()

    # SHAP Analysis v1 (global aggregator)
    print_with_indent(f"Feature importance via SHAP", indent_level=1)
    
    plt.clf()  # Ensure clean figure before plotting
    top_features, top_features_shap, importance_df_shap = \
        quantify_feature_importance_via_shap_analysis(
            model, X_explain, 
            output_dir=output_dir, 
            subject=subject,
            top_k=top_k,  # Number of top features to display
            top_n_features=top_k,  # Number of top important features to select
            return_scores=True,  # Set to True to return feature-importance DataFrames
            verbose=1)
    # NOTE: 
    # top_features: The list of top_k feature names.
    # top_features_shap: DataFrame of top_k.
    # feature_importance_shap: DataFrame of all features sorted by SHAP importance.

    print_emphasized(f"Top {top_k} important features (mean abs SHAP values)")
    display_dataframe_in_chunks(top_features_shap, num_rows=top_k)
    result_set['top_features_shap'] = result_set['top_features'] = top_features_shap
    result_set['importance_df_shap'] = importance_df_shap  # Full list

    # Rank motif-specific features based on SHAP values
    top_k_motifs = kargs.get('top_k_motifs', top_k)

    plt.clf()  # Ensure clean figure before plotting
    top_motif_df, full_motif_df = \
        quantify_feature_importance_via_shap_analysis_with_motifs(
            model, X_explain, 
            kmer_pattern=r'^\d+mer_.*', 
            top_n_motifs=top_k_motifs, 
            return_scores=True, 
            return_full_scores=True,
            verbose=1)

    if not top_motif_df.empty:
        print_emphasized(f"Top {top_k_motifs} important motif-specific features")
        display_dataframe_in_chunks(top_motif_df, num_rows=top_k_motifs)
        result_set['top_motifs_by_shap'] = result_set['top_motifs'] = top_motif_df

    # Plot SHAP summary with motifs (combining steps above)
    # plot_and_save_shap_summary_with_motifs(
    #     model, 
    #     X_explain, 
    #     output_dir=output_dir, 
    #     top_k=top_k,  # Number of top features to display
    #     top_n_features=top_k,  # Number of top important features to select
    #     kmer_pattern=r'^\d+mer_.*', 
    #     top_n_motifs=top_k_motifs, 
    #     return_scores=True, 
    #     verbose=1)

    # --------------------------------------------------------------------
    # SHAP Analysis v2 (deferred to the caller: train_error_classifier())

    # --------------------------------------------------------------------
    # Hypothesis Testing
    print_section_separator()

    print_emphasized("Analyzing feature importance via hypothesis testing  ...")

    # Classify features to identify feature columns
    feature_categories = classify_features(X_explain)
    # feature_columns = (
    #     feature_categories['categorical_features'] +
    #     feature_categories['numerical_features'] +
    #     feature_categories['derived_categorical_features'] + 
    # )

    # Run hypothesis tests
    output_file = f"{subject}-{model_type}-hypo-testing-top-shap-features.tsv"
    output_path = os.path.join(output_dir, output_file)

    plt.clf()  # Ensure clean figure before plotting
    print_emphasized(f"Running Hypothesis Testing on top {top_k} important shap-derived features ...")
    results_df, importance_df_hypotest_top_shap = \
        quantify_feature_importance_via_hypothesis_testing(
            X_explain, y_explain, 
            top_features, 
            output_path=output_path, 
            feature_categories=feature_categories, 
            alpha=0.05, 
            verbose=1, 
            save=False  # Saving defer to later in the workflow
        )
    # result_set['importance_df_hypotest_top_shap'] = importance_df_hypotest_top_shap
    print_with_indent(f"Top {top_k} important features (hypothesis testing)", indent_level=1)
    display_dataframe_in_chunks(importance_df_hypotest_top_shap, num_rows=top_k)

    # ------------------------------------------------
    # Run hypothesis tests on full set of features
    output_file = f"{subject}-{model_type}-hypo-testing-full.tsv"
    output_path = os.path.join(output_dir, output_file)

    plt.clf()
    print_emphasized(f"Running Hypothesis Testing on FULL shap-derived features ...")
    df_hypotest, importance_df_hypotest = \
        quantify_feature_importance_via_hypothesis_testing(
            X_explain, y_explain, 
            importance_df_shap,   # Use all features
            output_path=output_path, 
            feature_categories=feature_categories, 
            alpha=0.05, 
            verbose=1, 
            save=False  # Saving defer to later in the workflow
        )
    result_set['importance_df_hypotest'] = importance_df_hypotest
    
    print_with_indent(f"Top {top_k} important features (hypothesis testing)", indent_level=1)
    top_features_hypotest = importance_df_hypotest.head(top_k)
    display_dataframe_in_chunks(top_features_hypotest, num_rows=top_k)

    # Additionally plot the global importance score derived from hypothesis testing
    plot_feature_importance(
        importance_df_hypotest, 
        title="Feature Importance via Hypothesis Testing", 
        output_path=os.path.join(output_dir, f"{subject}-{model_type}-hypo-testing-barplot.pdf"), 
        use_continuous_color=True, 
        top_k=top_k
    )

    # ------------------------------------------------
    # Measure effect sizes
    # ------------------------------------------------
    print_section_separator()
    print_emphasized("Measuring effect sizes ...")

    effect_sizes_df, importance_df_effect_sizes = \
        quantify_feature_importance_via_measuring_effect_sizes(
            X_explain, 
            y_explain, 
            importance_df_shap,  # Use all features
            output_path=output_path, 
            feature_categories=feature_categories, 
            verbose=1, 
            save=False)

    # Plot feature importance based on effect sizes
    plot_feature_importance(
        importance_df_effect_sizes, 
        title="Feature Importance via Effect Sizes", 
        output_path=os.path.join(output_dir, f"{subject}-{model_type}-effect-sizes-barplot.pdf"), 
        use_continuous_color=True, 
        top_k=top_k, 
        rank_by_abs=True  # effect sizes can be both positive and negative
    )

    # ------------------------------------------------
    # Measure mutual information
    # ------------------------------------------------
    mi_df, importance_df_mi = \
        quantify_feature_importance_via_mutual_info(
            X_explain, 
            y_explain,  
            importance_df_shap,  # Use all features
            output_path=output_path, 
            feature_categories=feature_categories,  
            verbose=1, 
            save=False 
        )
    
    # Plot feature importance based on mutual information
    plot_feature_importance(
        importance_df_mi, 
        title="Feature Importance via Mutual Information", 
        output_path=os.path.join(output_dir, f"{subject}-{model_type}-mutual-info-barplot.pdf"), 
        use_continuous_color=True, 
        top_k=top_k, 
        # rank_by_abs=True  # mutual information is always positive
    )

    if save:
        # Todo: Standardize output path (see ErrorAnalyzer)
        print_emphasized(f"[i/o] Saving TOP {top_k} feature importance results ...")

        format = 'tsv'
        sep = '\t' if format == 'tsv' else ','

        # 1 Save top feature importance by shap values to file
        output_file = f"{subject}-{model_type}-importance-shap.{format}"
        output_path = os.path.join(output_dir, output_file)
        top_features_shap.to_csv(output_path, sep=sep, index=False)  # Top features by SHAP values
        print_with_indent(f"[i/o] Top feature importance (shap) saved to: {output_path}", indent_level=1)

        # 1a Save full feature importance by shap values to file
        output_file = f"{subject}-{model_type}-importance-shap-full.{format}"
        output_path = os.path.join(output_dir, output_file)
        importance_df_shap.to_csv(output_path, sep=sep, index=False)  # Full feature importance by SHAP values
        print_with_indent(f"[i/o] Full feature importance (shap) saved to: {output_path}", indent_level=1)

        # 2. Save top feature importance by xgboost values to file
        output_file = f"{subject}-{model_type}-importance-{importance_type}.{format}"
        output_path = os.path.join(output_dir, output_file)
        top_features_xgboost.to_csv(output_path, sep=sep, index=False)  # Top features by XGBoost importance
        print_with_indent(f"[i/o] Top feature importance (xgboost + {importance_type}) saved to: {output_path}", indent_level=1)

        # 2a. Save full feature importance by xgboost values to file
        output_file = f"{subject}-{model_type}-importance-{importance_type}-full.{format}"
        output_path = os.path.join(output_dir, output_file)
        importance_df_xgboost.to_csv(output_path, sep=sep, index=False)  # Full feature importance by XGBoost values
        print_with_indent(f"[i/o] Full feature importance (xgboost + {importance_type}) saved to: {output_path}", indent_level=1)

        # 3. Save top feature importance by hypothesis testing to file
        output_file = f"{subject}-{model_type}-importance-hypo-testing.{format}"
        output_path = os.path.join(output_dir, output_file)
        top_features_hypotest.to_csv(output_path, sep=sep, index=False)  # Top features by hypothesis testing
        print_with_indent(f"[i/o] Top feature importance (hypo-testing) saved to: {output_path}", indent_level=1)
        
        # 3a. Save full feature importance by hypothesis testing to file
        output_file = f"{subject}-{model_type}-importance-hypo-testing-full.{format}"
        output_path = os.path.join(output_dir, output_file)
        importance_df_hypotest.to_csv(output_path, sep=sep, index=False)  # Full feature importance by hypothesis testing
        print_with_indent(f"[i/o] Full feature importance (hypo-testing) saved to: {output_path}", indent_level=1)

        # 3b. Save full hypothesis testing result to file
        output_file = f"{subject}-{model_type}-hypo-testing-results.{format}"
        output_path = os.path.join(output_dir, output_file)
        df_hypotest.to_csv(output_path, sep=sep, index=False)  # Full hypothesis testing result
        print_with_indent(f"[i/o] Full hypothesis testing result saved to: {output_path}", indent_level=1)

        # 4. Save results of effect sizes to file
        output_file = f"{subject}-{model_type}-effect-sizes-results.{format}"
        output_path = os.path.join(output_dir, output_file)
        effect_sizes_df.to_csv(output_path, sep=sep, index=False)  # Full feature importance by effect sizes
        print_with_indent(f"[i/o] Full feature importance (effect sizes) saved to: {output_path}", indent_level=1)  

        # 4a. Save full list of feature importance by effect sizes to file
        output_file = f"{subject}-{model_type}-importance-effect-sizes-full.{format}"
        output_path = os.path.join(output_dir, output_file)
        importance_df_effect_sizes.to_csv(output_path, sep=sep, index=False)  # Full feature importance by effect sizes
        print_with_indent(f"[i/o] Full feature importance (effect sizes) saved to: {output_path}", indent_level=1)  

        # 5a. Save results of mutual information to file 
        output_file = f"{subject}-{model_type}-mutual-info-results.{format}"
        output_path = os.path.join(output_dir, output_file)
        mi_df.to_csv(output_path, sep=sep, index=False)  # Full feature importance by mutual information
        print_with_indent(f"[i/o] Full feature importance (mutual information) saved to: {output_path}", indent_level=1)  

        # 5b. Optionally save full list of feature importance by mutual information to file
        output_file = f"{subject}-{model_type}-importance-mutual-info-full.{format}"
        output_path = os.path.join(output_dir, output_file)
        importance_df_mi.to_csv(output_path, sep=sep, index=False)  # Full feature importance by mutual information
        print_with_indent(f"[i/o] Full feature importance (mutual information) saved to: {output_path}", indent_level=1)  

        # 6. Save motif-specific feature importance to file
        output_file = f"{subject}-{model_type}-motif-importance-shap.{format}"
        output_path = os.path.join(output_dir, output_file)
        top_motif_df.to_csv(output_path, sep=sep, index=False)  # Top motif features by SHAP values
        print_with_indent(f"[i/o] Motif importance saved to: {output_path}", indent_level=1)

        # 6a. Optionally save full motif-specific feature importance to file
        output_file = f"{subject}-{model_type}-motif-importance-shap-full.{format}"
        output_path = os.path.join(output_dir, output_file)
        full_motif_df.to_csv(output_path, sep=sep, index=False)  # Full motif features by SHAP values
        print_with_indent(f"[i/o] Full motif importance saved to: {output_path}", indent_level=1)
    
    return model, result_set


######################################################
# Feature Importance Comparison
######################################################


# Refactored to model_utils
# def compare_feature_importance_ranks(
#     importance_dfs, 
#     top_k=20,
#     primary_method='SHAP',
#     output_path=None, 
#     verbose=1, 
#     save=True,
#     plot_style="percentile_rank"
# ):
#     """
#     Compare feature importance across multiple methods.

#     Parameters
#     ----------
#     importance_dfs : dict
#         Dictionary where keys are method names (e.g., 'SHAP', 'XGBoost', 'Hypothesis')
#         and values are DataFrames with columns ['feature', 'importance_score'].
#     top_k : int
#         Number of top features to compare (based on the primary method).
#     primary_method : str
#         The key in importance_dfs used to select the top_k features.
#     output_path : str or None
#         Where to save the figure. Defaults to "feature_ranking_comparison.pdf" if None.
#     verbose : int
#         Verbosity level.
#     save : bool
#         If True, saves the plot; otherwise, displays it.
#     plot_style : {"rank", "reverse_rank", "raw", "percentile_rank"}
#         Representation style for importance scores.

#     Returns
#     -------
#     None
#     """
#     pass

# Refactored to model_utils
# def compare_feature_rankings(
#     importance_df_list,
#     method_names=None,
#     top_k=20,
#     verbose=True
# ):
#     """
#     Compare feature rankings from multiple importance DataFrames. 
#     Each df has columns ["feature","importance_score"] sorted descending.
#     Returns a summary of top-k overlaps and pairwise rank correlations.
#     """
#     pass


def plot_feature_ranking_overlap(shap_df, xgb_df, hypo_df, top_k=20, output_dir="output"):
    """
    Visualize ranking overlap between SHAP, XGBoost, and Hypothesis Testing.

    Parameters:
    - shap_df (pd.DataFrame): DataFrame with columns ['feature', 'importance_score'] from SHAP analysis.
    - xgb_df (pd.DataFrame): DataFrame with columns ['feature', 'importance_score'] from XGBoost feature importance.
    - hypo_df (pd.DataFrame): DataFrame with columns ['feature', 'importance_score'] from hypothesis testing.
    - top_k (int): Number of top features to compare.
    - output_dir (str): Directory to save the plots.

    Returns:
    - None (displays and saves plots)
    """
    # import seaborn as sns
    shap_features = set(shap_df.nlargest(top_k, "importance_score")["feature"])
    xgb_features = set(xgb_df.nlargest(top_k, "importance_score")["feature"])
    hypo_features = set(hypo_df.nlargest(top_k, "importance_score")["feature"])

    all_features = list(shap_features | xgb_features | hypo_features)
    df_overlap = pd.DataFrame(index=all_features, columns=["SHAP", "XGBoost", "Hypothesis"], dtype=int)
    
    df_overlap["SHAP"] = df_overlap.index.isin(shap_features).astype(int)
    df_overlap["XGBoost"] = df_overlap.index.isin(xgb_features).astype(int)
    df_overlap["Hypothesis"] = df_overlap.index.isin(hypo_features).astype(int)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_overlap.T, cmap="coolwarm", linewidths=0.5, cbar=False, annot=True, fmt="d")
    plt.title("Feature Importance Overlap Across Methods")
    plt.xlabel("Feature")
    plt.ylabel("Method")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save plot
    plt.savefig(f"{output_dir}/feature_ranking_overlap.pdf", dpi=300)
    plt.show()


def plot_violin_for_significant_features(df, top_features, class_column, output_dir="output"):
    """
    Generate violin plots for significantly different features from hypothesis testing.

    Parameters:
    - df (pd.DataFrame): Dataset containing feature values and class labels.
    - top_features (list): List of top features from hypothesis testing.
    - class_column (str): Column name for class labels.
    - output_dir (str): Directory to save plots.

    Returns:
    - None (displays and saves plots)
    """
    for feature in top_features:
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=df[class_column], y=df[feature], inner="quartile", palette="muted")
        plt.title(f"Feature Distribution: {feature}")
        plt.xlabel("Class Label")
        plt.ylabel(feature)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{output_dir}/violin_{feature}.pdf", dpi=300)
        plt.show()


######################################################
# Feature Importance Hypothesis Testing
######################################################


def quantify_feature_importance_via_mutual_info(X, y, top_features, *, feature_categories=None, verbose=1, output_path=None, **kargs):
    """
    Quantify feature importance using mutual information.
    
    This provides a unified approach to measure feature importance for both categorical and numerical features
    on the same scale. Mutual information measures how much knowing one variable reduces uncertainty about another.
    """
    print(f"Quantifying feature importance via mutual information ...")
    return compute_feature_mutual_info(
        X=X,
        y=y,
        top_features=top_features,
        feature_categories=feature_categories,
        verbose=verbose,
        output_path=output_path,
        **kargs
    )


def compute_feature_mutual_info(X, y, top_features, feature_categories=None, verbose=1, output_path=None, **kargs):
    """
    Compute mutual information between features and target to quantify importance.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature DataFrame.
    y : pd.Series or array
        Binary labels (0, 1).
    top_features : list or pd.DataFrame
        Features to compute mutual information for.
        - If a pd.DataFrame, it should have a column 'feature' with the feature names.
    feature_categories : dict (optional)
        Dictionary categorizing features: numerical, categorical, ordinal.
    verbose : int
        Verbosity level.
    output_path : str (optional)
        File path to save the results.
        
    Returns:
    --------
    mi_df : pd.DataFrame
        DataFrame summarizing features and their mutual information scores.
    feature_importance_df : pd.DataFrame
        Standardized DataFrame with columns ['feature', 'importance_score'] based on mutual information.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import OrdinalEncoder
    
    results = []
    
    # Feature configuration
    label_col = 'class_label'
    kmer_pattern = kargs.get("kmer_pattern", r'^\d+mer_.*')  # Regex for k-mers
    
    # Classify features if not already provided
    if feature_categories is None:
        from .analysis_utils import classify_features
        data = (X, y)
        feature_categories = classify_features(data, label_col=label_col, kmer_pattern=kmer_pattern)
    
    # Extract feature types
    categorical_vars = feature_categories.get('categorical_features', []) + \
                       feature_categories.get('derived_categorical_features', [])
    categorical_features = set(categorical_vars)
    
    # Extract feature list from DataFrame if needed
    if isinstance(top_features, pd.DataFrame):
        top_features = top_features["feature"].values
    
    # Get encoder for categorical features
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Ensure y is encoded as 0/1 for binary classification
    y_encoded = y.astype(int)
    
    n_display = 0
    
    for feature in top_features:
        if feature not in X.columns:
            continue
        
        # Determine feature type
        is_categorical = feature in categorical_features
        feature_type = 'Categorical' if is_categorical else 'Numerical'
        
        # Prepare feature data for MI calculation
        feature_data = X[feature].values.reshape(-1, 1)
        if is_categorical:
            # Encode categorical features
            try:
                feature_data = encoder.fit_transform(feature_data)
            except:
                # If encoding fails, try converting to string
                feature_data = np.array([str(x) for x in X[feature]]).reshape(-1, 1)
                feature_data = encoder.fit_transform(feature_data)
        
        # Calculate mutual information for classification
        mi_score = mutual_info_classif(feature_data, y_encoded, discrete_features=is_categorical)[0]
        
        effect_info = {
            'Feature': feature,
            'Type': feature_type,
            'MI Score': mi_score,
        }
        
        if verbose and n_display < 10:
            print(f"Feature: {feature}, MI Score: {mi_score:.6f}")
            n_display += 1
        
        results.append(effect_info)
    
    # Sort by mutual information score (descending)
    mi_df = pd.DataFrame(results).sort_values(by='MI Score', ascending=False).reset_index(drop=True)
    
    # Create a simplified DataFrame for plotting (compatible with plot_feature_importance)
    feature_importance_df = mi_df[['Feature', 'MI Score']].rename(
        columns={
            'Feature': 'feature',
            'MI Score': 'importance_score'
        }
    ).sort_values(by='importance_score', ascending=False).reset_index(drop=True)
    
    save = kargs.get("save", False)
    format_ = kargs.get("format", "tsv")
    model_type = kargs.get("model_type", "model")
    sep = '\t' if format_ == 'tsv' else ','
    
    if output_path is None:
        output_file = f"{model_type}-importance-mutual-info.{format_}"
        output_path = os.path.join(os.getcwd(), output_file)
    
    if save:
        mi_df.to_csv(output_path, sep=sep, index=False)
        if verbose:
            print(f"Saved mutual information results to {output_path}")
    
    return mi_df, feature_importance_df


def quantify_feature_importance_via_measuring_effect_sizes(
    X, y, top_features, *, feature_categories=None, verbose=1, output_path=None, **kargs
):
    """
    Compute practical effect sizes for given features to quantify importance.
    """
    # from scipy.stats import chi2_contingency
    print(f"Quantifying feature importance via measuring effect sizes ...")
    return compute_feature_effect_sizes(
        X=X,
        y=y,
        top_features=top_features,
        feature_categories=feature_categories,
        verbose=verbose, 
        output_path=output_path, 
        **kargs
    )


def compute_feature_effect_sizes(X, y, top_features, feature_categories=None, verbose=1, output_path=None, **kargs):
    """
    Compute practical effect sizes for given features to quantify importance.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature DataFrame.
    y : pd.Series or array
        Binary labels (0, 1).
    top_features : list or pd.DataFrame
        Features to compute effect sizes for.
        - If a pd.DataFrame, it should have a column 'feature' with the feature names.
    feature_categories : dict (optional)
        Dictionary categorizing features: numerical, categorical, ordinal.
    verbose : int
        Verbosity level.
    output_path : str (optional)
        File path to save the results.

    Returns:
    --------
    effect_sizes_df : pd.DataFrame
        DataFrame summarizing features and their effect sizes.
    feature_importance_df : pd.DataFrame
        Standardized DataFrame with columns ['feature', 'importance_score'] based on effect sizes.

    Usage:
    ------
        feature_effect_sizes_df = compute_feature_effect_sizes(X, y, top_features)
        print(feature_effect_sizes_df)
    """
    from scipy.stats import chi2_contingency

    results = []

    # Feature configuration
    label_col = 'class_label'
    kmer_pattern = kargs.get("kmer_pattern", r'^\d+mer_.*')  # Regex for k-mers

    # Classify features
    if feature_categories is None:
        data = (X, y)
        feature_categories = classify_features(data, label_col=label_col, kmer_pattern=kmer_pattern)
        # feature_columns = (
        #     feature_categories['categorical_features'] +
        #     feature_categories['numerical_features'] +
        #     feature_categories['derived_categorical_features'] + 
        # )

    # Extract and explicitly add motif features to numerical
    motif_vars = feature_categories.get('motif_features', [])
    categorical_vars = feature_categories.get('categorical_features', []) + \
                       feature_categories.get('derived_categorical_features', [])
    numerical_vars = feature_categories.get('numerical_features', []) + motif_vars
    ordinal_vars = set(feature_categories.get('ordinal_features', []))

    # Feature classification logic
    if feature_categories:
        numeric_features = set(numerical_vars)
        categorical_features = set(categorical_vars)   
        ordinal_features = set(ordinal_vars)
    else:
        # A default fallback (when feature_categories is effectively empty as an empty dict)
        numeric_features = set(df.select_dtypes(include=[np.number]).columns) - {label_col}
        categorical_features = set(df.select_dtypes(include=["object", "category"]).columns)
        binary_features = {col for col in df.columns if df[col].nunique() == 2 and col != "class_label"}
        categorical_features |= binary_features
        ordinal_features = set()
    # Debug prints for verification (highly recommended)
    if verbose:
        print(f"[DEBUG] Numerical features identified: {numeric_features}")
        print(f"[DEBUG] Categorical features identified: {categorical_features}")
        print(f"[DEBUG] Ordinal features identified: {ordinal_features}")

    # ------------------------------------------------------------

    if isinstance(top_features, pd.DataFrame):
        top_features = top_features["feature"].values

    n_display = 0

    for feature in top_features:
        if feature not in X.columns:
            continue

        pos_class = X.loc[y == 1, feature]
        neg_class = X.loc[y == 0, feature]

        if feature in numeric_features:
            mean_diff = pos_class.mean() - neg_class.mean()
            pooled_std = np.sqrt(
                ((len(pos_class)-1)*pos_class.var() + (len(neg_class)-1)*neg_class.var()) / 
                (len(pos_class) + len(neg_class) - 2)
            )
            cohen_d = mean_diff / pooled_std

            effect_info = {
                'Feature': feature,
                'Type': 'Numeric',
                'Effect Size Metric': "Cohen's d",
                'Effect Size': cohen_d
            }

        elif feature in ordinal_features:
            U_stat, _ = mannwhitneyu(pos_class, neg_class, alternative='two-sided')
            n1, n2 = len(pos_class), len(neg_class)
            rank_biserial_corr = 1 - (2 * U_stat) / (n1 * n2)

            effect_info = {
                'Feature': feature,
                'Type': 'Ordinal',
                'Effect Size Metric': 'Rank-Biserial Correlation',
                'Effect Size': rank_biserial_corr
            }

        elif feature in categorical_features:
            contingency = pd.crosstab(X[feature], y)
            chi2, _, _, _ = chi2_contingency(contingency)
            n = contingency.sum().sum()
            phi2 = chi2 / n
            r, k = contingency.shape
            cramer_v = np.sqrt(phi2 / (min(k - 1, r - 1)))

            effect_info = {
                'Feature': feature,
                'Type': 'Categorical',
                'Effect Size Metric': "Cramer's V",
                'Effect Size': cramer_v
            }

        else:
            continue

        if verbose:
            if n_display < 10:
                print(f"Feature: {feature}, Effect Size: {effect_info}")
                n_display += 1

        results.append(effect_info)

    effect_sizes_df = pd.DataFrame(results).sort_values(by='Effect Size', key=np.abs, ascending=False).reset_index(drop=True)

    feature_importance_df = effect_sizes_df[['Feature', 'Effect Size']].rename(
        columns={
            'Feature': 'feature',
            'Effect Size': 'importance_score'
        }
    ).sort_values(by='importance_score', key=np.abs, ascending=False).reset_index(drop=True)

    save = kargs.get("save", False)
    format_ = kargs.get("format", "tsv")
    model_type = kargs.get("model_type", "model")
    sep = '\t' if format_ == 'tsv' else ','

    if output_path is None:
        output_file = f"{model_type}-importance-effect-sizes.{format_}"
        output_path = os.path.join(os.getcwd(), output_file)

    if save:
        effect_sizes_df.to_csv(output_path, sep=sep, index=False)
        if verbose > 0:
            print(f"[i/o] Effect size results saved to: {output_path}")

    return effect_sizes_df, feature_importance_df  


######################################################


def compute_effect_size(df, top_features, class_column):
    """
    Compute Cohen's d effect size for numerical features.

    Parameters:
    - df (pd.DataFrame): Dataset containing feature values and class labels.
    - top_features (list): List of numerical features from hypothesis testing.
    - class_column (str): Column name for class labels.

    Returns:
    - effect_size_df (pd.DataFrame): DataFrame with effect sizes.
    """
    results = []
    for feature in top_features:
        pos_class = df[df[class_column] == 1][feature]
        neg_class = df[df[class_column] == 0][feature]
        
        mean_diff = np.mean(pos_class) - np.mean(neg_class)
        pooled_std = np.sqrt((np.var(pos_class) + np.var(neg_class)) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

        results.append({"feature": feature, "effect_size": effect_size})

    return pd.DataFrame(results).sort_values(by="effect_size", ascending=False)


def perform_feature_hypothesis_tests_v0(
    X, y, top_features, output_dir, feature_categories=None, subject="fp_vs_tp", alpha=0.05, verbose=1, **kargs
):
    """
    Perform hypothesis testing on top features to evaluate their separability
    between the positive class (errors: FP/FN) and the negative class (correct predictions: TP/TN).
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame (excluding label column).
    y : pd.Series or np.array
        Label column (binary).
    top_features : list
        List of top features from SHAP analysis.
    output_dir : str
        Directory where results will be saved.
    feature_categories : dict, optional
        Precomputed dictionary of feature types (keys: 'numerical_features', 'categorical_features').
    alpha : float, optional
        Significance threshold for hypothesis tests. Default is 0.05.
    verbose : int, optional
        Verbosity level. If >0, prints results.

    Returns
    -------
    results_df : pd.DataFrame
        Dataframe summarizing p-values and test results for each feature.
    """
    from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, fisher_exact, shapiro
    
    # Merge X and y for analysis
    df = X.copy()
    df['class_label'] = y  # Ensure class column is included

    # Feature classification logic
    if feature_categories:
        num_features = set(feature_categories.get("numerical_features", []))
        cat_features = set(feature_categories.get("categorical_features", []))
    else:
        num_features = set(df.select_dtypes(include=[np.number]).columns) - {"class_label"}
        cat_features = set(df.select_dtypes(include=["object", "category"]).columns)
        cat_features |= {col for col in df.columns if df[col].nunique() == 2 and col != "class_label"}  # Add binary features

    results = []
    
    for feature in top_features:
        if feature not in X.columns:
            continue  # Skip missing features
        
        pos_class = df[df["class_label"] == 1][feature]
        neg_class = df[df["class_label"] == 0][feature]

        # Initialize shapiro_p to None or np.nan
        shapiro_p = np.nan

        # **Numerical Feature Hypothesis Testing**
        if feature in num_features:
            # Step 1: Test for normality
            shapiro_p = shapiro(df[feature])[1]  # Get p-value
            is_normal = shapiro_p >= alpha  # If p >= 0.05, assume normality
            
            if is_normal:
                test_name = "t-test"
                stat, p_value = ttest_ind(pos_class, neg_class, equal_var=False)
            else:
                test_name = "Wilcoxon Rank-Sum"
                stat, p_value = mannwhitneyu(pos_class, neg_class, alternative='two-sided')

        # **Categorical Feature Hypothesis Testing (Includes Binary Features)**
        elif feature in cat_features:
            contingency_table = pd.crosstab(df[feature], df["class_label"])
            # This creates a contingency table (cross-tabulation) that counts occurrences of 
            # different feature values across the two classes (positive = 1, negative = 0).
            # => This table helps assess if a feature’s distribution is independent of class labels

            # Compute expected frequencies for the Chi-square test
            chi2_stat, chi2_p, dof, expected_freqs = chi2_contingency(contingency_table)
            min_expected = np.min(expected_freqs)
            # Check the minimum expected cell count before deciding between Chi-square vs. Fisher’s Exact Test

            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1 and min_expected >= 5:
                test_name = "Chi-square Test"
                stat, p_value = chi2_stat, chi2_p
            else:
                test_name = "Fisher’s Exact Test"
                stat, p_value = fisher_exact(contingency_table)

        else:
            continue  # Skip unknown feature types

        # Determine significance
        significant = p_value < alpha
        
        results.append({
            "Feature": feature,
            "Test": test_name,
            "Statistic": stat,
            "p-value": p_value,
            "Significant": significant,
            "Min Expected Cell Count": min_expected if feature in cat_features else np.nan,
            "Shapiro-Wilk p-value": shapiro_p if feature in num_features else np.nan,
        })

        if verbose > 0:
            print(f"{feature}: {test_name} | p = {p_value:.5f} | {'Significant' if significant else 'Not Significant'}")

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results).sort_values(by="p-value")

    # Save to file
    format = kargs.get("format", "tsv")
    model_type = kargs.get("model_type", "xgboost")
    sep = '\t' if format == 'tsv' else ','
    output_file = f"{subject}-{model_type}-importance-hypo-testing.{format}"
    output_path = os.path.join(output_dir, output_file)
    results_df.to_csv(output_path, sep=sep, index=False)

    return results_df


def quantify_feature_importance_via_hypothesis_testing(
    X, y, top_features, *, output_path=None, feature_categories=None, alpha=0.05, verbose=1, **kargs
):
    """
    Perform hypothesis testing on top features to evaluate their separability
    between the positive class (errors: FP/FN) and the negative class (correct predictions: TP/TN).
    """
    print(f"Quantifying feature importance via hypothesis testing ...")
    # return perform_feature_hypothesis_tests(
    #     X, y, top_features, output_path, feature_categories, alpha, verbose, **kargs)
    return perform_feature_hypothesis_tests(
        X=X,
        y=y,
        top_features=top_features,
        output_path=output_path,
        feature_categories=feature_categories,
        alpha=alpha,
        verbose=verbose,
        **kargs
    )


def perform_feature_hypothesis_tests(
    X, y, top_features, output_path=None, feature_categories=None, alpha=0.05, verbose=1, **kargs
):
    """
    Perform hypothesis testing on top features to evaluate their separability
    between the positive class (errors: FP/FN) and the negative class (correct predictions: TP/TN).

    Additionally, apply Benjamini-Hochberg FDR correction to reduce false positives
    when testing many features. Return updated significance results and FDR-adjusted
    -log10(p-value) as an "Importance Score (FDR)".

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame (excluding label column).
    y : pd.Series or np.array
        Label column (binary: 1 = errors, 0 = correct predictions).
    top_features : list
        List of top features from SHAP or other analyses that you want to test.
    output_path : str
        Path to save the full results as a file.
    feature_categories : dict, optional
        Precomputed dictionary of feature types (keys: 'numerical_features', 'categorical_features').
    alpha : float, optional
        Significance threshold for hypothesis tests. Default is 0.05.
        This same alpha is used in the BH FDR procedure.
    verbose : int, optional
        Verbosity level. If > 0, prints p-values and significance for each feature.

    Returns
    -------
    results_df : pd.DataFrame
        Dataframe summarizing raw p-values, FDR-adjusted p-values, and test results for each feature.
    feature_importance_df : pd.DataFrame
        Dataframe formatted as ('feature', 'importance_score') based on -log10(FDR-adjusted p-value),
        sorted in descending order of importance.
    """
    from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, fisher_exact, shapiro
    from statsmodels.stats.multitest import multipletests

    df = X.copy()
    df['class_label'] = y  # Ensure class column is included
    n_samples = len(df)

    # Feature configuration
    label_col = 'class_label'
    kmer_pattern = kargs.get("kmer_pattern", r'^\d+mer_.*')  # Regex for k-mers

    # Classify features
    if feature_categories is None:
        feature_categories = classify_features(df, label_col=label_col, kmer_pattern=kmer_pattern)

        # feature_columns = (
        #     feature_categories['categorical_features'] +
        #     feature_categories['numerical_features'] +
        #     feature_categories['derived_categorical_features'] + 
        # )

    # Extract and explicitly add motif features to numerical
    motif_vars = feature_categories.get('motif_features', [])
    categorical_vars = feature_categories.get('categorical_features', []) + \
                       feature_categories.get('derived_categorical_features', [])
    numerical_vars = feature_categories.get('numerical_features', []) + motif_vars

    # Feature classification logic
    if feature_categories:
        num_features = set(numerical_vars)
        cat_features = set(categorical_vars)   
    else:
        # A default fallback (when feature_categories is effectively empty as an empty dict)
        num_features = set(df.select_dtypes(include=[np.number]).columns) - {label_col}
        cat_features = set(df.select_dtypes(include=["object", "category"]).columns)
        binary_features = {col for col in df.columns if df[col].nunique() == 2 and col != "class_label"}
        cat_features |= binary_features

    # Debug prints for verification (highly recommended)
    if verbose:
        print(f"[DEBUG] Numerical features identified: {list(num_features)[:100]}")
        print(f"[DEBUG] Categorical features identified: {list(cat_features)[:100]}")

    # ------------------------------------------------------------

    results = []
    skipped_features = []  # List to track skipped features

    if isinstance(top_features, pd.DataFrame):
        top_features = top_features["feature"].values

    missing_features = set(top_features) - set(X.columns)
    if missing_features:
        print(f"[Warning] Missing features: {missing_features}")

    # 1) Collect raw p-values for each feature
    for feature in top_features:
        if feature not in X.columns:
            skipped_features.append(feature)  # Track missing features
            continue  # Skip missing features

        pos_class = df.loc[df["class_label"] == 1, feature]
        neg_class = df.loc[df["class_label"] == 0, feature]

        # Initialize shapiro_p to None or np.nan
        shapiro_p = np.nan

        if feature in num_features:
            # Mann-Whitney U works for both numeric and ordinal (ordered categories)
            # - When the sample size is large => use Mann-Whitney U
            # - When the sample size is small => check normality (use Shapiro-Wilk test)
            #     - If normal => use t-test
            # - Ordinal feature => use Mann-Whitney U
            if n_samples > 5000:
                # For large sample sizes, default to nonparametric test
                test_name = "Mann-Whitney U"
                stat, p_value = mannwhitneyu(pos_class, neg_class, alternative='two-sided')
            else: 
                # For small sample sizes, use Shapiro-Wilk test
                shapiro_p = shapiro(df[feature])[1]
                is_normal = (shapiro_p >= alpha)  # If p >= alpha => assume normal
                if is_normal:
                    test_name = "t-test"
                    stat, p_value = ttest_ind(pos_class, neg_class, equal_var=False)
                else:
                    test_name = "Mann-Whitney U"  # "Wilcoxon Rank-Sum" is the same as "Mann-Whitney U"
                    stat, p_value = mannwhitneyu(pos_class, neg_class, alternative='two-sided')

            # for categorical features, see below
        elif feature in cat_features:
            contingency_table = pd.crosstab(df[feature], df["class_label"])
            shape = contingency_table.shape

            # chi2_stat, chi2_p, dof, expected_freqs = chi2_contingency(contingency_table)
            # min_expected = np.min(expected_freqs)

            if shape == (2, 2):
                min_expected = chi2_contingency(contingency_table)[3].min()
                if min_expected < 5:
                    test_name = "Fisher’s Exact Test"
                    stat, p_value = fisher_exact(contingency_table)
                else:
                    test_name = "Chi-square Test"
                    stat, p_value, _, _ = chi2_contingency(contingency_table)
            elif shape[0] > 1 and shape[1] > 1:
                test_name = "Chi-square Test (larger table)"
                stat, p_value, _, expected = chi2_contingency(contingency_table)
                if expected.min() < 5 and verbose:
                    print_with_indent(f"Warning: Low expected cell count ({expected.min():.2f}) for '{feature}'", indent_level=1)
                min_expected = expected.min()
            else:
                if verbose:
                    print_with_indent(f"Skipping feature '{feature}' due to invalid table shape: {shape}", indent_level=1)
                skipped_features.append(feature)
                continue
        else:
            # skip unknown feature type
            print_with_indent(f"Skipping unknown feature type: {feature} (type={df[feature].dtype})", indent_level=1)
            skipped_features.append(feature)  # Track unknown feature types
            continue

        # avoid log(0) in -log10(p)
        if p_value < 1e-300:
            p_value = 1e-300

        importance_score = -np.log10(p_value)
        # raw significance
        significant = p_value < alpha

        row_info = {
            "Feature": feature,
            "Test": test_name,
            "Statistic": stat,
            "p-value (raw)": p_value,
            "Importance Score (raw) (-log10 p)": importance_score,
            "Significant (raw)": significant,
            "Min Expected Cell Count": min_expected if feature in cat_features else np.nan,
            "Shapiro-Wilk p-value": shapiro_p if feature in num_features else np.nan,
        }
        results.append(row_info)

        if verbose > 0:
            print(f"{feature}: {test_name} | p = {p_value:.5g} | {'Significant' if significant else 'Not Significant'}")

    if verbose > 0: 
        # Check how many features are significant before FDR correction
        num_significant = sum(row["Significant (raw)"] for row in results)
        print(f"[summary] {num_significant} features are significant before FDR correction.")

    # 2) Build results DataFrame sorted by raw p-value
    results_df = pd.DataFrame(results).sort_values(by="p-value (raw)")

    # 3) Apply Benjamini-Hochberg FDR correction to all raw p-values
    raw_pvals = results_df["p-value (raw)"].values
    rejected, adj_pvals, _, _ = multipletests(raw_pvals, alpha=alpha, method='fdr_bh')
    # NOTE: 
    # - rejected is a boolean array telling us if each test is significant at FDR < alpha.
    # - adj_pvals are the FDR-adjusted p-values.

    # 4) Update the DataFrame with FDR p-values
    results_df["p-value (BH)"] = adj_pvals  # FDR-adjusted p-value
    results_df["Significant (BH)"] = rejected  # Significant at FDR < alpha
    
    # FDR-based importance => -log10(adjusted p-value)
    
    # results_df["Importance Score (FDR)"] = -np.log10(adj_pvals.clip(lower=1e-300))
    # NOTE: May lead to ValueError: One of max or min must be given happens 
    # because numpy.clip() in certain NumPy versions requires both a min and max argument.
    adj_pvals_clipped = np.maximum(adj_pvals, 1e-300)  # ensures we never take log(0)
    results_df["Importance Score (FDR)"] = -np.log10(adj_pvals_clipped)

    # reorder by FDR p-value ascending
    results_df = results_df.sort_values(by="p-value (BH)").reset_index(drop=True)
    # the most significant features appear at the top.

    # 5) Build final "feature_importance_df" using FDR-based importance
    feature_importance_df = results_df[["Feature", "Importance Score (FDR)"]].rename(
        columns={
            "Feature": "feature",
            "Importance Score (FDR)": "importance_score"
        }
    ).sort_values(by="importance_score", ascending=False).reset_index(drop=True)
    # Standarize the column names for consistency with other feature importance methods

    if verbose: 
        # Check how many features remain significant under FDR
        num_significant = results_df["Significant (BH)"].sum()
        print(f"[summary] {num_significant} features remain significant after FDR correction (n_top_features={len(top_features)}).")
        print_with_indent(f"Percentage of significant features: {num_significant / len(results_df) * 100:.2f}%", indent_level=1)

        # Print skipped features
        if skipped_features:
            print(f"[info] Skipped features (n={len(skipped_features)}): {', '.join(skipped_features)}")

    # Save to file (optional)
    save = kargs.get("save", False)
    format_ = kargs.get("format", "tsv")
    model_type = kargs.get("model_type", "xgboost")
    sep = '\t' if format_ == 'tsv' else ','

    if output_path is None:
        output_file = f"{model_type}-importance-hypo-testing.{format_}"
        output_path = os.path.join(os.getcwd(), output_file)

    if save: 
        results_df.to_csv(output_path, sep=sep, index=False)
        if verbose > 0:
            print(f"[i/o] Hypothesis testing results (with BH-FDR) saved to: {output_path}")

    return results_df, feature_importance_df


def perform_feature_hypothesis_tests_with_dataframe(
    df, top_features, class_column, output_dir, alpha=0.05, verbose=1
):
    """
    Perform hypothesis testing on top features to evaluate their separability
    between the positive class (errors: FP/FN) and the negative class (correct predictions: TP/TN).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing feature values and the class label.
    top_features : list
        List of top features from SHAP analysis.
    class_column : str
        Column name representing class labels (1 = errors, 0 = correct predictions).
    output_dir : str
        Directory where results will be saved.
    alpha : float, optional
        Significance threshold for hypothesis tests. Default is 0.05.
    verbose : int, optional
        Verbosity level. If >0, prints results.

    Returns
    -------
    results_df : pd.DataFrame
        Dataframe summarizing p-values and test results for each feature.
    """
    from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, fisher_exact, shapiro

    results = []
    
    for feature in top_features:
        if feature not in df.columns:
            continue  # Skip missing features
        
        feature_data = df[[feature, class_column]].dropna()
        pos_class = feature_data[feature_data[class_column] == 1][feature]
        neg_class = feature_data[feature_data[class_column] == 0][feature]

        # Determine feature type
        if feature_data[feature].dtype in [np.float64, np.int64]:  # Numerical Feature
            # Step 1: Test for normality
            shapiro_p = shapiro(feature_data[feature])[1]  # Get p-value
            is_normal = shapiro_p >= alpha  # If p >= 0.05, assume normality
            
            if is_normal:
                test_name = "t-test"
                stat, p_value = ttest_ind(pos_class, neg_class, equal_var=False)
            else:
                test_name = "Wilcoxon Rank-Sum"
                stat, p_value = mannwhitneyu(pos_class, neg_class, alternative='two-sided')

        elif feature_data[feature].dtype == 'object':  # Categorical Feature
            contingency_table = pd.crosstab(feature_data[feature], feature_data[class_column])
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                test_name = "Chi-square Test"
                stat, p_value, _, _ = chi2_contingency(contingency_table)
            else:
                test_name = "Fisher’s Exact Test"
                stat, p_value = fisher_exact(contingency_table)

        else:  # Binary Feature
            contingency_table = pd.crosstab(feature_data[feature], feature_data[class_column])
            if contingency_table.shape[0] == 2 and contingency_table.shape[1] == 2:
                test_name = "Chi-square Test"
                stat, p_value, _, _ = chi2_contingency(contingency_table)
            else:
                test_name = "Fisher’s Exact Test"
                stat, p_value = fisher_exact(contingency_table)

        # Determine significance
        significant = p_value < alpha
        
        results.append({
            "Feature": feature,
            "Test": test_name,
            "Statistic": stat,
            "p-value": p_value,
            "Significant": significant,
            "Shapiro-Wilk p-value": shapiro_p if feature_data[feature].dtype in [np.float64, np.int64] else np.nan,
        })

        if verbose > 0:
            print(f"{feature}: {test_name} | p = {p_value:.5f} | {'Significant' if significant else 'Not Significant'}")

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results).sort_values(by="p-value")

    # Save to file
    results_df.to_csv(f"{output_dir}/feature_hypothesis_tests.csv", index=False)

    return results_df


def quantify_feature_importance_via_xgboost(
    model, 
    X, *, 
    output_path=None, 
    # subject="xgboost", 
    top_k=10, 
    save=True, 
    importance_type="weight", 
    color_map="viridis",      # <-- new param for colormap name
    verbose=1
):
    """
    Quantify feature importance via XGBoost.
    """
    print(f"Quantifying feature importance via XGBoost ({importance_type}) ...")
    return plot_xgboost_feature_importance(
        model=model,
        X=X,
        output_path=output_path,
        top_k=top_k,
        save=save,
        importance_type=importance_type,
        color_map=color_map,
        verbose=verbose
    )


def plot_xgboost_feature_importance(
    model, 
    X, 
    output_path=None, 
    top_k=10, 
    save=True, 
    importance_type="weight", 
    color_map="viridis",
    verbose=1,
    **kwargs
):
    """
    Plot and save the top-k feature importance from an XGBoost model with enhanced aesthetics.

    Parameters:
    - model: Trained XGBoost model.
    - X: DataFrame containing feature columns.
    - output_path (str): Path to save the plot.
    - top_k (int): Number of top features to display.
    - save (bool): Whether to save the plot.
    - importance_type (str): Importance type ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
    - verbose (int): Verbosity level (0 = silent, 1 = verbose).

    Returns:
    - top_features (pd.DataFrame): Top-k features with importance scores.
    - feature_importance_df (pd.DataFrame): DataFrame with all feature importance scores.

    Memo: 
    -  Feature importance ranking via XGBoost:
       - Weight: The number of times a feature is used to split the data across all trees.
       - Gain: The average gain of the feature when it is used in trees.
       - Cover: The average coverage (number of samples affected) of the feature when it is used in trees.
       - Total Gain: The total gain of the feature when it is used in trees.
       - Total Cover: The total coverage of the feature when it is used in trees.

    Updates: 
    - color_map: Use a colormap for better visual distinction between features.
    """

    import matplotlib.cm as cm
    valid_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    if importance_type not in valid_types:
        raise ValueError(f"Invalid importance_type '{importance_type}'. Choose from {valid_types}.")

    if verbose > 0:
        print(f"[info] Using importance type: {importance_type}")

    feature_importance_dict = model.get_booster().get_score(importance_type=importance_type)
    feature_importance_df = pd.DataFrame.from_dict(
        feature_importance_dict, 
        orient='index', 
        columns=['importance_score']
    ).reset_index().rename(columns={'index': 'feature'})

    # Add any features missing in the dictionary with 0 importance
    missing_features = set(X.columns) - set(feature_importance_df['feature'])
    if missing_features:
        missing_df = pd.DataFrame({'feature': list(missing_features), 'importance_score': 0.0})
        feature_importance_df = pd.concat([feature_importance_df, missing_df], ignore_index=True)

    feature_importance_df = feature_importance_df.sort_values(by='importance_score', ascending=False)
    top_features = feature_importance_df.head(top_k)

    if verbose > 0:
        print(f"[info] Top {top_k} features by {importance_type}:")
        print(top_features)

    plt.figure(figsize=(12, 8))

    cmap = cm.get_cmap(color_map)

    # -- Min–max scaling for the top features --
    scores = top_features['importance_score'].values
    min_val = scores.min()
    max_val = scores.max()
    denom = max_val - min_val if max_val != min_val else 1e-9
    normalized_scores = (scores - min_val) / denom

    # Map normalized scores to RGBA
    colors = [cmap(val) for val in normalized_scores]

    sns.barplot(
        x='importance_score',
        y='feature',
        data=top_features,
        palette=colors,
        edgecolor='black',
        **kwargs
    )

    plt.xlabel("Importance Score", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.title(f"Top {top_k} XGBoost Feature Importance ({importance_type.capitalize()})", fontsize=16, pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save:
        if not output_path: 
            output_path = os.path.join(os.getcwd(), f"xgboost-importance-{importance_type}.pdf")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        if verbose > 0:
            print(f"[info] Feature importance plot saved to: {output_path}")
        plt.close()
    else:
        plt.show()

    return top_features, feature_importance_df


def plot_xgboost_feature_importance_v0(
    model, 
    X, 
    output_path=None, 
    top_k=10, 
    save=True, 
    importance_type="weight", 
    verbose=1
):
    """
    Plot and save the top-k feature importance from an XGBoost model with enhanced aesthetics.

    Parameters:
    - model: Trained XGBoost model.
    - X: DataFrame containing feature columns.
    - output_path (str): Path to save the plot.
    - top_k (int): Number of top features to display.
    - save (bool): Whether to save the plot.
    - importance_type (str): Importance type ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
    - verbose (int): Verbosity level (0 = silent, 1 = verbose).

    Returns:
    - top_features (pd.DataFrame): Top-k features with importance scores.
    - feature_importance_df (pd.DataFrame): DataFrame with all feature importance scores.

    Memo: 
    -  Feature importance ranking via XGBoost:
       - Weight: The number of times a feature is used to split the data across all trees.
       - Gain: The average gain of the feature when it is used in trees.
       - Cover: The average coverage (number of samples affected) of the feature when it is used in trees.
       - Total Gain: The total gain of the feature when it is used in trees.
       - Total Cover: The total coverage of the feature when it is used in trees.

    Updates: 
    - Color coding: Used a color gradient (sns.light_palette) based on importance scores, 
        clearly highlighting the most influential features.
    """

    # Validate importance type
    valid_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    if importance_type not in valid_types:
        raise ValueError(f"Invalid importance_type '{importance_type}'. Choose from {valid_types}.")

    if verbose > 0:
        print(f"[info] Using importance type: {importance_type}")

    # Get feature importance
    feature_importance_dict = model.get_booster().get_score(importance_type=importance_type)

    # Convert to DataFrame
    feature_importance_df = pd.DataFrame.from_dict(
        feature_importance_dict, orient='index', columns=['importance_score']
    ).reset_index().rename(columns={'index': 'feature'})

    # Include missing features with zero importance
    missing_features = set(X.columns) - set(feature_importance_df['feature'])
    if missing_features:
        missing_df = pd.DataFrame({'feature': list(missing_features), 'importance_score': 0.0})
        feature_importance_df = pd.concat([feature_importance_df, missing_df], ignore_index=True)

    # Sort and select top-k features
    feature_importance_df = feature_importance_df.sort_values(by='importance_score', ascending=False)
    top_features = feature_importance_df.head(top_k)

    if verbose > 0:
        print(f"[info] Top {top_k} features by {importance_type}:")
        print(top_features)

    # Aesthetic plot setup
    plt.figure(figsize=(12, 8))
    cmap = sns.light_palette("viridis", reverse=True, as_cmap=True)
    colors = cmap(top_features['importance_score'] / top_features['importance_score'].max())

    sns.barplot(
        x='importance_score', y='feature', data=top_features,
        palette=colors, edgecolor='black'
    )

    plt.xlabel("Importance Score", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.title(f"Top {top_k} XGBoost Feature Importance ({importance_type.capitalize()})", fontsize=16, pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save or display plot
    if save:
        if not output_path: 
            output_path = os.path.join(os.getcwd(), f"xgboost-importance-{importance_type}.pdf")

        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        if verbose > 0:
            print(f"[info] Feature importance plot saved to: {output_path}")
        plt.close()
    else:
        plt.show()

    return top_features, feature_importance_df


def plot_shap_summary(model, X_test, output_dir="output", plot_name="shap_summary.pdf", top_k=10, verbose=1):
    """
    Generate and save SHAP summary plot.

    Parameters:
    ----------
    model : trained model
        The trained machine learning model (e.g., XGBoost).
    X_test : Polars DataFrame or Pandas DataFrame
        The test dataset used for generating SHAP values.
    output_dir : str
        Directory to save the SHAP summary plot.
    plot_name : str
        Name of the output plot file.
    verbose : int
        Verbosity level.

    Returns:
    -------
    None
    """
    if isinstance(X_test, pl.DataFrame):
        X_test = X_test.to_pandas()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize SHAP explainer and calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Generate SHAP summary plot
    shap.summary_plot(shap_values, X_test, show=False, max_display=top_k)

    # Save plot
    shap_summary_path = os.path.join(output_dir, plot_name)
    if verbose:
        print(f"[info] Saving SHAP summary plot to: {shap_summary_path}")
    plt.savefig(shap_summary_path)
    plt.close()


def plot_shap_summary_with_margin(
        shap_values, X, output_dir="output", output_file="shap_summary_with_margin.pdf", top_k=10, save=True, figsize=(12, 8)):
    """
    Plot SHAP summary and save the plot while ensuring no label truncation.

    Parameters:
    - shap_values: SHAP values for the dataset.
    - X: The dataset (Pandas DataFrame or NumPy array).
    - output_dir (str): Directory to save the plot.
    - save (bool): Whether to save the plot.
    - figsize (tuple): Size of the figure.
    """
    shap.summary_plot(shap_values, X, show=False, plot_size=figsize, max_display=top_k)
    
    # Adjust layout to prevent cutoff
    plt.gcf().tight_layout()  # Automatically adjusts margins
    plt.subplots_adjust(left=0.25)  # Add extra space on the left for long labels

    if save:
        os.makedirs(output_dir, exist_ok=True)
        shap_summary_path = os.path.join(output_dir, output_file)
        plt.savefig(shap_summary_path, bbox_inches="tight")  # Ensure the entire figure is saved
        print(f"SHAP summary plot saved to: {shap_summary_path}")
        plt.close()
    else: 
        plt.show()


def quantify_feature_importance_via_shap_analysis(
    model,
    X_test, *, 
    output_dir,
    top_k,
    top_n_features,
    suffix="",
    subject="xgboost",
    verbose=1,
    return_scores=False,
    
    # NEW: For local aggregator
    collect_local_shap=False,      # If True, gather top K local features per sample
    local_top_k=25,               # Number of local features to store (like local summary)
    local_shap_out=None,          # If provided, a dict or structure to store local shap results
    
    # NEW: For motif vs. non‐motif logic
    separate_motif_features=True, 
    # If True, we do an extra step to separate “motif_.*” from others and add two more plots
    
    motif_pattern=r'^\d+mer_.*',
    **kargs
):
    """
    Perform SHAP analysis, produce standard SHAP plots, and also build a 
    local top-K aggregator that compares error vs. correct samples.
    """
    print(f"Quantifying feature importance via SHAP analysis ...")
    return perform_shap_analysis(   # or plot_and_save_shap_summary(...)
        model=model,
        X_test=X_test,
        output_dir=output_dir,
        top_k=top_k,
        top_n_features=top_n_features,
        suffix=suffix,
        subject=subject,
        verbose=verbose,
        return_scores=return_scores,
        collect_local_shap=collect_local_shap,
        local_top_k=local_top_k,
        local_shap_out=local_shap_out,
        separate_motif_features=separate_motif_features, 
        motif_pattern=motif_pattern,
        **kargs
    )


# Enhanced main SHAP analysis function
def perform_shap_analysis(
    model,
    X_test,
    output_dir,
    top_k,
    top_n_features,
    suffix="",
    subject="xgboost",
    verbose=1,
    return_scores=False,
    collect_local_shap=False,
    local_top_k=25,
    local_shap_out=None,
    separate_motif_features=False,
    motif_pattern=r'^\d+mer_.*',
    **kargs
):
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if verbose:
        print(f"[info] SHAP values shape: {shap_values.shape}")

    base_filename = f"{subject}-{suffix}" if suffix else subject

    # 1) SHAP Summary Plot
    plot_shap_summary_with_margin(
        shap_values, X_test,
        output_dir=output_dir,
        output_file=f"{base_filename}-shap_summary_with_margin.pdf",
        top_k=top_k
    )

    # 2) Global importance barplot (all features)
    feature_importance = np.abs(shap_values).mean(axis=0)
    full_importances = pd.DataFrame({
        'feature': X_test.columns,
        'importance_score': feature_importance
    }).sort_values(by='importance_score', ascending=False).reset_index(drop=True)

    plot_shap_global_importance(
        full_importances,
        title="Global SHAP Feature Importance",
        output_path=os.path.join(output_dir, f"{base_filename}-global_importance-barplot.pdf"),
        top_k=top_k
    )

    # 3 & 4) Motif and Non-motif global importance
    if separate_motif_features:
        motif_df = full_importances[full_importances['feature'].str.match(motif_pattern)]
        nonmotif_df = full_importances[~full_importances['feature'].str.match(motif_pattern)]

        plot_shap_global_importance(
            motif_df,
            title="Motif-specific SHAP Feature Importance",
            output_path=os.path.join(output_dir, f"{base_filename}-motif_importance-barplot.pdf"),
            top_k=top_k
        )

        plot_shap_global_importance(
            nonmotif_df,
            title="Non-motif SHAP Feature Importance",
            output_path=os.path.join(output_dir, f"{base_filename}-nonmotif_importance-barplot.pdf"),
            top_k=top_k
        )

    # 3) Extract top N features
    top_features_df = full_importances.head(top_n_features)
    top_features = top_features_df['feature'].tolist()

    if verbose:
        print(f"[info] Top {top_n_features} features identified:")
        print(top_features)

    # 4) Local SHAP aggregator (if requested)
    local_shap_agg = {}
    if collect_local_shap:
        for i in range(shap_values.shape[0]):
            row_vals = shap_values[i]
            if row_vals.ndim == 2:
                row_vals = row_vals.sum(axis=0)

            abs_vals = np.abs(row_vals)
            top_indices = np.argsort(-abs_vals)[:local_top_k]
            local_top_features = [(X_test.columns[idx], row_vals[idx]) for idx in top_indices]

            if local_shap_out is not None:
                local_shap_out[i] = local_top_features

        if verbose:
            print(f"[info] Collected local top-{local_top_k} features for each sample.")

    if return_scores:
        return top_features, top_features_df, full_importances
    else:
        return top_features


def plot_and_save_shap_summary(
    model,
    X_test,
    output_dir,
    top_k,
    top_n_features,
    suffix="",
    subject="xgboost",
    verbose=1,
    return_scores=False,
    
    # NEW: For local aggregator
    collect_local_shap=False,      # If True, gather top K local features per sample
    local_top_k=25,               # Number of local features to store (like local summary)
    local_shap_out=None,          # If provided, a dict or structure to store local shap results
    
    # NEW: For motif vs. non‐motif logic
    separate_motif_features=False, # If True, we do an extra step to separate “motif_.*” from others
    motif_pattern=r'^\d+mer_.*',
    **kargs
):
    """
    Plot and save SHAP summary, initialize SHAP explainer, calculate SHAP values,
    and select top N important features. Optionally gather local top-K features per sample.

    Parameters
    ----------
    model : object
        Trained model (e.g. XGBoost model) compatible with shap.TreeExplainer.
    X_test : pd.DataFrame
        Test dataset (features only). 
    output_dir : str
        Directory to save the plots.
    top_k : int
        Number of top features to plot in the SHAP summary plots.
    top_n_features : int
        Number of top important features to select in a subset. 
    suffix : str
        Suffix to append to output filenames.
    subject : str
        Prefix/subject for output filenames.
    verbose : int
        Verbosity level. 
    return_scores : bool
        If True, we return not only top_features but also a DataFrame of them 
        plus the full set of sorted features.

    collect_local_shap : bool
        If True, we compute local shap for each row to gather top local features.
        This can be expensive for large datasets.
    local_top_k : int
        Number of local top features to store per sample if collect_local_shap=True.
    local_shap_out : dict or None
        If provided, store local top-K features in this dict for external usage.
        E.g. local_shap_out["row_index"] => list of (feature, shap_value).

    separate_motif_features : bool
        If True, we separately rank motif vs. non‐motif features. This can be 
        useful to highlight overshadowed motif columns. 
    motif_pattern : str (regex)
        The pattern for motif columns (e.g. r'^\d+mer_.*').

    Returns
    -------
    top_features : list[str]
        List of top N important features (by absolute mean SHAP).
    top_features_df : pd.DataFrame (optional)
        DataFrame containing top N (feature,importance).
    full_features_df : pd.DataFrame (optional)
        All features sorted by importance.
    """
    import shap
    import re

    # 1) Produce standard SHAP summary plots
    # output_file = "shap_summary.pdf" if not suffix else f"shap_summary_{suffix}.pdf"
    # if subject:
    #     output_file = f"{subject}-{output_file}"

    # plt.clf()
    # plot_shap_summary(
    #     model, 
    #     X_test, 
    #     output_dir=output_dir, 
    #     top_k=top_k,
    #     plot_name=output_file
    # )

    # 2) Initialize SHAP explainer, compute shap_values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if verbose > 0:
        print(f"[info] shape(shap_values): {shap_values.shape} for X shape={X_test.shape}")

    # 3) An optional second plot with margin
    output_file = "shap_summary_with_margin.pdf" if not suffix else f"shap_summary_with_margin_{suffix}.pdf"
    if subject:
        output_file = f"{subject}-{output_file}"

    plt.clf()
    plot_shap_summary_with_margin(
        shap_values, 
        X_test, 
        output_dir=output_dir, 
        top_k=top_k,
        output_file=output_file
    )

    # 3b) An optional beeswarm
    output_file = "shap_beeswarm.pdf" if not suffix else f"shap_beeswarm_{suffix}.pdf"
    if subject:
        output_file = f"{subject}-{output_file}"
    output_path = os.path.join(output_dir, output_file)

    plot_shap_beeswarm(
        shap_values, 
        X_test, 
        max_display=top_k, 
        output_path=output_path,
        show_plot=False
    )

    # 4) Compute global feature importance by mean(|SHAP|)
    if shap_values.ndim == 2:
        # shape => (samples, features)
        feature_importance = np.abs(shap_values).mean(axis=0)
    elif shap_values.ndim == 3:
        # shape => (samples, classes, features)
        # sum or average across classes
        feature_importance = np.abs(shap_values).mean(axis=0).sum(axis=0)
    else:
        raise ValueError(f"Unexpected shap_values shape: {shap_values.shape}")

    all_features = X_test.columns
    full_importances = pd.DataFrame({
        'feature': all_features,
        'importance_score': feature_importance
    }).sort_values(by='importance_score', ascending=False).reset_index(drop=True)

    # 5) Identify top N
    top_features_df = full_importances.head(top_n_features)
    top_features = top_features_df['feature'].tolist()

    if verbose:
        print(f"[info] Top {top_n_features} important features:\n{top_features}")

    # 6) Optional: separate motif features from non‐motifs
    #    e.g. if you suspect “score” overshadowing your k-mers.
    if separate_motif_features:
        motif_regex = re.compile(motif_pattern)
        # separate them
        motif_df = full_importances[full_importances['feature'].apply(lambda f: bool(motif_regex.match(f)))]
        nonmotif_df = full_importances[~full_importances['feature'].apply(lambda f: bool(motif_regex.match(f)))]
        
        # You could do something like: top_motif = motif_df.head(...) 
        # or compute top 30 motif features, etc.

    # 7) If local aggregator is desired, do it
    if collect_local_shap:
        # For each row i, we can do shap_values[i], 
        # gather top local_top_k features by abs shap, store them. 
        if shap_values.ndim == 2:
            # shap_values => shape (n_samples, n_features)
            for i in range(shap_values.shape[0]):
                row_vals = shap_values[i, :]
                # get top local_top_k
                absvals = np.abs(row_vals)
                idx_sorted = np.argsort(-absvals)  # descending
                top_idx = idx_sorted[:local_top_k]
                
                local_result = []
                for j in top_idx:
                    f_ = X_test.columns[j]
                    val_ = row_vals[j]
                    local_result.append((f_, val_))
                
                if local_shap_out is not None:
                    local_shap_out[i] = local_result
        elif shap_values.ndim == 3:
            # shape => (n_samples, n_classes, n_features)
            # maybe sum across classes for local?
            for i in range(shap_values.shape[0]):
                # sum across classes
                row_vals = shap_values[i].sum(axis=0)  # shape => (features,)
                absvals = np.abs(row_vals)
                idx_sorted = np.argsort(-absvals)
                top_idx = idx_sorted[:local_top_k]

                local_result = []
                for j in top_idx:
                    f_ = X_test.columns[j]
                    val_ = row_vals[j]
                    local_result.append((f_, val_))
                
                if local_shap_out is not None:
                    local_shap_out[i] = local_result

        if verbose:
            print(f"[info] Collected local top-{local_top_k} shap features for each sample.")

    # 8) Return final
    #    Typically, you already had `top_features` as list. 
    #    We extend it so you can optionally get `top_features_df` + `full_importances`
    if return_scores:
        return top_features, top_features_df, full_importances
    else:
        return top_features


def quantify_feature_importance_via_shap_analysis_with_local_agg(
    model,
    X_test,
    y_test, *, 
    output_dir,
    error_label=1,
    correct_label=0,
    local_top_k=10,
    global_top_k=20,
    plot_top_k=15,
    subject="shap-analysis",
    suffix="",
    verbose=1,
    return_all=False
): 
    """
    Perform SHAP analysis, produce standard SHAP plots, and also build a 
    local top-K aggregator that compares error vs. correct samples.
    """
    print(f"Quantifying feature importance via SHAP analysis with local aggregator ...")
    return shap_analysis_with_local_agg_plots(
        model=model,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
        error_label=error_label,
        correct_label=correct_label,
        local_top_k=local_top_k,
        global_top_k=global_top_k,
        plot_top_k=plot_top_k,
        subject=subject,
        suffix=suffix,
        verbose=verbose,
        return_all=return_all
    )


def shap_analysis_with_local_agg_plots(
    model,
    X_test,
    y_test,
    output_dir,
    error_label=1,
    correct_label=0,
    local_top_k=10,
    global_top_k=20,
    plot_top_k=15,
    subject="shap-analysis",
    suffix="",
    verbose=1,
    return_all=False
):
    """
    Perform SHAP analysis, produce standard SHAP plots, and also build a 
    local top-K aggregator that compares error vs. correct samples.

    Parameters
    ----------
    model : trained model (e.g. XGBoost)
    X_test : pd.DataFrame
        Test features (shape: [n_samples, n_features]).
    y_test : array-like of shape (n_samples,)
        Binary labels. error_label=1 => error class, correct_label=0 => correct class.
    output_dir : str
        Directory to save any CSV or output plots.
    error_label : int, default=1
        The label for “error” class (FP or FN).
    correct_label : int, default=0
        The label for “correct” class (TP or TN).
    local_top_k : int, default=10
        Number of top features to consider per example.
    global_top_k : int, default=20
        Number of top features to highlight in the final global ranking CSV.
    plot_top_k : int, default=15
        Number of top features to display in summary plots (beeswarm, etc.).
    subject : str, default="shap-analysis"
        Used in filenames for saving outputs.
    suffix : str, default=""
        Appended to filenames to differentiate runs.
    verbose : int, default=1
        Controls debugging / prints.
    return_all : bool, default=False
        If True, return the global importance DataFrame and local aggregator freq DataFrame as well.

    Returns
    -------
    top_global_features : list of str
        The top “global_top_k” features by mean(|SHAP|).
    (if return_all=True) global_importance_df : pd.DataFrame
    (if return_all=True) local_freq_df : pd.DataFrame

    Notes
    -----
    1) We assume a binary model, so shap_values has shape (n_samples, n_features).
    2) If your model is multiclass, you can adapt by summing or averaging across classes.
       See model_utils.aggregate_shap_values()

    3) This function uses shap’s built-in summary plots. Alternatively, you can 
       re-use your own custom plot_shap_summary / plot_shap_beeswarm, etc.

    4) Global aggregator: 
       The function calculates the absolute value of the SHAP matrix and averages across 
       all samples, producing a global importance metric (i.e. mean|shap_values|) for each feature. 
        - It then sorts these features in descending order of importance and saves them to a CSV
        - Finally, it returns a list top_global_features, which are the top global_top_k features.

    5) Local Aggregator: 
       - The function loops over each sample i in the test set, 
       and for each sample, it calculates the absolute value of the SHAP vector, 
       and then sorts these absolute values in descending order to find the top local_top_k features. 
       - It then counts how often each feature appears in the top local_top_k list for error vs. correct samples. 
       This gives us a frequency count of how often each feature is important for errors vs. corrects.
       - After all rows are processed, it converts these frequency counts into a 
        frequency difference table (local_freq_df), which is saved to a CSV
       - That table shows, for each feature, how often it appeared in the local-top-k sets 
          among error vs. correct samples, and the resulting difference in frequency.
          freq_diff=freq_error−freq_correct
       
    """
    # from .model_utils import aggregate_shap_values
    # from collections import defaultdict

    os.makedirs(output_dir, exist_ok=True)
    # --------------------------------------------------------------------
    # 1) Create and apply shap Explainer
    # --------------------------------------------------------------------
    if verbose:
        print("[info] Building shap TreeExplainer ...")

    explainer = shap.TreeExplainer(model)
    shap_values = raw_shap_values = explainer.shap_values(X_test)  
    # shape => (n_samples, n_features) for binary classification or regression

    if verbose:
        shap_val_type = "list" if isinstance(raw_shap_values, list) else "array"
        print(f"[info] raw_shap_values type={shap_val_type}")

    if verbose:
        print(f"[info] (Raw) shap_values.shape={raw_shap_values.shape}, X_test.shape={X_test.shape}")

    # --------------------------------------------------------------------
    # 2) Plot standard SHAP summary (force plot, bar plot, beeswarm, etc.)
    # shap.summary_plot can produce multiple types
    # --------------------------------------------------------------------
    # (A) summary_plot (type='bar' or type='dot') => top features
    shap.summary_plot(raw_shap_values, X_test, max_display=plot_top_k, show=False)
    plt.title(f"SHAP Summary ({subject})", fontsize=14)
    file_out = os.path.join(output_dir, f"{subject}-shap_summary_bar{suffix}.pdf")
    plt.savefig(file_out, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"[info] Saved SHAP summary bar => {file_out}")

    # (B) beeswarm => often use type='dot'
    shap.summary_plot(raw_shap_values, X_test, plot_type="dot", max_display=plot_top_k, show=False)
    plt.title(f"SHAP Beeswarm ({subject})", fontsize=14)
    file_out = os.path.join(output_dir, f"{subject}-shap_beeswarm{suffix}.pdf")
    plt.savefig(file_out, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"[info] Saved SHAP beeswarm => {file_out}")

    # Additional plots if you want ...
    # e.g. shap.dependence_plot(...) for certain features

    # 2a) Aggregate to 2D shap values if multiclass
    shap_values = aggregate_shap_values(raw_shap_values, aggregation='mean')
    # No-op if already 2D
    if verbose:
        print(f"[info] Aggregated shap_values.shape={shap_values.shape}")

    # --------------------------------------------------------------------
    # 3) Compute global aggregator => mean(|SHAP|)
    # --------------------------------------------------------------------
    abs_shap = np.abs(shap_values)  # shape => (n_samples, n_features)
    global_importance = abs_shap.mean(axis=0)  # shape => (n_features,)
    feature_names = X_test.columns

    global_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_score": global_importance
    }).sort_values(by="importance_score", ascending=False).reset_index(drop=True)

    # top features by mean|SHAP|
    top_global_df = global_importance_df.head(global_top_k)
    top_global_features = top_global_df["feature"].tolist()

    if verbose:
        print(f"[info] Top {global_top_k} global features => {top_global_features}")

    # Save them
    csv_out = os.path.join(output_dir, f"{subject}-global_shap_importance{suffix}.csv")
    global_importance_df.to_csv(csv_out, index=False)
    if verbose:
        print(f"[i/o] Saved global importance => {csv_out}")

    # --------------------------------------------------------------------
    # 4) Local aggregator => local_top_k
    # --------------------------------------------------------------------
    freq_error = defaultdict(int)
    freq_correct = defaultdict(int)
    n_error_samples = 0
    n_correct_samples = 0

    y_test_arr = np.array(y_test)

    for i in range(shap_values.shape[0]):  # shap_values shape => (n_samples, n_features)
        row_label = y_test_arr[i]
        row_shap = shap_values[i,:]
        # pick top local_top_k by absolute shap
        sorted_idx = np.argsort(np.abs(row_shap))[::-1]
        top_idx = sorted_idx[:local_top_k]
        top_features_local = [feature_names[j] for j in top_idx]

        if row_label == error_label:
            n_error_samples += 1
            for f_ in top_features_local:
                freq_error[f_] += 1
        elif row_label == correct_label:
            n_correct_samples += 1
            for f_ in top_features_local:
                freq_correct[f_] += 1
        else:
            # ignore rows that might not fit the binary scheme
            pass

    if verbose:
        print(f"[info] local aggregator done => {n_error_samples} error, {n_correct_samples} correct")

    local_recs = []
    all_feats_local = set(freq_error.keys()) | set(freq_correct.keys())
    for feat in all_feats_local:
        e_count = freq_error[feat]
        c_count = freq_correct[feat]
        e_freq = e_count / (n_error_samples + 1e-9)
        c_freq = c_count / (n_correct_samples + 1e-9)
        diff = e_freq - c_freq  # frequency delta: how often it appears in errors relative to corrects
        local_recs.append((feat, e_freq, c_freq, diff, e_count, c_count))

    local_freq_df = pd.DataFrame(local_recs, columns=[
        "feature", "freq_error","freq_correct","freq_diff","count_error","count_correct"
    ])
    local_freq_df.sort_values("freq_diff", ascending=False, inplace=True, ignore_index=True)

    csv_out = os.path.join(output_dir, f"{subject}-local_top{local_top_k}_freq{suffix}.csv")
    local_freq_df.to_csv(csv_out, index=False)
    if verbose:
        print(f"[i/o] Saved local aggregator => {csv_out}")
        print(local_freq_df.head(10))

    # --------------------------------------------------------------------
    # Return
    # --------------------------------------------------------------------
    if return_all:
        # Return top features (list) plus global_importance_df, local_freq_df
        return top_global_features, global_importance_df, local_freq_df
    else:
        return top_global_features


def compute_gene_level_shap_aggregator(
    model,
    X,
    y,
    gene_ids,
    local_top_k=10,
    error_label=1,
    correct_label=0,
    verbose=1
):
    """
    Compute SHAP-based local-frequency aggregation per gene.

    Parameters
    ----------
    model : trained ML model (e.g., XGBoost)
    X : pd.DataFrame
        Feature set.
    y : pd.Series or np.array
        Binary labels (1=error_label, 0=correct_label).
    gene_ids : pd.Series or np.array
        Gene IDs corresponding to each row in X and y.
    local_top_k : int, default=10
        Number of top features to consider per example.
    error_label : int, default=1
        Label representing errors (e.g., FP or FN).

    Returns
    -------
    gene_aggregator : dict
        Mapping from gene_id to a frequency DataFrame (feature, error_freq, correct_freq, diff).
    
    global_aggregator_df : pd.DataFrame
        Global aggregator across all genes (feature, error_freq, correct_freq, diff).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    gene_aggregators = {}

    unique_genes = np.unique(gene_ids)
    global_counts = {error_label: defaultdict(int), correct_label: defaultdict(int)}
    global_example_counts = {error_label: 0, correct_label: 0}

    for gene_id in unique(gene_ids):
        gene_mask = (gene_ids == gene_id)
        X_gene = X[gene_mask]
        y_gene = y[gene_mask]
        shap_gene = shap_values[gene_mask]

        num_examples_local = {error_label: 0, correct_label: 0}
        top_token_counts_local = {error_label: defaultdict(int), correct_label: defaultdict(int)}

        for i in range(X_gene.shape[0]):
            label = y_gene.iloc[i]
            label_str = error_label if label == 1 else correct_label

            shap_row = shap_gene[i, :]
            top_indices = np.argsort(-np.abs(shap_row))[:local_top_k]
            top_features = X.columns[top_indices]

            num_examples_local[label_str] += 1
            for feat in top_features:
                top_token_counts[label_str][feat] += 1
                # Also update global aggregator
                global_counts[label_str][feat] += 1

        # Compute local frequencies
        local_comparison = build_token_frequency_comparison(
            top_token_counts_local=num_examples_local,
            num_examples=num_examples_local,
            error_label=error_label,
            correct_label=correct_label
        )

        gene_level_results = pd.DataFrame(local_comparison, columns=["feature", "error_freq", "correct_freq", "diff"])
        gene_aggregator[gene_id] = gene_level_results

    # Compute global aggregator (across all genes)
    global_example_counts = {error_label: sum(y==1), correct_label: sum(y==0)}
    top_token_counts_global = {error_label: defaultdict(int), correct_label: defaultdict(int)}

    for i in range(shap_values.shape[0]):
        label_str = error_label if y[i] == 1 else correct_label

        shap_row = shap_values[i, :]
        top_indices = np.argsort(-np.abs(shap_row))[:local_top_k]
        top_features = X.columns[top_indices]

        for tk in top_features:
            top_token_counts[label_str][tk] += 1

    global_comparison = build_token_frequency_comparison(
        top_token_counts=top_token_counts,
        num_examples=num_examples,
        error_label=error_label,
        correct_label=correct_label
    )

    global_aggregator_df = pd.DataFrame(global_comparison, columns=["feature", "error_freq", "correct_freq", "diff"])

    return gene_level_results, global_aggregator_df


def plot_and_save_shap_summary_with_motifs(
        model,
        X_test,
        output_dir,
        top_k,
        top_n_features,
        kmer_pattern=r'^\d+mer_.*',
        top_n_motifs=10,
        return_scores=False,
        **kargs
    ):
    """
    Plot and save SHAP summary, rank top N features, and rank motif-specific features.

    Parameters:
    - model: Trained model.
    - X_test: Test dataset.
    - output_dir (str): Directory to save SHAP plots.
    - top_k (int): Number of top features to plot in the SHAP summary.
    - top_n_features (int): Number of top important features to rank.
    - kmer_pattern (str): Regex pattern for motif-specific features.
    - top_n_motifs (int): Number of top motif-specific features to rank.
    - return_scores (bool): Whether to return feature importance scores.

    Returns:
    - results (dict or DataFrame): 
        If `return_scores` is False, returns a dictionary with feature lists.
        If `return_scores` is True, returns a dictionary with DataFrames including scores.
    """
    verbose = kargs.get('verbose', 1)

    # General SHAP summary and top N features
    top_features_df = full_importances = None
    top_features, *rest = plot_and_save_shap_summary(
        model, 
        X_test, 
        output_dir, 
        top_k=top_k,  # Number of top features to display
        top_n_features=top_k,  # Number of top important features to select 
        return_scores=return_scores, 
        verbose=verbose 
    )

    if return_scores: 
        top_features_df, full_importances = rest

    # Motif-specific SHAP analysis
    top_motifs = rank_motif_features_by_shap(
        model, 
        X_test, 
        kmer_pattern=kmer_pattern, 
        top_n_motifs=top_n_motifs, 
        return_scores=return_scores, 
        verbose=1
    )

    # Return results as DataFrames or feature lists
    if return_scores:
        return {
            "top_features_df": top_features,
            "top_motifs_df": top_motifs,
        }
    return {
        "top_features": top_features,
        "top_motifs": top_motifs,
    }


def quantify_feature_importance_via_shap_analysis_with_motifs(
    model, X_test, *,
    kmer_pattern=r'^\d+mer_.*',
    top_n_motifs=10, 
    verbose=1, 
    return_scores=False, 
    return_full_scores=False
):
    """
    Quantify feature importance via SHAP analysis with motifs.
    """
    print(f"Quantifying feature importance via SHAP analysis with motifs ONLY ...")
    return rank_motif_features_by_shap(
        model=model,
        X_test=X_test,
        kmer_pattern=kmer_pattern,
        top_n_motifs=top_n_motifs,
        verbose=verbose,
        return_scores=return_scores,
        return_full_scores=return_full_scores
    )


def rank_motif_features_by_shap(
    model, X_test, 
    # kmer_pattern=r'^[2345]mer_.*',
    kmer_pattern=r'^\d+mer_.*',
    top_n_motifs=10, 
    verbose=1, 
    return_scores=False, 
    return_full_scores=False
):
    """
    Rank motif-specific features based on SHAP values.

    Parameters:
    - model: Trained XGBoost model or compatible model.
    - X_test: Test dataset (Pandas DataFrame).
    - kmer_pattern (str): Regex pattern to filter motif-specific features (e.g., k-mers).
    - top_n_motifs (int): Number of top motifs to select.
    - verbose (int): Verbosity level.
    - return_scores (bool): If True, return the SHAP-value based importance scores along with the features.
    - return_full_scores (bool): If True, return a DataFrame containing all motif features and their SHAP scores.

    Returns:
    - top_motifs (list): List of top N motif-specific features based on SHAP importance.
    - (optional) top_motifs_df (pd.DataFrame): DataFrame with motif-specific features and SHAP scores.
    """
    import re
    import numpy as np

    # Filter columns matching the motif pattern
    motif_columns = [col for col in X_test.columns if re.match(kmer_pattern, col)]
    if verbose > 0:
        print(f"[info] Found {len(motif_columns)} motif-specific features matching pattern '{kmer_pattern}'.")

    if not motif_columns:
        print("[warn] No motif-specific features found. Check your kmer_pattern or input data.")
        if return_scores: 
            if return_full_scores:
                return (pd.DataFrame(), pd.DataFrame())
            return pd.DataFrame()
        return []

    # Ensure the model variable is an XGBoost model trained with TreeExplainer compatibility.
    # if hasattr(model, 'named_steps') and 'xgboost' in model.named_steps:
    #     model = model.named_steps['xgboost']  # Extract the XGBoost model from the pipeline

    # Compute SHAP values for all features
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test)

    # Extract SHAP values for motif-specific features
    motif_indices = [X_test.columns.get_loc(col) for col in motif_columns]
    motif_shap_values = shap_values[:, motif_indices]

    # Calculate mean absolute SHAP values for motif-specific features
    feature_importance = np.abs(motif_shap_values).mean(axis=0)

    # Create a full DataFrame with all motifs and their scores
    full_motifs_df = pd.DataFrame({
        'motif': motif_columns,
        'importance_score': feature_importance
    }).sort_values(by='importance_score', ascending=False)

    # Rank motifs by importance
    top_indices = np.argsort(feature_importance)[-top_n_motifs:]  # Indices of top N motifs
    top_motifs = [motif_columns[i] for i in top_indices]
    top_motif_scores = feature_importance[top_indices]

    if verbose > 0:
        print(f"[info] Top {top_n_motifs} important motif-specific features: {top_motifs}")

    if return_scores:
        # Create a DataFrame to return for top motifs
        top_motifs_df = pd.DataFrame({
            'motif': top_motifs,
            'importance_score': top_motif_scores
        }).sort_values(by='importance_score', ascending=False)

        if return_full_scores:
            return top_motifs_df, full_motifs_df
        return top_motifs_df

    return top_motifs


def demo_mini_ml_pipeline():
    pass


def demo_to_xy(): 
    # Example data
    data = {
        "gene_id": ["gene1", "gene2", "gene3"],
        "transcript_id": ["tx1", "tx2", "tx3"],
        "position": [100, 200, 300],
        "score": [0.9, 0.85, 0.95],
        "splice_type": ["donor", "acceptor", "donor"],
        "chrom": ["chr1", "chr2", "chr1"],
        "strand": ["+", "-", "+"],
        "has_consensus": [1, 0, 1],
        "gene_type": ["protein_coding", "lncRNA", "protein_coding"],
        "2mer_GT": [5, 3, 7],
        "3mer_GTA": [2, 1, 3],
        "label": [1, 0, 1]
    }

    df = pd.DataFrame(data)

    # Convert to features and labels
    X, y = to_xy(df, dummify=True, verbose=1)

    print("\nFeatures (X):")
    print(X.head())
    print("\nLabels (y):")
    print(y.head())


def demo_xgboost_sklearn_compatibility(): 
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.datasets import make_classification

    class CompatibleXGBClassifier(XGBClassifier, ClassifierMixin):
        def __sklearn_tags__(self):
            return {
                "non_deterministic": True,  # XGBoost may produce non-deterministic results
                "requires_positive_X": False,
                "poor_score": False,
                "allow_nan": True,  # XGBoost can handle NaN values
            }

    # Example Dataset
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    # Define model and cross-validation
    model = CompatibleXGBClassifier()
    skf = StratifiedKFold(n_splits=5)

    # Cross-validation predictions
    y_proba_cv = cross_val_predict(model, X, y, cv=skf, method="predict_proba")
    print("Cross-validation successful!")


def demo_analyze_error_classifier_dataset(experiment='hard_genes', pred_type='FP', **kargs): 
    from meta_spliceai.splice_engine.analysis_utils import (
        classify_features, 
        count_unique_ids,
        subset_non_motif_features
    )
    from meta_spliceai.splice_engine.seq_model_utils import (
        verify_sequence_lengths, 
    )
    from meta_spliceai.splice_engine.data_loader import (
        load_error_classifier_dataset
    )

    col_label = kargs.get('col_label', 'label')
    col_tid = kargs.get('col_tid', 'transcript_id')
    verbose = kargs.get('verbose', 1)
    subject = kargs.get('subject', "analysis_sequences_sampled")

    print_emphasized("[info] Loading error classifier dataset ...")
    df_trainset = load_error_classifier_dataset(pred_type=pred_type, verbose=0)
    print_with_indent(f"Columns: {display_feature_set(df_trainset, max_kmers=100)}", indent_level=1)

    # feature_categories = classify_features(df, label_col=col_label, verbose=verbose)
    print_emphasized("[info] Analyzing feature categories and take non-motif features ...")
    df_non_motif = subset_non_motif_features(df_trainset, label_col=col_label, max_unique_for_categorical=10)
    print_with_indent(f"Columns: {list(df_non_motif.columns)}", indent_level=1)

    print_emphasized("[info] Loading analysis sequence dataset ...")
    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    # splice_pos_df = mefd.load_splice_positions(aggregated=True)

    # NOTE: Use run_spliceai_workflow.error_analysis_workflow() to generate the error analysis data 
    analysis_sequence_df = mefd.load_analysis_sequences(aggregated=True, subject=subject)  # Output is a Polars DataFrame

    # Verify sequence lengths
    verify_sequence_lengths(analysis_sequence_df, sequence_col='sequence')

    result = count_unique_ids(df, col_gid='gene_id', col_tid=col_tid, verbose=1, return_ids=False)

    print_with_indent(f"[info] Number of unique gene IDs: {result['num_unique_genes']}", indent_level=1)
    print_with_indent(f"[info] Number of unique transcript IDs: {result['num_unique_transcripts']}", indent_level=1)
    print_with_indent(f"[info] Shape of the analysis sequence DataFrame: {analysis_sequence_df.shape}", indent_level=2)
    # NOTE: Total set with Ensembl: (2,915,508, 11) ~ 3M rows, 11 columns
    print_with_indent(f"[info] Columns: {list(analysis_sequence_df.columns)}", indent_level=1)


def workflow_train_error_classifier(experiment='hard_genes', **kargs):
    from itertools import product

    enable_check_existing = kargs.get('enable_check_existing', False)

    # output_dir = os.path.join(ErrorAnalyzer.analysis_dir, experiment) if experiment else ErrorAnalyzer.analysis_dir
    model_type = kargs.get('model_type', 'xgboost')
    analyzer = ErrorAnalyzer(experiment=experiment, model_type=model_type.lower())
    
    # Define possible labels
    error_labels = ["FP", "FN"]  
    correct_labels = ["TP", "TN"]  

    # Iterate through all combinations, excluding (FP, TN)
    for splice_type in ["any", "donor", "acceptor"]:
        
        for error_label, correct_label in product(error_labels, correct_labels):
            if (error_label, correct_label) == ("FP", "TN"):
                continue  # Skip the unwanted combination

            print_emphasized(f"[info] Training error classifier: {error_label} vs {correct_label} ...")
            print_with_indent(f"Experiment: {experiment}", indent_level=1)
            print_with_indent(f"Splice type: {splice_type}", indent_level=1)

            output_dir = \
                analyzer.set_analysis_output_dir(
                    error_label=error_label, 
                    correct_label=correct_label, 
                    splice_type=splice_type)
            os.makedirs(output_dir, exist_ok=True)
            print_with_indent(f"[demo] Output directory set to: {output_dir}", indent_level=1)

            # ----------------------------------------------------------------
            # Define a unique dummy file name for the combination
            # dummy_file_pattern = re.compile(rf'^{splice_type}_{error_label}_vs_{correct_label}_done\.txt$')
            dummy_file_name = f"{splice_type}_{error_label}_vs_{correct_label}_done.txt"
            dummy_file_path = os.path.join(output_dir, dummy_file_name)

            # Check if the dummy file exists
            if enable_check_existing: 
                if os.path.exists(dummy_file_path):
                    print(f"[info] Skipping {splice_type}: {error_label} vs {correct_label} as it is already processed.")
                    continue
            else: 
                # Clean up the dummy status files
                for file in os.listdir(output_dir): 
                    # Check if the file name follows the expected format
                    if file.startswith(f"{splice_type}_") and file.endswith("_done.txt"):
                        os.remove(os.path.join(output_dir, file))
                        print(f"[info] Removed dummy file: {file}")
            # ----------------------------------------------------------------

            for remove_strong_predictors in [False, ]:  # [True, False]:
                print_emphasized(f"[info] Training error classifier ...")
                print_with_indent(f"Experiment: {experiment}", indent_level=1)
                print_with_indent(f"Error label: {error_label}", indent_level=1)
                print_with_indent(f"Correct label: {correct_label}", indent_level=1)
                print_with_indent(f"[i/o] Output directory: {output_dir}", indent_level=1)
                print_with_indent(f"Remove strong predictors: {remove_strong_predictors}", indent_level=1)

                model, result_set = \
                    train_error_classifier(
                        pred_type=error_label, 
                        remove_strong_predictors=remove_strong_predictors, 
                        output_dir=output_dir, 

                        error_label=error_label, 
                        correct_label=correct_label,
                        splice_type=splice_type,
                        model_type=model_type,
                        verbose=1
                    )

            # Create the dummy file to mark completion
            with open(dummy_file_path, 'w') as f:
                f.write(f"Completed processing {splice_type}: {error_label} vs {correct_label}")

        
def demo_train_error_classifier(experiment='hard_genes', pred_type='FP', **kargs):
    
    # output_dir = os.path.join(ErrorAnalyzer.analysis_dir, experiment) if experiment else ErrorAnalyzer.analysis_dir
    analyzer = ErrorAnalyzer(experiment=experiment)

    model_type = kargs.get('model_type', 'xgboost')
    correct_label = kargs.get('correct_label', 'TP')
    error_label = kargs.get("error_label", pred_type)

    output_dir = analyzer.set_analysis_output_dir(pred_type=pred_type, model_type=model_type.lower())
    os.makedirs(output_dir, exist_ok=True)
    print_with_indent(f"[demo] Output directory set to: {output_dir}", indent_level=1)

    for remove_strong_predictors in [False, ]:  # [True, False]:
        print_emphasized(f"[info] Training error classifier ...")
        print_with_indent(f"Experiment: {experiment}", indent_level=1)
        print_with_indent(f"Pred type: {pred_type}", indent_level=1)
        print_with_indent(f"[i/o] Output directory: {output_dir}", indent_level=1)
        print_with_indent(f"Remove strong predictors: {remove_strong_predictors}", indent_level=1)

        train_error_classifier(
            pred_type=pred_type, 
            remove_strong_predictors=remove_strong_predictors, 
            output_dir=output_dir, 

            error_label=error_label, 
            correct_label=correct_label,
            model_type=model_type,
            verbose=1
        )


def demo_make_training_set(n_genes=1000, **kargs):
    from meta_spliceai.splice_engine.splice_error_analyzer import workflow_training_data_generation

    custom_genes = kargs.get('custom_genes', ['ENSG00000104435', 'ENSG00000130477'])  # STMN2, UNC13A

    workflow_training_data_generation(
        strategy='average', 
        threshold=0.5, 
        kmer_sizes=[6, ], 
        custom_genes=custom_genes, 
        n_genes=n_genes, 
        subset_genes=True, 
        subset_policy='hard'
    )


def demo(): 
    from itertools import product
    

    # Creating featurized training dataset based on gene selection criteria (e.g. hard genes)
    # from meta_spliceai.splice_engine.splice_error_analyzer import demo_make_training_set
    # demo_make_training_set(n_genes=1000)

    # demo_to_xy()

    # Test XGBoost 
    # demo_xgboost_sklearn_compatibility()
    
    # Load error classifier training data and analyze features and labels
    # demo_analyze_error_classifier_dataset(experiment='hard_genes', pred_type=pred_type)

    # Training error classifier
    # demo_train_error_classifier(experiment='hard_genes', pred_type=pred_type)

    # Workflow to train error classifier
    workflow_train_error_classifier(experiment='hard_genes', model_type='xgboost')


if __name__ == "__main__":
    demo()