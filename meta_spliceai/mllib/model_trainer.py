import os
import re
from pathlib import Path
import polars as pl
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Module dependencies
# meta_spliceai/
# ├── mllib/
# │   ├── model_tracker.py
# │   └── model_trainer.py  # The current file

# from meta_spliceai.mllib.experiment_tracker import ExperimentTracker

# from .model_tracker import ModelTracker
from meta_spliceai.mllib.model_tracker import ModelTracker

# from . import model_selection as ms
from meta_spliceai.mllib import model_selection as ms
from meta_spliceai.mllib import plot_performance_curve as ppc

# from meta_spliceai.sphere_pipeline.data_model import (
#     DataSource, 
#     TranscriptIO, 
#     SequenceDescriptor, 
#     Sequence, 
#     Concept
# )


def to_xy_v0(df, dummify=True, drop_first=False, verbose=1, **kargs):
    """
    Convert a DataFrame into features (X) and labels (y), supporting both Pandas and Polars DataFrames.

    Parameters:
    ----------
    df : DataFrame
        The input DataFrame (Pandas or Polars).
    dummify : bool
        Whether to one-hot encode categorical variables.
    drop_first : bool
        Whether to drop the first category in one-hot encoding.
    verbose : int
        Verbosity level for printing progress.

    Returns:
    -------
    X : DataFrame
        The feature DataFrame (with categorical variables encoded if applicable).
    y : Series or DataFrame
        The label column.
    """
    import pandas as pd
    import polars as pl
    
    # Ensure DataFrame is Pandas or Polars
    if isinstance(df, pl.DataFrame):
        is_polars = True
        df = df.to_pandas()
    elif isinstance(df, pd.DataFrame):
        is_polars = False
    else:
        raise TypeError("df must be a Pandas or Polars DataFrame.")

    # Extract configuration
    label_col = kargs.get("label_col", "label")
    categorical_vars = kargs.get("categorical_vars", ['splice_type', 'chrom', 'strand', 'has_consensus', 'gene_type'])
    numerical_vars = kargs.get("numerical_vars", [
        'position', 'score', 'gc_content', 'sequence_complexity', 'sequence_length',
        'transcript_length', 'tx_start', 'tx_end', 'gene_start', 'gene_end',
        'gene_length', 'num_exons', 'avg_exon_length', 'median_exon_length',
        'total_exon_length', 'total_intron_length', 'n_splice_sites', 'num_overlaps',
        'absolute_position', 'distance_to_start', 'distance_to_end'
    ])
    motif_vars_prefix = kargs.get("motif_vars_prefix", ['2mer_', '3mer_'])

    # Separate the label column
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' is missing in the input DataFrame.")
    y = df[label_col].reset_index(drop=True)
    df = df.drop(columns=[label_col])

    # Identify motif variables
    motif_vars = [col for col in df.columns if any(col.startswith(prefix) for prefix in motif_vars_prefix)]

    # Collect all feature columns
    feature_cols = numerical_vars + motif_vars + categorical_vars
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Subset the DataFrame
    X = df[feature_cols].reset_index(drop=True)

    # One-hot encode categorical variables
    if dummify:
        if verbose:
            print(f"[info] One-hot encoding categorical variables: {categorical_vars}")
        X = get_dummies_and_verify(X, categorical_vars=categorical_vars, drop_first=drop_first, verbose=verbose)

    # Replace NaN values with 0
    X = X.fillna(0)

    # Convert back to Polars if the input was Polars
    if is_polars:
        X = pl.DataFrame(X)
        y = pl.Series(label_col, y)

    return X, y


def classify_features(df, label_col='label', max_unique_for_categorical=10):
    """
    Classify features into categories: motif, ID, label, categorical, and numerical.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - label_col (str): The column name for the label/class (default is 'label').
    - max_unique_for_categorical (int): Threshold for the maximum number of unique values
      to classify a numerical column as categorical (default is 10).

    Returns:
    - dict: A dictionary with feature categories.
    """
    
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    feature_categories = {
        'motif_features': [col for col in df.columns if re.match(r'^\d+mer_.*', col)],
        'id_columns': [col for col in df.columns if col in ['gene_id', 'transcript_id']],
        'class_labels': [label_col] if label_col in df.columns else [],
        'categorical_features': [],
        'numerical_features': [],
        'derived_categorical_features': []  # To store numericals classified as categorical due to unique value threshold
    }

    for col in df.columns:
        if col in feature_categories['motif_features'] or col in feature_categories['id_columns'] or col == label_col:
            continue

        col_data = df[col]
        unique_values = set(col_data.dropna().unique())

        if isinstance(col_data.dtype, pd.CategoricalDtype) or col_data.dtype == 'object' or unique_values <= {True, False}:
            feature_categories['categorical_features'].append(col)
        elif pd.api.types.is_numeric_dtype(col_data):
            if len(unique_values) <= max_unique_for_categorical:
                feature_categories['derived_categorical_features'].append(col)
            else:
                feature_categories['numerical_features'].append(col)

    return feature_categories


def to_xy(df, dummify=True, drop_first=False, verbose=1, **kargs):
    """
    Convert a DataFrame into features (X) and labels (y), supporting both Pandas and Polars DataFrames.

    Parameters:
    ----------
    df : DataFrame
        The input DataFrame (Pandas or Polars).
    dummify : bool
        Whether to one-hot encode categorical variables.
    drop_first : bool
        Whether to drop the first category in one-hot encoding.
    verbose : int
        Verbosity level for printing progress.

    Returns:
    -------
    X : DataFrame
        The feature DataFrame (with categorical variables encoded if applicable).
    y : Series or DataFrame
        The label column.
    """
    
    # Ensure DataFrame is Pandas or Polars
    if isinstance(df, pl.DataFrame):
        is_polars = True
        df = df.to_pandas()
    elif isinstance(df, pd.DataFrame):
        is_polars = False
    else:
        raise TypeError("df must be a Pandas or Polars DataFrame.")

    # Extract configuration
    label_col = kargs.get("label_col", "label")
    kmer_pattern = kargs.get("kmer_pattern", r'^\d+mer_.*')  # Regex for k-mers

    # Classify features
    feature_categories = kargs.get("feature_categories", classify_features(df, label_col))

    # Extract categorized features
    motif_vars = feature_categories['motif_features']
    categorical_vars = feature_categories['categorical_features'] + feature_categories['derived_categorical_features']
    numerical_vars = feature_categories['numerical_features']

    # Separate the label column
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' is missing in the input DataFrame.")
    y = df[label_col].reset_index(drop=True)
    df = df.drop(columns=[label_col])

    # Collect all feature columns
    feature_cols = numerical_vars + motif_vars + categorical_vars
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Subset the DataFrame
    X = df[feature_cols].reset_index(drop=True)

    # One-hot encode categorical variables
    if dummify and categorical_vars:
        if verbose:
            print(f"[info] One-hot encoding categorical variables: {categorical_vars}")
        X = pd.get_dummies(X, columns=categorical_vars, drop_first=drop_first)

    # Replace NaN values with 0
    X = X.fillna(0)

    # Convert back to Polars if the input was Polars
    if is_polars:
        X = pl.DataFrame(X)
        y = pl.Series(label_col, y)

    return X, y


def get_dummies_and_verify(df, categorical_vars=None, drop_first=False, verbose=1, **kargs):
    """
    Encode categorical variables using pd.get_dummies(), with robust handling for Polars DataFrames.

    Parameters:
    ----------
    df : DataFrame
        The input DataFrame (Pandas or Polars).
    categorical_vars : list
        List of categorical variables to encode. If None, automatically detect.
    drop_first : bool
        Whether to drop the first category in one-hot encoding.
    verbose : int
        Verbosity level for printing progress.

    Returns:
    -------
    DataFrame
        The DataFrame after encoding categorical variables.
    """
    import pandas as pd
    import polars as pl

    # Ensure DataFrame is Pandas
    if isinstance(df, pl.DataFrame):
        is_polars = True
        df = df.to_pandas()
    else:
        is_polars = False

    if categorical_vars is None:
        categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if verbose:
        print(f"[info] Identified categorical variables: {categorical_vars}")

    # Apply pd.get_dummies
    df = pd.get_dummies(df, columns=categorical_vars, drop_first=drop_first)

    if is_polars:
        df = pl.DataFrame(df)

    return df



def mini_ml_pipeline(X, y, model_name='xgboost', concept=None, **kargs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    # from mllib.experiment_tracker import ExperimentTracker
    from .model_tracker import ModelTracker
    from . import model_selection as ms
    from . import plot_performance_curve as ppc

    def show_model_evaluation_results(output_dict): 
        # Show detailed classification performance report
        avg_f1 = output_dict['f1']
        avg_precision = output_dict.get('precision', '?')
        avg_recall = output_dict.get('recall', '?')
        avg_roc_auc = output_dict['auc']
        avg_fpr = output_dict.get('fpr', '?')
        avg_fnr = output_dict.get('fnr', '?')
        avg_feature_importances = output_dict['feature_importance']

        most_common_hyperparams = output_dict['most_common_hyperparams']
        best_scoring_hyperparams = output_dict['best_scoring_hyperparams']
        # NOTE: If search algorithm searches through real intervals for some hyperparameters (e.g. learning rate), 
        #       chances are that the same hyperparameter setting will never be encountered twice, in which case
        #       it makes more sense to choose the setting that led to highest performance score (rather than choosing
        #       the most common setting across CV iterations)
        best_hyperparams = best_scoring_hyperparams

        print(f"[evaluation] model: {model_name}")

        # Displaying the SHAP-based feature importances for the diabetes dataset using SVM with RBF kernel
        feature_importance_df = output_dict['feature_importance_df']

        print(f"... F1 score: {avg_f1}")
        print(f"... recall / sensitivity: {avg_recall} =?= {1-avg_fnr}") # TPR = TP/P = TP/(TP+FN)
        print(f"... TNR / specificity: {1-avg_fpr}")  # TN/N
        print(f"... PPV / precision: {avg_precision}")  # TP/(TP+FP)  
        print(f"... FPR: {avg_fpr}") # fall-out, FP/N
        print(f"... FNR: {avg_fnr}") # miss rate, FN/P
        print(f"-" * 85)
        print(f"... ROCAUC:    {avg_roc_auc}")
        print(f"... feature importance scores:\n{avg_feature_importances}\n")
        print(f"... most common hyperparams:\n{most_common_hyperparams}\n")
        print(f"... highest scoring hyperparams:\n{best_scoring_hyperparams}\n")

        # Model evaluation using other metrics
        # Todo 

        best_model = output_dict['best_model']
        print(f"... hyperp in best model:\n{best_model.get_params()}\n")

    # ----------------------------------------------------
    model_suffix = kargs.get("model_suffix", None) # Supplementary model identifier   

    if concept is None: # Used to distinguish different experimental settings
        ftype = kargs.get("ftype", "seq-featurized")
        labeling_concept = kargs.get("labeling_concept", '')
        random_control = kargs.get("randomize_control_group", False)
    else: 
        ftype = concept.ftype
        labeling_concept = concept.concept
        random_control = concept.randomize_control_group

    # Todo: Merge these parameters to model tracker    
    n_folds = kargs.get("n_folds", 5)
    use_nested_cv = kargs.get("use_nested_cv", False)

    if isinstance(X, pd.DataFrame): 
        feature_names = list(X.columns)
    else: 
        feature_names = kargs.get('feature_names', [])

    print(f"(mini_ml_pipeline) feature names:\n{feature_names[:10]} ...\n")
    output_dict = ms.nested_cv_and_feaure_importance(X, y, 
                    model_name=model_name, 
                    n_folds=n_folds,  # n_folds_outer, n_folds_inner,
                    feature_names=feature_names, 
                    use_nested_cv=use_nested_cv)
    
    show_model_evaluation_results(output_dict)

    best_model = output_dict['best_model'] # best hyperparams set but NOT fitted

    # -------------------------------------------------------
    
    # Organize the experimental results via an experiment tracker
    experiment_id = f"{labeling_concept}-{ftype}" if labeling_concept else ftype
    if concept is not None: 
        experiment_id = concept.generate_experiment_id()
        if not ftype in experiment_id: 
            experiment_id = f"{experiment_id}-{ftype}"
    if model_suffix:
        experiment_id = f"{experiment_id}-{model_suffix}"

    n_classes = len(set(y))

    tracker = kargs.get("tracker", None)
    if tracker is None: 
        experiment_name = kargs.get("experiment", "classifier")
        tracker = ModelTracker(
                    experiment=experiment_name, 
                    model_type='descriptor', 
                    model_name=model_name, 
                    model_suffix=model_suffix)
    assert isinstance(tracker, ModelTracker)

    output_dir = kargs.get("output_dir", tracker.experiment_dir)    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # -------------------------------------------------------

    if n_classes > 2: 
        label_dict = kargs.get("label_names", concept.label_names)
        output_path = os.path.join(output_dir, 
                                   f"confusion_matrix-{experiment_id}.pdf")

        # Plot parameters 
        title = kargs.get("title", 'Normalized Confusion Matrix')
        fig_text = kargs.get("fig_text", None)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

        best_model.fit(X_train, y_train)

        print("[evaluation] Plotting normalized confusion matrix ...")
        ppc.plot_normalized_confusion_matrix(best_model, X_test, y_test, 
                                             label_dict=label_dict, 
                                             output_path=output_path)
        

        # Plot ROC curve in one-vs-all fashion
        print("[evaluation] Plotting performance curves for each class using a one-vs-all approach ...")
        output_file = f"ROC_Curve-CV-OVA-{experiment_id}.pdf"

        # Plot parameters 
        title = kargs.get("title", 'Receiver Operating Characteristic')
        fig_text = kargs.get("fig_text", None)

        # Performance curve
        ppc.plot_roc_curve_cv_multiclass(
            best_model, 
            X, y, 
            n_folds=5, 
            title=title, 
            # fig_text=fig_text,
            class_names=label_dict,
            output_file=output_file, output_dir=output_dir, 
            verbose=1)  # assuming that categorical encoding has been done
        

    else: 
        output_file = f"ROC_Curve-CV-{experiment_id}.pdf"

        # Plot parameters 
        title = kargs.get("title", 'Receiver Operating Characteristic')
        fig_text = kargs.get("fig_text", None)

        # Performance curve
        ppc.plot_roc_curve_cv(
            best_model, 
            X, y, 
            n_folds=5, 
            title=title, 
            fig_text=fig_text,
            output_file=output_file, 
            output_dir=output_dir, 
            verbose=1) # assuming that categorical encoding has been done
        
    ##############################
        
    # Feature importance ranking
    plt.clf()
    feature_importance_df = output_dict['feature_importance_df']
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    ext = "pdf"
    top_n = 20
    sns.barplot(x='Importance', y='Feature', 
                data=feature_importance_df[:top_n], 
                hue='Feature', palette="viridis_r") # other color scheme: "Blues_d"
    # NOTE: legend=False is removed from sns.barplot() 
    #  - Some version of sns.barplot does not support 'legend' 
    #  - AttributeError: Rectangle.set() got an unexpected keyword argument 'legend'
    plt.legend().remove()
    
    plt.title('Feature Importances')
    # plt.show()
    output_file = f"feature_importance-top{top_n}-{experiment_id}.{ext}" 
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)  
    print(f"[output] Saving feature importance (top n={top_n} features) to:\n{output_path}\n")

    print("#" * 80); print()

    return output_dict


def mini_ml_pipeline_v0(X, y, model_name='xgboost', file_id='test', **kargs):

    def show_model_evaluation_results(output_dict): 
        # Show detailed classification performance report
        avg_f1 = output_dict['f1']
        avg_precision = output_dict.get('precision', '?')
        avg_recall = output_dict.get('recall', '?')
        avg_roc_auc = output_dict['auc']
        avg_fpr = output_dict.get('fpr', '?')
        avg_fnr = output_dict.get('fnr', '?')
        avg_feature_importances = output_dict['feature_importance']

        most_common_hyperparams = output_dict['most_common_hyperparams']
        best_scoring_hyperparams = output_dict['best_scoring_hyperparams']
        # NOTE: If search algorithm searches through real intervals for some hyperparameters (e.g. learning rate), 
        #       chances are that the same hyperparameter setting will never be encountered twice, in which case
        #       it makes more sense to choose the setting that led to highest performance score (rather than choosing
        #       the most common setting across CV iterations)
        best_hyperparams = best_scoring_hyperparams

        print(f"[evaluation] model: {model_name}")

        # Displaying the SHAP-based feature importances for the diabetes dataset using SVM with RBF kernel
        feature_importance_df = output_dict['feature_importance_df']

        print(f"... F1 score: {avg_f1}")
        print(f"... recall / sensitivity: {avg_recall} =?= {1-avg_fnr}") # TPR = TP/P = TP/(TP+FN)
        print(f"... TNR / specificity: {1-avg_fpr}")  # TN/N
        print(f"... PPV / precision: {avg_precision}")  # TP/(TP+FP)  
        print(f"... FPR: {avg_fpr}") # fall-out, FP/N
        print(f"... FNR: {avg_fnr}") # miss rate, FN/P
        print(f"-" * 85)
        print(f"... ROCAUC:    {avg_roc_auc}")
        print(f"... feature importance scores:\n{avg_feature_importances}\n")
        print(f"... most common hyperparams:\n{most_common_hyperparams}\n")
        print(f"... highest scoring hyperparams:\n{best_scoring_hyperparams}\n")

        # Model evaluation using other metrics
        # Todo 

        best_model = output_dict['best_model']
        print(f"... hyperp in best model:\n{best_model.get_params()}\n")

    # ----------------------------------------------------
    model_suffix = kargs.get("model_suffix", None) # Supplementary model identifier   

    # Todo: Merge these parameters to model tracker    
    n_folds = kargs.get("n_folds", 5)
    use_nested_cv = kargs.get("use_nested_cv", False)

    if isinstance(X, pd.DataFrame): 
        feature_names = list(X.columns)
    else: 
        feature_names = kargs.get('feature_names', [])

    print(f"(mini_ml_pipeline) feature names:\n{feature_names[:10]} ...\n")
    output_dict = ms.nested_cv_and_feaure_importance(X, y, 
                    model_name=model_name, 
                    n_folds=n_folds,  # n_folds_outer, n_folds_inner,
                    feature_names=feature_names, 
                    use_nested_cv=use_nested_cv)
    
    show_model_evaluation_results(output_dict)

    best_model = output_dict['best_model'] # best hyperparams set but NOT fitted

    # -------------------------------------------------------
    
    # Organize the experimental results via an experiment tracker
    experiment_id = file_id

    n_classes = len(set(y))

    tracker = kargs.get("tracker", None)
    if tracker is None: 
        experiment_name = kargs.get("experiment", "classifier")
        tracker = ModelTracker(experiment=experiment_name, 
                                        model_type='descriptor', 
                                            model_name=model_name, model_suffix=model_suffix)
    assert isinstance(tracker, ModelTracker)

    output_dir = kargs.get("output_dir", tracker.experiment_dir)    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # -------------------------------------------------------

    if n_classes > 2: 
        label_dict = kargs.get("label_names", {0: 'negative', 1: 'positive'})
        output_path = os.path.join(output_dir, 
                                   f"confusion_matrix-{experiment_id}.pdf")

        # Plot parameters 
        title = kargs.get("title", 'Normalized Confusion Matrix')
        fig_text = kargs.get("fig_text", None)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

        best_model.fit(X_train, y_train)

        print("[evaluation] Plotting normalized confusion matrix ...")
        ppc.plot_normalized_confusion_matrix(best_model, X_test, y_test, 
                                             label_dict=label_dict, 
                                             output_path=output_path)
        

        # Plot ROC curve in one-vs-all fashion
        print("[evaluation] Plotting performance curves for each class using a one-vs-all approach ...")
        output_file = f"ROC_Curve-CV-OVA-{experiment_id}.pdf"

        # Plot parameters 
        title = kargs.get("title", 'Receiver Operating Characteristic')
        fig_text = kargs.get("fig_text", None)

        # Performance curve
        ppc.plot_roc_curve_cv_multiclass(best_model, X, y, n_folds=5, 
                    title=title, 
                    # fig_text=fig_text,
                    class_names=label_dict,
                    output_file=output_file, output_dir=output_dir, verbose=1) # assuming that categorical encoding has been done
        

    else: 
        output_file = f"ROC_Curve-CV-{experiment_id}.pdf"

        # Plot parameters 
        title = kargs.get("title", 'Receiver Operating Characteristic')
        fig_text = kargs.get("fig_text", None)

        # Performance curve
        ppc.plot_roc_curve_cv(best_model, X, y, n_folds=5, 
                    title=title, 
                    fig_text=fig_text,
                    output_file=output_file, output_dir=output_dir, verbose=1) # assuming that categorical encoding has been done
        
    ##############################
        
    # Feature importance ranking
    plt.clf()
    feature_importance_df = output_dict['feature_importance_df']
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    ext = "pdf"
    top_n = 20
    sns.barplot(x='Importance', y='Feature', 
                    data=feature_importance_df[:top_n], 
                        hue='Feature', palette="viridis_r", legend=False) # "Blues_d"
    
    plt.title('Feature Importances')
    # plt.show()
    output_file = f"feature_importance-top{top_n}-{experiment_id}.{ext}" 
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)  
    print(f"[output] Saving feature importance (top n={top_n} features) to:\n{output_path}\n")

    print("#" * 80); print()

    return output_dict