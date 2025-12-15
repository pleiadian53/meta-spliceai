import os, sys
import numpy as np
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

from meta_spliceai.mllib import ModelTracker
from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.analyzer import Analyzer

from tqdm import tqdm

# Refactor to utils_plot
import matplotlib.pyplot as plt 


from meta_spliceai.mllib.model_trainer import (
    mini_ml_pipeline,
    get_dummies_and_verify, 
    to_xy, 
)

from .splice_error_analyzer import (
    ErrorAnalyzer, 
)

# ML Libraries
from meta_spliceai.mllib.plot_performance_curve import (
    plot_roc_curve_cv,  # plot_roc_curve_cv(model, X, y, n_folds=10, figsize=None, **kargs)
    plot_prc_curve_cv   # plot_prc_curve_cv(model, X, y, n_folds=10, figsize=None, **kargs)
)
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def train_fp_classifier(**kargs): 

    n_splits = kargs.get('n_splits', 5)
    top_k = kargs.get('top_k', 20)
    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')

    df_trainset = mefd.load_featurized_dataset(aggregated=True)
    print_emphasized(f"Training set: {df_trainset.shape}")
    print_with_indent(f"Columns: {list(df_trainset.columns)}", indent_level=1)

    if isinstance(df_trainset, pl.DataFrame):
        df_trainset = df_trainset.to_pandas()

    X, y = to_xy(df_trainset, dummify=True, verbose=1)

    print("\nFeatures (X):")
    display_dataframe_in_chunks(X.head())
    print("\nLabels (y):")
    print(y.head())

    model, feature_importance = \
        xgboost_pipeline(X, y, output_dir=ErrorAnalyzer.analysis_dir, top_k=top_k, n_splits=n_splits)

    # Additonal plots
    print_emphasized("[info] Generating CV-driven ROCAUC plot ...")
    ext = 'pdf'
    subject = "fp_vs_tp"
    fig_text = "FP-specific error analysis model explanability in ROCAUC"
    plot_roc_curve_cv(
        model, X, y, n_folds=n_splits, figsize=(10, 10), 
        output_dir=ErrorAnalyzer.analysis_dir, 
        output_file=f"{subject}-xgboost-ROC-CV.{ext}",
        fig_text=fig_text, 
        save=True)

    print_emphasized("[info] Generating CV-driven PRC plot ...")
    fig_text = "FP-specific error analysis model explanability in PRC"
    plot_prc_curve_cv(
        model, X, y, n_folds=n_splits, figsize=(10, 10), 
        output_dir=ErrorAnalyzer.analysis_dir, 
        output_file=f"{subject}-xgboost-PRC-CV.{ext}",
        fig_text=fig_text, save=True)

    

def xgboost_pipeline(
    X, y, top_k=10, n_splits=5, random_state=42, 
    output_dir=None, save=True, subject="fp_vs_tp", importance_type="weight", **kargs):
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
    random_state : int
        Seed for reproducibility.

    Returns:
    -------
    None
    """
    verbose = kargs.get('verbose', 1)

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Initialize XGBoost classifier
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
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr = 1 - recall_score(y_test, y_pred, pos_label=0)  # False Positive Rate
    fnr = 1 - recall

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}")

    # ROC Curve
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr_curve, tpr_curve)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_curve, tpr_curve, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.grid()

    if save: 
        ext = 'pdf'
        output_file = kargs.get('output_file', f"{subject}-xgboost-roc.{ext}") 
        output_path = os.path.join(output_dir, output_file)
        plt.savefig(output_path)
        plt.close()
    else: 
        plt.show()

    # PRC Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label=f"PR Curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid()

    if save: 
        ext = 'pdf'
        output_file = kargs.get('output_file', f"{subject}-xgboost-prc.{ext}") 
        output_path = os.path.join(output_dir, output_file)
        plt.savefig(output_path)
        plt.close()
    else: 
        plt.show()

    # Feature Importance
    print_emphasized("[info] Generating feature importance plots ...")
    # feature_importance = pd.DataFrame({
    #     'Feature': X.columns,
    #     'Importance': model.feature_importances_
    # }).sort_values(by="Importance", ascending=False)

    # top_features = feature_importance.head(top_k)

    # plt.figure(figsize=(10, 8))
    # plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
    # plt.xlabel("Importance")
    # plt.title(f"Top {top_k} Features by Importance")
    # plt.gca().invert_yaxis()
    # plt.grid()

    # if save:
    #     ext = 'pdf'
    #     output_file = kargs.get('output_file', f"{subject}-xgboost-feature-importance.{ext}") 
    #     output_path = os.path.join(output_dir, output_file)
    #     plt.savefig(output_path)
    #     plt.close()
    # else:   
    #     plt.show()
    feature_importance = \
        plot_xgboost_feature_importance(model, X, output_dir=output_dir, subject=subject, top_k=top_k, save=True)

    # print("Top Features:")
    # print(top_features)

    # SHAP Summary
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test, show=False)
    # shap_summary_path = os.path.join(output_dir, "shap_summary.pdf")
    # plt.savefig(shap_summary_path)
    # plt.close()

    # Plot and save SHAP summary
    plot_shap_summary(
        model, X_test, 
        output_dir=output_dir, 
        top_k=top_k,
        plot_name="shap_summary.pdf")

    # Initialize SHAP explainer and calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plot_shap_summary_with_margin(
        shap_values, X_test, 
        output_dir=output_dir, 
        top_k=top_k,
        output_file="shap_summary_with_margin.pdf")

    return model, feature_importance


# Feature Importance Plot with Adjusted Margins
def plot_xgboost_feature_importance(model, X, output_dir="output", subject="xgboost", top_k=10, save=True):
    """
    Plot and save the top-k feature importance from an XGBoost model with adjusted margins for labels.

    Parameters:
    - model: Trained XGBoost model.
    - X: DataFrame containing feature columns.
    - output_dir (str): Directory to save the plot.
    - subject (str): Prefix for the output file name.
    - top_k (int): Number of top features to display.
    - save (bool): Whether to save the plot.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Generate feature importance data
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    top_features = feature_importance.head(top_k)
    print_emphasized(f"Top Features:")
    # print_with_indent(top_features, indent_level=1)
    display(top_features)

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
    plt.xlabel("Importance", fontsize=12)
    plt.title(f"Top {top_k} Features by Importance", fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid()

    # Adjust layout to prevent cutoff
    plt.gcf().tight_layout()  # Automatically adjusts margins
    plt.subplots_adjust(left=0.25)  # Increase space on the left for labels

    # Save or display the plot
    if save:
        os.makedirs(output_dir, exist_ok=True)
        ext = 'pdf'
        output_file = f"{subject}-xgboost-feature-importance.{ext}" 
        output_path = os.path.join(output_dir, output_file)
        plt.savefig(output_path, bbox_inches="tight")  # Ensure nothing is cut off
        print(f"Feature importance plot saved to: {output_path}")
        plt.close()
    else:
        plt.show()

    return feature_importance



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


def demo(): 
    # demo_to_xy()

    train_fp_classifier()


if __name__ == "__main__":
    demo()