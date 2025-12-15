import os
import pandas as pd

workbooks = ["any", "donor", "acceptor"]
model_types = ["fn_vs_tp", "fp_vs_tp", "fn_vs_tn"]

descriptions_map = {
    "feature-distributions.pdf": (
        "Distribution plots comparing feature values between the two classes "
        "(e.g., FN vs. TP)."
    ),
    "feature-importance-comparison.pdf": (
        "Visual comparison of feature importance metrics (e.g., SHAP vs. "
        "XGBoost gain)."
    ),
    "global_importance-barplot.pdf": (
        "Barplot of globally aggregated feature importance scores across the dataset."
    ),
    "global_shap_importance-meta.csv": (
        "Table containing aggregated global SHAP importance scores per feature."
    ),
    "local_top25_freq-meta.csv": (
        "Table listing top-25 most frequently locally important features "
        "(local SHAP) per sample."
    ),
    "local-shap-frequency-comparison-meta.pdf": (
        "Plot comparing frequency of local SHAP importance of features between "
        "the two classes. Highlights context-specific feature influence."
    ),
    "motif_importance-barplot.pdf": (
        "Barplot showing importance rankings specifically for motif-related features."
    ),
    "nonmotif_importance-barplot.pdf": (
        "Barplot showing importance rankings for non-motif genomic features "
        "(e.g., number of exons, splice site scores)."
    ),
    "shap_beeswarm-meta.pdf": (
        "Beeswarm plot showing SHAP values for top features at the individual "
        "prediction level."
    ),
    "shap_summary_bar-meta.pdf": (
        "Barplot summarizing average SHAP values per feature."
    ),
    "shap_summary_with_margin.pdf": (
        "Enhanced SHAP summary plot including margin information to clarify "
        "class distinctions (FN vs. TP, etc.)."
    ),
    "xgboost-effect-sizes-barplot.pdf": (
        "Ranked barplot of feature importance derived from calculated effect sizes "
        "(e.g., Cohen’s d)."
    ),
    "xgboost-effect-sizes-results.tsv": (
        "Table with quantitative effect size measurements per feature (e.g., Cohen’s d)."
    ),
    "xgboost-hypo-testing-barplot.pdf": (
        "Barplot ranking features based on importance scores derived from hypothesis "
        "testing (p-values transformed to -log10 scale)."
    ),
    "xgboost-hypo-testing-results.tsv": (
        "Raw results from statistical hypothesis tests (Mann-Whitney U, etc.), including "
        "test statistics, p-values, and significance after BH correction."
    ),
    "xgboost-importance-effect-sizes-full.tsv": (
        "Full table combining all features with their effect-size measurements "
        "(e.g., Cohen’s d) for XGBoost-based analysis."
    ),
    "xgboost-importance-hypo-testing-full.tsv": (
        "Full table combining all features with hypothesis-testing-based feature importance scores for XGBoost-based analysis."
    ),
    "xgboost-importance-hypo-testing.tsv": (
        "Summary table of hypothesis testing-based feature importance scores retaining only the top K features."
    ),
    "xgboost-importance-shap-full.tsv": (
        "Full table merging all features with their SHAP importance values for XGBoost-based analysis."
    ),
    "xgboost-importance-shap.tsv": (
        "Summary table of XGBoost-based features with SHAP importance values."
    ),
    "xgboost-importance-total_gain-barplot.pdf": (
        "Barplot ranking features by XGBoost ‘total gain’ metric."
    ),
    "xgboost-importance-weight-barplot.pdf": (
        "Barplot ranking features by XGBoost ‘weight’ (frequency of splits involving the feature)."
    ),
    "xgboost-importance-xgboost-total_gain-full.tsv": (
        "Full table capturing all features with XGBoost total gain metrics."
    ),
    "xgboost-importance-xgboost-total_gain.tsv": (
        "Summary table of XGBoost total gain metrics per feature."
    ),
    "xgboost-importance-xgboost-weight-full.tsv": (
        "Full table capturing all features with XGBoost weight metrics."
    ),
    "xgboost-importance-xgboost-weight.tsv": (
        "Summary table of XGBoost weight metrics per feature."
    ),
    "xgboost-motif-importance-shap-full.tsv": (
        "Full table capturing motif-specific features along with SHAP importance values."
    ),
    "xgboost-motif-importance-shap.tsv": (
        "Summary table for motif-specific SHAP importance values."
    ),
    "xgboost-PRC-CV-5folds.pdf": (
        "5-fold cross-validated Precision-Recall curve for XGBoost model performance."
    ),
    "xgboost-PRC-CV.pdf": (
        "Cross-validated Precision-Recall curve for XGBoost model performance."
    ),
    "xgboost-prc.pdf": (
        "Precision-Recall curve for XGBoost model performance on a train/test split."
    ),
    "xgboost-ROC-CV-5folds.pdf": (
        "5-fold cross-validated ROC curve for XGBoost model performance."
    ),
    "xgboost-ROC-CV.pdf": (
        "Cross-validated ROC curve for XGBoost model performance."
    ),
    "xgboost-roc.pdf": (
        "ROC curve for XGBoost model performance on a train/test split."
    ),
}


for splice_type in workbooks:
    excel_writer = pd.ExcelWriter(f"{splice_type}.xlsx", engine="xlsxwriter")

    for model_type in model_types:
        # Gather files from directory e.g. f"{splice_type}/{model_type}"
        folder_path = f"{splice_type}/{model_type}"
        if not os.path.isdir(folder_path):
            continue

        rows = []
        for fname in sorted(os.listdir(folder_path)):
            # build a row with the filename and the matched description
            file_key = fname.replace(f"{splice_type}_{model_type}-", "")
            # or do a more robust match for the suffix
            desc = descriptions_map.get(file_key, "No description yet") 
            rows.append({"Filename": fname, "Description": desc})

        df_sheet = pd.DataFrame(rows)
        df_sheet.to_excel(excel_writer, sheet_name=model_type, index=False)

    excel_writer.save()
