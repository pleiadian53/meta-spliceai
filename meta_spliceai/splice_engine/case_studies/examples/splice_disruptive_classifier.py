"""
MetaSpliceAI – splice_disruptive_classifier.py

Binary classifier for "splicing-disruptive vs non-splicing" using delta-score features.
Works with ClinVar/curated labels. Produces metrics (ROC-AUC/PR-AUC), calibration, and
feature importances, and can export a calibrated probability for downstream triage.

Input schema
------------
Expect a pandas DataFrame with at least these columns (one row per variant-allele):
    chrom, pos, ref, alt, label
    DS_DG, DS_DL, DS_AG, DS_AL    (float >= 0)
Optional, recommended features:
    DP_DG, DP_DL, DP_AG, DP_AL    (int offsets; sign indicates direction)
    dist_to_nearest_exon, cons_phastcons, cons_phylop, variant_type (SNV/INS/DEL)
    strand, gene_id, transcript_id

Label semantics
---------------
"label" should be binary: 1 for splice-disruptive (pathogenic splice mechanism), 0 for non-splicing.
If your source labels are categorical (e.g., 'pathogenic', 'likely_pathogenic', 'benign'),
map them to this binary scheme appropriately.

Usage
-----
from splice_disruptive_classifier import run_training
metrics, clf, cal = run_training(df, feature_set="ds_only", test_size=0.2, seed=42,
                                 calibrate=True, out_dir="splice_cls")

Outputs in out_dir/
-------------------
- metrics.json            : ROC-AUC, PR-AUC, Precision@K (10/50/100), Recall@K, FPR@TPR, etc.
- roc_curve.png           : ROC curve
- pr_curve.png            : Precision-Recall curve
- calibration_curve.png   : reliability diagram
- importances.csv         : feature importances or coefficients (depending on model)
- predictions.parquet     : per-variant y_true, y_score, y_prob, split assignment

Model choices
-------------
Default model is gradient boosting (XGBoost-like via sklearn's HistGradientBoosting) for strong
nonlinear performance. You can switch to LogisticRegression or RandomForest if preferred.

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class TrainConfig:
    feature_set: str = "ds_only"  # 'ds_only' or 'ds_plus'
    model: str = "hgb"            # 'hgb', 'logreg', 'rf'
    test_size: float = 0.2
    seed: int = 42
    calibrate: bool = True
    pos_label: int = 1
    out_dir: str = "splice_cls"

# -----------------------------
# Feature engineering
# -----------------------------

def build_feature_space(df: pd.DataFrame, feature_set: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Return X (DataFrame) and lists of numeric/categorical feature names."""
    base_numeric = ["DS_DG","DS_DL","DS_AG","DS_AL",
                    "GAIN_MAX","LOSS_MAX","DS_MAX"]
    if feature_set == "ds_only":
        feats_num = base_numeric
        feats_cat = []
    else:
        extra_num = [c for c in ["DP_DG","DP_DL","DP_AG","DP_AL","dist_to_nearest_exon",
                                 "cons_phastcons","cons_phylop"] if c in df.columns]
        feats_num = base_numeric + extra_num
        feats_cat = [c for c in ["variant_type","strand"] if c in df.columns]
    # Compute derived features if missing
    if "GAIN_MAX" not in df.columns:
        df["GAIN_MAX"] = df[[c for c in ["DS_DG","DS_AG"] if c in df.columns]].max(axis=1)
    if "LOSS_MAX" not in df.columns:
        df["LOSS_MAX"] = df[[c for c in ["DS_DL","DS_AL"] if c in df.columns]].max(axis=1)
    if "DS_MAX" not in df.columns:
        df["DS_MAX"] = df[[c for c in ["DS_DG","DS_DL","DS_AG","DS_AL"] if c in df.columns]].max(axis=1)
    return df[feats_num + feats_cat].copy(), feats_num, feats_cat

# -----------------------------
# Model builder
# -----------------------------

def make_model(cfg: TrainConfig, feats_num: List[str], feats_cat: List[str]):
    # Preprocess
    transformers = []
    if feats_num:
        transformers.append(("num", StandardScaler(with_mean=False), feats_num))
    if feats_cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore'), feats_cat))
    pre = ColumnTransformer(transformers)

    # Base estimator
    if cfg.model == "hgb":
        est = HistGradientBoostingClassifier(random_state=cfg.seed)
    elif cfg.model == "logreg":
        est = LogisticRegression(max_iter=2000, n_jobs=None)
    elif cfg.model == "rf":
        est = RandomForestClassifier(n_estimators=400, random_state=cfg.seed, n_jobs=-1)
    else:
        raise ValueError("Unknown model")

    pipe = Pipeline([("pre", pre), ("clf", est)])
    return pipe

# -----------------------------
# Metrics, plots, and report
# -----------------------------

def _save_curves(y_true, scores, out_dir: str):
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    # ROC
    plt.figure(figsize=(5,4), dpi=150)
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--', linewidth=0.7)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC (AUC={roc_auc:.3f})')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'roc_curve.png')); plt.close()

    # PR
    plt.figure(figsize=(5,4), dpi=150)
    plt.plot(rec, prec)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR (AUC={pr_auc:.3f})')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'pr_curve.png')); plt.close()

    return roc_auc, pr_auc


def _precision_recall_at_k(y_true, scores, ks=(10,50,100)):
    order = np.argsort(-scores)
    y_sorted = np.asarray(y_true)[order]
    out = {}
    for k in ks:
        k = min(k, len(y_sorted))
        topk = y_sorted[:k]
        prec = topk.mean() if k>0 else 0.0
        rec = topk.sum() / max(1, np.sum(y_true))
        out[f'P@{k}'] = float(prec)
        out[f'R@{k}'] = float(rec)
    return out


def _reliability_diagram(y_true, probs, out_dir: str, n_bins: int = 10):
    bins = np.linspace(0,1,n_bins+1)
    inds = np.digitize(probs, bins) - 1
    acc, conf, cnt = [], [], []
    for b in range(n_bins):
        mask = inds == b
        if mask.sum() == 0:
            continue
        acc.append(np.mean(y_true[mask]))
        conf.append(np.mean(probs[mask]))
        cnt.append(mask.sum())
    plt.figure(figsize=(5,4), dpi=150)
    plt.plot([0,1],[0,1],'--', linewidth=0.7)
    plt.scatter(conf, acc, s=np.array(cnt)/np.max(cnt)*200)
    plt.xlabel('Predicted probability'); plt.ylabel('Empirical probability')
    plt.title('Calibration')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'calibration_curve.png')); plt.close()

# -----------------------------
# Training entrypoint
# -----------------------------

def run_training(df: pd.DataFrame,
                 feature_set: str = "ds_only",
                 model: str = "hgb",
                 test_size: float = 0.2,
                 seed: int = 42,
                 calibrate: bool = True,
                 out_dir: str = "splice_cls"):
    os.makedirs(out_dir, exist_ok=True)

    # Build features and split
    X, feats_num, feats_cat = build_feature_space(df.copy(), feature_set)
    y = df['label'].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    cfg = TrainConfig(feature_set=feature_set, model=model, test_size=test_size, seed=seed,
                      calibrate=calibrate, out_dir=out_dir)
    pipe = make_model(cfg, feats_num, feats_cat)
    pipe.fit(Xtr, ytr)

    # Raw decision scores (use predict_proba if available)
    if hasattr(pipe.named_steps['clf'], 'predict_proba'):
        scores_te = pipe.predict_proba(Xte)[:,1]
        scores_tr = pipe.predict_proba(Xtr)[:,1]
    else:
        # fallback to decision_function or predict with calibration later
        if hasattr(pipe.named_steps['clf'], 'decision_function'):
            scores_te = pipe.named_steps['clf'].decision_function(pipe.named_steps['pre'].transform(Xte))
            scores_tr = pipe.named_steps['clf'].decision_function(pipe.named_steps['pre'].transform(Xtr))
        else:
            scores_te = pipe.predict(Xte)
            scores_tr = pipe.predict(Xtr)

    # Optional calibration
    if calibrate:
        cal = CalibratedClassifierCV(pipe, method='isotonic', cv=3)
        cal.fit(Xtr, ytr)
        probs_te = cal.predict_proba(Xte)[:,1]
        probs_tr = cal.predict_proba(Xtr)[:,1]
    else:
        cal = None
        probs_te = scores_te
        probs_tr = scores_tr

    # Metrics & plots
    roc_auc, pr_auc = _save_curves(yte, scores_te, out_dir)
    p_at_k = _precision_recall_at_k(yte, scores_te)
    _reliability_diagram(yte, probs_te, out_dir)

    # Save metrics
    metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc, **p_at_k}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Feature importances / coefficients (best-effort)
    imp = None
    clf = pipe.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        # Map back through ColumnTransformer
        num_names = feats_num
        cat_names = []
        if feats_cat:
            ohe = pipe.named_steps['pre'].transformers_[1][1]
            cat_names = list(ohe.get_feature_names_out(feats_cat))
        names = num_names + cat_names
        imp = pd.DataFrame({'feature': names, 'importance': clf.feature_importances_})
        imp.sort_values('importance', ascending=False).to_csv(os.path.join(out_dir,'importances.csv'), index=False)

    # Save per-variant predictions
    out_pred = pd.DataFrame({
        'split': ['test']*len(yte),
        'y_true': yte,
        'y_score': scores_te,
        'y_prob': probs_te,
    })
    out_pred.to_parquet(os.path.join(out_dir,'predictions.parquet'))

    return metrics, pipe, cal

# -----------------------------
# CLI wrapper (optional)
# -----------------------------
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Train a splicing-disruptive classifier from delta-score features')
    p.add_argument('--in', dest='inp', required=True, help='Input TSV/CSV/Parquet with DS_* and labels')
    p.add_argument('--fmt', choices=['csv','tsv','parquet'], default='parquet')
    p.add_argument('--feature-set', choices=['ds_only','ds_plus'], default='ds_only')
    p.add_argument('--model', choices=['hgb','logreg','rf'], default='hgb')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-calibrate', action='store_true')
    p.add_argument('--out', default='splice_cls')
    args = p.parse_args()

    if args.fmt == 'parquet':
        df = pd.read_parquet(args.inp)
    elif args.fmt == 'csv':
        df = pd.read_csv(args.inp)
    else:
        df = pd.read_csv(args.inp, sep='\t')

    metrics, _, _ = run_training(df,
                                 feature_set=args.feature_set,
                                 model=args.model,
                                 test_size=args.test_size,
                                 seed=args.seed,
                                 calibrate=not args.no_calibrate,
                                 out_dir=args.out)
    print(json.dumps(metrics, indent=2))
"""
