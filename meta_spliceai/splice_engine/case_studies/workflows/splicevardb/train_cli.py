
#!/usr/bin/env python3

import os, sys, json, yaml, pathlib, argparse
from typing import Dict, Any, List
import pandas as pd

try:
    import polars as pl
except Exception:
    pl = None

# Optional deps: install as needed
try:
    import xgboost as xgb
except Exception:
    xgb = None

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_table(cfg: Dict[str, Any]) -> pd.DataFrame:
    p = cfg["dataset"]["paths"]["features_parquet"]
    if p.endswith(".parquet"):
        # Prefer Polars if available for speed
        if pl is not None:
            return pl.read_parquet(p).to_pandas()
        return pd.read_parquet(p)
    elif p.endswith(".csv") or p.endswith(".tsv"):
        sep = "," if p.endswith(".csv") else "\t"
        return pd.read_csv(p, sep=sep)
    else:
        raise ValueError(f"Unsupported table format: {p}")

def apply_sample_weights(df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    weights = np.full(len(df), cfg["sample_weights"].get("default", 1.0), dtype=float)
    rules = cfg["sample_weights"].get("rules", [])
    for rule in rules:
        when = rule.get("when", {})
        mask = np.ones(len(df), dtype=bool)
        for k, v in when.items():
            mask &= (df.get(k) == v)
        weights[mask] = rule.get("weight", 1.0)
    # weak class policy downweight/holdout handled in splitting; just return weights here
    return weights

def make_splits(df: pd.DataFrame, cfg: Dict[str, Any]):
    group_col = cfg["splits"]["group_col"]
    train_frac = cfg["splits"]["train_frac"]
    valid_frac = cfg["splits"]["valid_frac"]
    test_frac  = cfg["splits"]["test_frac"]

    gss = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=42)
    train_idx, hold_idx = next(gss.split(df, groups=df[group_col]))
    df_train = df.iloc[train_idx].copy()
    df_hold  = df.iloc[hold_idx].copy()

    # split hold into valid/test
    gss2 = GroupShuffleSplit(n_splits=1, train_size=valid_frac/(valid_frac+test_frac), random_state=43)
    valid_idx, test_idx = next(gss2.split(df_hold, groups=df_hold[group_col]))
    df_valid = df_hold.iloc[valid_idx].copy()
    df_test  = df_hold.iloc[test_idx].copy()
    return df_train, df_valid, df_test

def build_preprocess(cfg: Dict[str, Any], df: pd.DataFrame) -> ColumnTransformer:
    cols = cfg["columns"]
    num_cols = [c for c in cols.get("features_numeric", []) if c in df.columns]
    cat_cols = [c for c in cols.get("features_categorical", []) if c in df.columns]

    num_impute = cfg["transforms"]["numeric"].get("impute", {}).get("strategy", "median")
    scaler = cfg["transforms"]["numeric"].get("scale", "standard")
    from sklearn.impute import SimpleImputer
    num_steps = [("impute", SimpleImputer(strategy=num_impute))]
    if scaler == "standard":
        num_steps.append(("scale", StandardScaler()))
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        num_steps.append(("scale", MinMaxScaler()))

    cat_encoder = OneHotEncoder(handle_unknown=cfg["transforms"]["categorical"].get("handle_unknown","ignore"))
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", cat_encoder, cat_cols)
        ],
        remainder="drop"
    )
    return pre, num_cols, cat_cols

def choose_model(cfg: Dict[str, Any]):
    mtype = cfg["model"]["type"]
    if mtype == "xgboost":
        if xgb is None:
            raise RuntimeError("xgboost is not installed. Install with: pip install xgboost")
        model = xgb.XGBClassifier(
            **cfg["model"]["params"],
            n_jobs=0,
            tree_method="hist"
        )
        return model
    elif mtype == "logistic_regression":
        return LogisticRegression(max_iter=5000, class_weight=cfg["model"].get("class_weighting"))
    else:
        raise ValueError(f"Unsupported model type: {mtype}")

def evaluate(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = float("nan")
    out["average_precision"] = average_precision_score(y_true, y_prob)
    out["f1"] = f1_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred)
    out["recall"] = recall_score(y_true, y_pred)
    return out

def run(cfg_path: str, artifacts_dir: str):
    cfg = load_cfg(cfg_path)
    os.makedirs(artifacts_dir, exist_ok=True)

    df = load_table(cfg)

    # Label mapping
    label_col = cfg["labeling"]["label_col"]
    mapping = cfg["labeling"]["mapping"]
    weak = set(cfg["labeling"].get("weak_classes", []))

    # filter out weak if policy=holdout
    if cfg["labeling"].get("weak_policy","holdout") == "holdout":
        df = df[~df[label_col].isin(weak)].copy()

    df["label"] = df[label_col].map(mapping).astype(int)
    weights = apply_sample_weights(df, cfg)

    # splits
    df_train, df_valid, df_test = make_splits(df, cfg)

    # columns
    cols = cfg["columns"]
    X_train = df_train[cols["features_numeric"] + cols["features_categorical"]].copy()
    y_train = df_train["label"].values
    w_train = weights[df_train.index]

    X_valid = df_valid[cols["features_numeric"] + cols["features_categorical"]].copy()
    y_valid = df_valid["label"].values

    X_test  = df_test[cols["features_numeric"] + cols["features_categorical"]].copy()
    y_test  = df_test["label"].values

    pre, num_cols, cat_cols = build_preprocess(cfg, df)
    model = choose_model(cfg)

    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(X_train, y_train, clf__sample_weight=w_train)

    # evaluation
    def predict_proba(pipe, X):
        try:
            return pipe.predict_proba(X)[:,1]
        except Exception:
            # some models may not implement predict_proba; fallback to decision_function
            from sklearn.preprocessing import MinMaxScaler
            s = pipe.decision_function(X).reshape(-1,1)
            return MinMaxScaler().fit_transform(s).ravel()

    ypv = predict_proba(pipe, X_valid)
    ypt = predict_proba(pipe, X_test)

    ev_valid = evaluate(y_valid, ypv)
    ev_test  = evaluate(y_test, ypt)

    # save artifacts
    joblib.dump(pipe, os.path.join(artifacts_dir, "model.joblib"))
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump({"valid": ev_valid, "test": ev_test}, f, indent=2)

    # simple slice report by region_class if present
    if "region_class" in df.columns:
        import pandas as pd
        df_test_local = df_test.copy()
        df_test_local["y_prob"] = ypt
        slices = df_test_local.groupby("region_class", dropna=False)
        slicerep = {}
        for name, grp in slices:
            if len(grp) < 5: 
                continue
            slicerep[str(name)] = evaluate(grp["label"].values, grp["y_prob"].values)
        with open(os.path.join(artifacts_dir, "slice_metrics.json"), "w") as f:
            json.dump(slicerep, f, indent=2)

    print("Training complete.")
    print("Valid:", ev_valid)
    print("Test:", ev_test)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="features.yaml")
    ap.add_argument("--artifacts", default="artifacts/splicevardb_meta/")
    args = ap.parse_args()
    run(args.config, args.artifacts)
