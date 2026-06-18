#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_train_ecfp_baseline_models_colab.py

目的:
    FAERS由来アナフィラキシー報告シグナルのmodel_ready CSVを用いて、
    ECFP4ベースラインモデルをGoogle Colab上で構築・評価する。

入力:
    output/gnn_dataset_model_ready_colab_YYYYMMDD_HHMMSS.csv
    または --input で指定したCSV

モデル:
    1. ECFP4 + Logistic regression
    2. ECFP4 + RandomForest
    3. ECFP4 + LightGBM（lightgbmが利用可能な場合）

評価:
    - Repeated stratified K-fold cross-validation
    - ROC-AUC
    - PR-AUC / Average precision
    - Balanced accuracy
    - Sensitivity
    - Specificity
    - Accuracy
    - F1
    - MCC
    - Brier score
    - Fold-wise predictions
    - Label permutation test（任意）

出力:
    output/ecfp_baseline_results_YYYYMMDD_HHMMSS.xlsx
    output/ecfp_baseline_fold_predictions_YYYYMMDD_HHMMSS.csv
    output/ecfp_baseline_model_summary_YYYYMMDD_HHMMSS.csv

実行例:
    python 05_train_ecfp_baseline_models_colab.py \
      --input output/gnn_dataset_model_ready_colab_20260521_085012.csv \
      --output-dir output \
      --n-splits 5 \
      --n-repeats 20 \
      --n-permutations 100

注意:
    本スクリプトは探索的評価用である。
    データ数が小さいため、性能値は分割に依存する。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# 0. Optional dependencies
# ============================================================

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


# ============================================================
# 1. Utilities
# ============================================================

def find_latest_model_ready_csv(output_dir: Path) -> Optional[Path]:
    files = sorted(output_dir.glob("gnn_dataset_model_ready_colab_*.csv"))
    if files:
        return files[-1]
    files = sorted(output_dir.glob("gnn_dataset_model_ready_*.csv"))
    if files:
        return files[-1]
    return None


def sanitize_filename(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def smiles_to_ecfp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> Optional[np.ndarray]:
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is not available. Install with: pip install rdkit")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
        useChirality=use_chirality,
    )
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def make_ecfp_matrix(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    fps = []
    valid_rows = []
    failed = []

    for i, row in df.iterrows():
        smi = row.get(smiles_col)
        fp = smiles_to_ecfp(smi, radius=radius, n_bits=n_bits, use_chirality=use_chirality)
        if fp is None:
            failed.append(i)
        else:
            fps.append(fp)
            valid_rows.append(i)

    if not fps:
        raise RuntimeError("No valid ECFP fingerprints were generated.")

    X = np.vstack(fps).astype(np.float32)
    out_df = df.loc[valid_rows].reset_index(drop=True).copy()

    if failed:
        print(f"[WARN] Failed SMILES rows: {len(failed)}")
        print(df.loc[failed, ["analysis_drug_name", smiles_col]].head(20).to_string(index=False))

    return X, out_df


def get_models(random_state: int = 42) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    models["logistic_l2"] = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=5000,
            random_state=random_state,
        )),
    ])

    models["logistic_l1"] = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            penalty="l1",
            C=0.2,
            solver="liblinear",
            max_iter=5000,
            random_state=random_state,
        )),
    ])

    models["random_forest"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )

    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=-1,
            min_child_samples=5,
            subsample=0.9,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            random_state=random_state,
            objective="binary",
            verbose=-1,
        )

    return models


def predict_proba_positive(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1 / (1 + np.exp(-z))
    pred = model.predict(X)
    return pred.astype(float)


def choose_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        idx = int(np.nanargmax(j))
        t = float(thr[idx])
        if not np.isfinite(t):
            return 0.5
        # 極端な閾値を回避
        return float(min(max(t, 0.01), 0.99))
    except Exception:
        return 0.5


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, float] = {}

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
        out["pr_auc"] = average_precision_score(y_true, y_prob, sample_weight=sample_weight)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    out["accuracy"] = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    out["sensitivity"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0, sample_weight=sample_weight)

    # specificity = recall for negative class
    out["specificity"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0, sample_weight=sample_weight)
    out["precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0, sample_weight=sample_weight)
    out["f1"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0, sample_weight=sample_weight)

    try:
        out["mcc"] = matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)
    except Exception:
        out["mcc"] = np.nan

    try:
        out["brier"] = brier_score_loss(y_true, y_prob, sample_weight=sample_weight)
    except Exception:
        out["brier"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"] = float(tn)
    out["fp"] = float(fp)
    out["fn"] = float(fn)
    out["tp"] = float(tp)
    out["threshold"] = float(threshold)

    return out


def sample_weight_for_model(model: Any, sample_weight: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if sample_weight is None:
        return None

    # Pipelineの場合は末尾のclfへ渡す。
    if isinstance(model, Pipeline):
        last_name = model.steps[-1][0]
        return {f"{last_name}__sample_weight": sample_weight}

    return {"sample_weight": sample_weight}


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "mcc",
        "brier",
    ]
    rows = []
    for model_name, g in metrics_df.groupby("model"):
        for metric in metric_cols:
            x = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(x) == 0:
                rows.append({
                    "model": model_name,
                    "metric": metric,
                    "n": 0,
                    "mean": np.nan,
                    "sd": np.nan,
                    "median": np.nan,
                    "q025": np.nan,
                    "q975": np.nan,
                })
            else:
                rows.append({
                    "model": model_name,
                    "metric": metric,
                    "n": len(x),
                    "mean": float(x.mean()),
                    "sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                    "median": float(x.median()),
                    "q025": float(x.quantile(0.025)),
                    "q975": float(x.quantile(0.975)),
                })
    return pd.DataFrame(rows)


def add_sheet(writer, df: pd.DataFrame, sheet_name: str) -> None:
    sheet_name = sheet_name[:31]
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    ws.freeze_panes = "A2"
    for col_cells in ws.columns:
        max_len = 0
        col_letter = col_cells[0].column_letter
        for cell in col_cells[:200]:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 60)


# ============================================================
# 2. Cross-validation
# ============================================================

def run_repeated_cv(
    X: np.ndarray,
    df: pd.DataFrame,
    models: Dict[str, Any],
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int = 42,
    use_sample_weight: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    y = df["model_label"].astype(int).values
    weights = df["label_weight"].astype(float).values if use_sample_weight and "label_weight" in df.columns else None

    min_class = int(pd.Series(y).value_counts().min())
    n_splits_eff = min(n_splits, min_class)
    if n_splits_eff < 2:
        raise RuntimeError("At least two samples per class are required for stratified CV.")

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits_eff,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    metrics_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    for model_name, base_model in models.items():
        print(f"[INFO] CV model: {model_name}")

        for fold_id, (tr, te) in enumerate(cv.split(X, y), start=1):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            w_tr = weights[tr] if weights is not None else None
            w_te = weights[te] if weights is not None else None

            model = clone(base_model)
            fit_kwargs = sample_weight_for_model(model, w_tr)
            if fit_kwargs is None:
                model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr, **fit_kwargs)

            p_tr = predict_proba_positive(model, X_tr)
            p_te = predict_proba_positive(model, X_te)

            # Use train fold to choose Youden threshold; avoids direct test leakage.
            thr_youden = choose_youden_threshold(y_tr, p_tr)

            # Metrics at 0.5
            m05 = binary_metrics(y_te, p_te, threshold=0.5, sample_weight=w_te)
            m05.update({
                "model": model_name,
                "fold": fold_id,
                "threshold_type": "fixed_0.5",
                "n_train": len(tr),
                "n_test": len(te),
                "n_train_pos": int(y_tr.sum()),
                "n_test_pos": int(y_te.sum()),
            })
            metrics_rows.append(m05)

            # Metrics at train Youden
            my = binary_metrics(y_te, p_te, threshold=thr_youden, sample_weight=w_te)
            my.update({
                "model": model_name,
                "fold": fold_id,
                "threshold_type": "train_youden",
                "n_train": len(tr),
                "n_test": len(te),
                "n_train_pos": int(y_tr.sum()),
                "n_test_pos": int(y_te.sum()),
            })
            metrics_rows.append(my)

            for local_idx, global_idx in enumerate(te):
                row = df.iloc[global_idx]
                pred_rows.append({
                    "model": model_name,
                    "fold": fold_id,
                    "analysis_drug_name": row.get("analysis_drug_name"),
                    "model_label": int(row.get("model_label")),
                    "label_weight": float(row.get("label_weight")),
                    "faers_signal_class": row.get("faers_signal_class"),
                    "canonical_smiles": row.get("canonical_smiles"),
                    "inchikey": row.get("inchikey"),
                    "prob_positive": float(p_te[local_idx]),
                    "pred_0.5": int(p_te[local_idx] >= 0.5),
                    "pred_train_youden": int(p_te[local_idx] >= thr_youden),
                    "train_youden_threshold": float(thr_youden),
                })

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.DataFrame(pred_rows)

    return metrics_df, preds_df


def run_permutation_test(
    X: np.ndarray,
    df: pd.DataFrame,
    models: Dict[str, Any],
    n_permutations: int = 100,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weight: bool = True,
) -> pd.DataFrame:
    if n_permutations <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    y_orig = df["model_label"].astype(int).values
    weights = df["label_weight"].astype(float).values if use_sample_weight and "label_weight" in df.columns else None

    min_class = int(pd.Series(y_orig).value_counts().min())
    n_splits_eff = min(n_splits, min_class)
    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    rows = []
    for model_name, base_model in models.items():
        print(f"[INFO] Permutation test: {model_name}")
        for perm_id in range(1, n_permutations + 1):
            y_perm = rng.permutation(y_orig)
            aucs = []
            praucs = []

            for tr, te in cv.split(X, y_perm):
                model = clone(base_model)
                w_tr = weights[tr] if weights is not None else None
                w_te = weights[te] if weights is not None else None
                fit_kwargs = sample_weight_for_model(model, w_tr)

                if fit_kwargs is None:
                    model.fit(X[tr], y_perm[tr])
                else:
                    model.fit(X[tr], y_perm[tr], **fit_kwargs)

                p = predict_proba_positive(model, X[te])
                if len(np.unique(y_perm[te])) == 2:
                    aucs.append(roc_auc_score(y_perm[te], p, sample_weight=w_te))
                    praucs.append(average_precision_score(y_perm[te], p, sample_weight=w_te))

            rows.append({
                "model": model_name,
                "permutation": perm_id,
                "roc_auc_mean": float(np.mean(aucs)) if aucs else np.nan,
                "pr_auc_mean": float(np.mean(praucs)) if praucs else np.nan,
            })

    return pd.DataFrame(rows)


# ============================================================
# 3. Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="model_ready CSV. If omitted, latest output/gnn_dataset_model_ready_colab_*.csv is used.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--no-chirality", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-sample-weight", action="store_true")
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        print("[ERROR] RDKit is not available. Install in Colab with: pip install rdkit", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input is None:
        input_path = find_latest_model_ready_csv(output_dir)
        if input_path is None:
            print("[ERROR] No model_ready CSV found. Please pass --input.", file=sys.stderr)
            return 1
    else:
        input_path = Path(args.input)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        return 1

    print(f"[INFO] Loading: {input_path}")
    df = pd.read_csv(input_path)

    required = ["canonical_smiles", "model_label", "label_weight", "analysis_drug_name", "faers_signal_class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}", file=sys.stderr)
        return 1

    # Remove duplicate InChIKey if any, preserving higher confidence by simple sorting.
    # No automatic deduplication by default; only report duplicates.
    dup_inchi = (
        df[df["inchikey"].notna()]
        .groupby("inchikey")
        .filter(lambda x: len(x) > 1)
        .sort_values("inchikey")
    )
    if len(dup_inchi) > 0:
        print("[WARN] Duplicate InChIKey records detected. They are retained in this exploratory run.")
        print(dup_inchi[["inchikey", "analysis_drug_name", "model_label", "faers_signal_class"]].to_string(index=False))

    X, df_valid = make_ecfp_matrix(
        df,
        smiles_col="canonical_smiles",
        radius=args.radius,
        n_bits=args.n_bits,
        use_chirality=(not args.no_chirality),
    )

    y = df_valid["model_label"].astype(int).values
    print(f"[INFO] Valid molecules: {len(df_valid)}")
    print(pd.Series(y).value_counts().rename_axis("label").reset_index(name="n").to_string(index=False))

    models = get_models(random_state=args.random_state)
    print(f"[INFO] Models: {list(models.keys())}")
    if not LIGHTGBM_AVAILABLE:
        print("[WARN] LightGBM is not available. Install with: pip install lightgbm")

    metrics_df, preds_df = run_repeated_cv(
        X=X,
        df=df_valid,
        models=models,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        use_sample_weight=(not args.no_sample_weight),
    )

    summary_df = summarize_metrics(metrics_df)

    permutation_df = run_permutation_test(
        X=X,
        df=df_valid,
        models=models,
        n_permutations=args.n_permutations,
        n_splits=args.n_splits,
        random_state=args.random_state,
        use_sample_weight=(not args.no_sample_weight),
    )

    # Simple permutation p-values based on mean fixed_0.5 ROC-AUC across CV.
    observed_auc = (
        metrics_df[metrics_df["threshold_type"].eq("fixed_0.5")]
        .groupby("model")["roc_auc"]
        .mean()
        .reset_index(name="observed_roc_auc_mean")
    )

    if not permutation_df.empty:
        perm_p_rows = []
        for _, r in observed_auc.iterrows():
            model = r["model"]
            obs = r["observed_roc_auc_mean"]
            perm = permutation_df[permutation_df["model"].eq(model)]["roc_auc_mean"].dropna()
            if len(perm) > 0:
                p = (1 + (perm >= obs).sum()) / (len(perm) + 1)
            else:
                p = np.nan
            perm_p_rows.append({
                "model": model,
                "observed_roc_auc_mean": obs,
                "n_permutations": len(perm),
                "permutation_p_value": p,
                "perm_roc_auc_mean": float(perm.mean()) if len(perm) else np.nan,
                "perm_roc_auc_q95": float(perm.quantile(0.95)) if len(perm) else np.nan,
            })
        permutation_summary = pd.DataFrame(perm_p_rows)
    else:
        permutation_summary = pd.DataFrame()

    # Data summary
    data_summary = pd.DataFrame([
        ["input_file", str(input_path)],
        ["n_records_input", len(df)],
        ["n_records_valid", len(df_valid)],
        ["n_positive", int((df_valid["model_label"] == 1).sum())],
        ["n_negative", int((df_valid["model_label"] == 0).sum())],
        ["radius", args.radius],
        ["n_bits", args.n_bits],
        ["use_chirality", not args.no_chirality],
        ["n_splits", args.n_splits],
        ["n_repeats", args.n_repeats],
        ["n_permutations", args.n_permutations],
        ["sample_weight_used", not args.no_sample_weight],
        ["lightgbm_available", LIGHTGBM_AVAILABLE],
    ], columns=["item", "value"])

    class_counts = (
        df_valid.groupby(["model_label", "faers_signal_class"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = output_dir / f"ecfp_baseline_results_{ts}.xlsx"
    out_preds = output_dir / f"ecfp_baseline_fold_predictions_{ts}.csv"
    out_metrics = output_dir / f"ecfp_baseline_fold_metrics_{ts}.csv"
    out_summary = output_dir / f"ecfp_baseline_model_summary_{ts}.csv"

    preds_df.to_csv(out_preds, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_metrics, index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print(f"[INFO] Writing Excel: {out_xlsx}")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        add_sheet(writer, data_summary, "01_data_summary")
        add_sheet(writer, class_counts, "02_class_counts")
        add_sheet(writer, summary_df, "03_metric_summary")
        add_sheet(writer, metrics_df, "04_fold_metrics")
        add_sheet(writer, preds_df, "05_fold_predictions")
        add_sheet(writer, observed_auc, "06_observed_auc")
        if not permutation_df.empty:
            add_sheet(writer, permutation_summary, "07_permutation_summary")
            add_sheet(writer, permutation_df, "08_permutation_raw")
        if len(dup_inchi) > 0:
            add_sheet(writer, dup_inchi, "09_duplicate_inchikey")
        add_sheet(writer, df_valid, "10_model_ready_used")

    print("[INFO] Done.")
    print("===== Data summary =====")
    print(data_summary.to_string(index=False))
    print("===== Class counts =====")
    print(class_counts.to_string(index=False))
    print("===== Metric summary: ROC-AUC / PR-AUC =====")
    print(
        summary_df[summary_df["metric"].isin(["roc_auc", "pr_auc", "balanced_accuracy"])]
        .sort_values(["metric", "mean"], ascending=[True, False])
        .to_string(index=False)
    )
    if not permutation_df.empty:
        print("===== Permutation summary =====")
        print(permutation_summary.to_string(index=False))

    print("[INFO] Output files:")
    print(f"  {out_xlsx}")
    print(f"  {out_preds}")
    print(f"  {out_metrics}")
    print(f"  {out_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
