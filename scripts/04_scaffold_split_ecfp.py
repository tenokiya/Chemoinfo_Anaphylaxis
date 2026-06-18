#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_scaffold_split_ecfp_baseline_colab.py

目的:
    FAERS由来アナフィラキシー報告シグナルのmodel_ready CSVを用いて、
    Bemis–Murcko scaffold単位のgroup splitでECFPベースラインモデルを評価する。

入力:
    output/gnn_dataset_model_ready_colab_YYYYMMDD_HHMMSS.csv
    または --input で指定したCSV

評価:
    - Bemis–Murcko scaffoldをRDKitで計算
    - 同一scaffoldをtrain/testにまたがらせない
    - StratifiedGroupKFoldが利用可能な場合はそれを使用
    - 利用不可または分割困難な場合は、独自のgreedy group splitへfallback
    - ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity, F1, MCC等を算出
    - fixed 0.5 threshold と train-fold Youden threshold の両方を評価

モデル:
    1. ECFP4 + Logistic regression L2
    2. ECFP4 + RandomForest
    3. ECFP4 + LightGBM（利用可能な場合）

出力:
    output/scaffold_split_ecfp_results_YYYYMMDD_HHMMSS.xlsx
    output/scaffold_split_ecfp_fold_predictions_YYYYMMDD_HHMMSS.csv
    output/scaffold_split_ecfp_fold_metrics_YYYYMMDD_HHMMSS.csv
    output/scaffold_split_ecfp_model_summary_YYYYMMDD_HHMMSS.csv

実行例:
    python 06_scaffold_split_ecfp_baseline_colab.py \
      --input output/gnn_dataset_model_ready_colab_20260521_085012.csv \
      --output-dir output \
      --n-splits 5 \
      --n-repeats 20 \
      --n-permutations 100

注意:
    scaffold splitでは、foldによって陽性/陰性が偏ることがある。
    小規模データセットでは結果の分散が大きくなるため、random split結果と併記する。
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
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
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
    STRATIFIED_GROUP_KFOLD_AVAILABLE = True
except Exception:
    STRATIFIED_GROUP_KFOLD_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
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


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip())


def smiles_to_mol(smiles: Any):
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is not available. Install with: pip install rdkit")
    if pd.isna(smiles) or str(smiles).strip() == "":
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def smiles_to_ecfp(
    smiles: Any,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> Optional[np.ndarray]:
    mol = smiles_to_mol(smiles)
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
    valid_idx = []
    failed_idx = []

    for i, row in df.iterrows():
        fp = smiles_to_ecfp(
            row.get(smiles_col),
            radius=radius,
            n_bits=n_bits,
            use_chirality=use_chirality,
        )
        if fp is None:
            failed_idx.append(i)
        else:
            fps.append(fp)
            valid_idx.append(i)

    if not fps:
        raise RuntimeError("No valid ECFP fingerprints were generated.")

    out_df = df.loc[valid_idx].reset_index(drop=True).copy()
    X = np.vstack(fps).astype(np.float32)

    if failed_idx:
        print(f"[WARN] Failed SMILES rows: {len(failed_idx)}")
        cols = [c for c in ["analysis_drug_name", smiles_col] if c in df.columns]
        print(df.loc[failed_idx, cols].head(20).to_string(index=False))

    return X, out_df


def get_scaffold_key(row: pd.Series, include_chirality: bool = False) -> str:
    """
    Bemis–Murcko scaffoldを返す。
    scaffoldが空のacyclic moleculeでは、全てが同一groupになることを避けるため、
    InChIKeyまたはcanonical SMILESに基づく個別groupを割り当てる。
    """
    smiles = row.get("canonical_smiles")
    mol = smiles_to_mol(smiles)
    if mol is None:
        name = norm_text(row.get("analysis_drug_name"))
        return f"INVALID__{name}"

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=include_chirality,
        )
    except Exception:
        scaffold = ""

    if scaffold is None:
        scaffold = ""

    scaffold = str(scaffold).strip()

    if scaffold == "":
        inchikey = norm_text(row.get("inchikey"))
        if inchikey:
            return f"NO_SCAFFOLD__{inchikey}"
        smi = norm_text(smiles)
        if smi:
            return f"NO_SCAFFOLD__{smi}"
        name = norm_text(row.get("analysis_drug_name"))
        return f"NO_SCAFFOLD__{name}"

    return scaffold


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
    for (model_name, threshold_type), g in metrics_df.groupby(["model", "threshold_type"]):
        for metric in metric_cols:
            x = pd.to_numeric(g[metric], errors="coerce").dropna()
            rows.append({
                "model": model_name,
                "threshold_type": threshold_type,
                "metric": metric,
                "n": len(x),
                "mean": float(x.mean()) if len(x) else np.nan,
                "sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                "median": float(x.median()) if len(x) else np.nan,
                "q025": float(x.quantile(0.025)) if len(x) else np.nan,
                "q975": float(x.quantile(0.975)) if len(x) else np.nan,
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
# 2. Scaffold grouping and splitters
# ============================================================

def make_scaffold_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("scaffold_key", dropna=False)
        .agg(
            n=("analysis_drug_name", "count"),
            n_pos=("model_label", lambda x: int((x == 1).sum())),
            n_neg=("model_label", lambda x: int((x == 0).sum())),
            drugs=("analysis_drug_name", lambda x: " | ".join(map(str, x))),
            classes=("faers_signal_class", lambda x: " | ".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .sort_values(["n", "n_pos"], ascending=[False, False])
    )
    g["pos_rate"] = g["n_pos"] / g["n"]
    return g


def get_sgkf_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> List[Tuple[int, np.ndarray, np.ndarray, str]]:
    splits: List[Tuple[int, np.ndarray, np.ndarray, str]] = []

    if not STRATIFIED_GROUP_KFOLD_AVAILABLE:
        return splits

    unique_groups = np.unique(groups)
    n_splits_eff = min(n_splits, len(unique_groups), int(pd.Series(y).value_counts().min()))
    if n_splits_eff < 2:
        return splits

    fold_global = 0
    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits_eff,
            shuffle=True,
            random_state=random_state + rep,
        )
        for tr, te in sgkf.split(X, y, groups):
            # Require both classes in train and test for ROC-AUC.
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            fold_global += 1
            splits.append((fold_global, tr, te, "StratifiedGroupKFold"))
    return splits


def greedy_group_split_once(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Greedy scaffold group split.
    大きいgroupから順に、各foldの陽性率・サイズが全体に近くなるよう配置する。
    """
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame({"idx": np.arange(len(y)), "y": y, "group": groups})

    group_stats = (
        df.groupby("group")
        .agg(n=("idx", "count"), n_pos=("y", "sum"))
        .reset_index()
    )
    group_stats["n_neg"] = group_stats["n"] - group_stats["n_pos"]

    # 大きいgroupを優先し、同サイズではランダムにする。
    group_stats["_rand"] = rng.random(len(group_stats))
    group_stats = group_stats.sort_values(["n", "_rand"], ascending=[False, True])

    total_pos = int(y.sum())
    total_neg = int(len(y) - y.sum())
    total_n = len(y)

    n_splits_eff = min(n_splits, len(group_stats), total_pos, total_neg)
    if n_splits_eff < 2:
        raise RuntimeError("Not enough groups/classes for greedy scaffold split.")

    folds = [{"groups": [], "n": 0, "pos": 0, "neg": 0} for _ in range(n_splits_eff)]

    target_n = total_n / n_splits_eff
    target_pos = total_pos / n_splits_eff
    target_neg = total_neg / n_splits_eff

    for _, row in group_stats.iterrows():
        best_fold = None
        best_score = None
        for k in range(n_splits_eff):
            n_new = folds[k]["n"] + int(row["n"])
            p_new = folds[k]["pos"] + int(row["n_pos"])
            ng_new = folds[k]["neg"] + int(row["n_neg"])
            score = (
                abs(n_new - target_n) / max(target_n, 1)
                + abs(p_new - target_pos) / max(target_pos, 1)
                + abs(ng_new - target_neg) / max(target_neg, 1)
            )
            if best_score is None or score < best_score:
                best_score = score
                best_fold = k
        folds[best_fold]["groups"].append(row["group"])
        folds[best_fold]["n"] += int(row["n"])
        folds[best_fold]["pos"] += int(row["n_pos"])
        folds[best_fold]["neg"] += int(row["n_neg"])

    splits = []
    all_idx = np.arange(len(y))
    for k in range(n_splits_eff):
        test_groups = set(folds[k]["groups"])
        te = df[df["group"].isin(test_groups)]["idx"].values
        tr = np.setdiff1d(all_idx, te)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        splits.append((tr, te))

    return splits


def get_greedy_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> List[Tuple[int, np.ndarray, np.ndarray, str]]:
    splits: List[Tuple[int, np.ndarray, np.ndarray, str]] = []
    fold_global = 0
    for rep in range(n_repeats):
        one = greedy_group_split_once(
            y=y,
            groups=groups,
            n_splits=n_splits,
            random_state=random_state + rep,
        )
        for tr, te in one:
            fold_global += 1
            splits.append((fold_global, tr, te, "greedy_group_split"))
    return splits


# ============================================================
# 3. Evaluation
# ============================================================

def run_scaffold_cv(
    X: np.ndarray,
    df: pd.DataFrame,
    models: Dict[str, Any],
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int = 42,
    use_sample_weight: bool = True,
    splitter: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["model_label"].astype(int).values
    groups = df["scaffold_key"].astype(str).values
    weights = df["label_weight"].astype(float).values if use_sample_weight and "label_weight" in df.columns else None

    if splitter in ["auto", "sgkf"]:
        splits = get_sgkf_splits(
            X=X,
            y=y,
            groups=groups,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    else:
        splits = []

    if not splits and splitter in ["auto", "greedy"]:
        print("[WARN] StratifiedGroupKFold unavailable or failed. Falling back to greedy group split.")
        splits = get_greedy_splits(
            y=y,
            groups=groups,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    if not splits:
        raise RuntimeError("No valid scaffold splits were generated.")

    fold_rows = []
    for fold_id, tr, te, split_method in splits:
        fold_rows.append({
            "fold": fold_id,
            "split_method": split_method,
            "n_train": len(tr),
            "n_test": len(te),
            "n_train_pos": int(y[tr].sum()),
            "n_train_neg": int(len(tr) - y[tr].sum()),
            "n_test_pos": int(y[te].sum()),
            "n_test_neg": int(len(te) - y[te].sum()),
            "n_train_scaffolds": len(set(groups[tr])),
            "n_test_scaffolds": len(set(groups[te])),
            "train_scaffolds": " | ".join(sorted(set(groups[tr]))[:50]),
            "test_scaffolds": " | ".join(sorted(set(groups[te]))[:50]),
        })

    fold_info = pd.DataFrame(fold_rows)

    metrics_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    for model_name, base_model in models.items():
        print(f"[INFO] Scaffold CV model: {model_name}")

        for fold_id, tr, te, split_method in splits:
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
            thr_youden = choose_youden_threshold(y_tr, p_tr)

            for threshold_type, thr in [("fixed_0.5", 0.5), ("train_youden", thr_youden)]:
                m = binary_metrics(y_te, p_te, threshold=float(thr), sample_weight=w_te)
                m.update({
                    "model": model_name,
                    "fold": fold_id,
                    "split_method": split_method,
                    "threshold_type": threshold_type,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "n_train_pos": int(y_tr.sum()),
                    "n_test_pos": int(y_te.sum()),
                    "n_train_scaffolds": len(set(groups[tr])),
                    "n_test_scaffolds": len(set(groups[te])),
                })
                metrics_rows.append(m)

            for local_idx, global_idx in enumerate(te):
                row = df.iloc[global_idx]
                pred_rows.append({
                    "model": model_name,
                    "fold": fold_id,
                    "split_method": split_method,
                    "analysis_drug_name": row.get("analysis_drug_name"),
                    "model_label": int(row.get("model_label")),
                    "label_weight": float(row.get("label_weight")),
                    "faers_signal_class": row.get("faers_signal_class"),
                    "scaffold_key": row.get("scaffold_key"),
                    "canonical_smiles": row.get("canonical_smiles"),
                    "inchikey": row.get("inchikey"),
                    "prob_positive": float(p_te[local_idx]),
                    "pred_0.5": int(p_te[local_idx] >= 0.5),
                    "pred_train_youden": int(p_te[local_idx] >= thr_youden),
                    "train_youden_threshold": float(thr_youden),
                })

    return pd.DataFrame(metrics_rows), pd.DataFrame(pred_rows), fold_info


def run_permutation_test(
    X: np.ndarray,
    df: pd.DataFrame,
    models: Dict[str, Any],
    n_permutations: int = 100,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weight: bool = True,
    splitter: str = "auto",
) -> pd.DataFrame:
    if n_permutations <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    y_orig = df["model_label"].astype(int).values
    groups = df["scaffold_key"].astype(str).values
    weights = df["label_weight"].astype(float).values if use_sample_weight and "label_weight" in df.columns else None

    rows = []

    for model_name, base_model in models.items():
        print(f"[INFO] Scaffold permutation: {model_name}")

        for perm_id in range(1, n_permutations + 1):
            y_perm = rng.permutation(y_orig)
            tmp_df = df.copy()
            tmp_df["model_label"] = y_perm

            try:
                # For computational efficiency, one scaffold split set per permutation.
                if splitter in ["auto", "sgkf"]:
                    splits = get_sgkf_splits(
                        X=X,
                        y=y_perm,
                        groups=groups,
                        n_splits=n_splits,
                        n_repeats=1,
                        random_state=random_state + perm_id,
                    )
                else:
                    splits = []

                if not splits and splitter in ["auto", "greedy"]:
                    splits = get_greedy_splits(
                        y=y_perm,
                        groups=groups,
                        n_splits=n_splits,
                        n_repeats=1,
                        random_state=random_state + perm_id,
                    )

                aucs = []
                praucs = []
                for _, tr, te, _method in splits:
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
                    "n_valid_folds": len(aucs),
                })

            except Exception as e:
                rows.append({
                    "model": model_name,
                    "permutation": perm_id,
                    "roc_auc_mean": np.nan,
                    "pr_auc_mean": np.nan,
                    "n_valid_folds": 0,
                    "error": str(e),
                })

    return pd.DataFrame(rows)


# ============================================================
# 4. Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="model_ready CSV. If omitted, latest output/gnn_dataset_model_ready_colab_*.csv is used.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--no-chirality", action="store_true")
    parser.add_argument("--scaffold-chirality", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-sample-weight", action="store_true")
    parser.add_argument("--splitter", default="auto", choices=["auto", "sgkf", "greedy"])
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

    X, df_valid = make_ecfp_matrix(
        df,
        smiles_col="canonical_smiles",
        radius=args.radius,
        n_bits=args.n_bits,
        use_chirality=(not args.no_chirality),
    )

    df_valid["model_label"] = df_valid["model_label"].astype(int)
    df_valid["label_weight"] = df_valid["label_weight"].astype(float)
    df_valid["scaffold_key"] = df_valid.apply(
        lambda r: get_scaffold_key(r, include_chirality=args.scaffold_chirality),
        axis=1,
    )

    y = df_valid["model_label"].values
    print(f"[INFO] Valid molecules: {len(df_valid)}")
    print(pd.Series(y).value_counts().rename_axis("label").reset_index(name="n").to_string(index=False))

    scaffold_summary = make_scaffold_summary(df_valid)
    print(f"[INFO] Number of scaffold groups: {len(scaffold_summary)}")
    print("[INFO] Largest scaffold groups:")
    print(scaffold_summary.head(15).to_string(index=False))

    models = get_models(random_state=args.random_state)
    print(f"[INFO] Models: {list(models.keys())}")
    if not LIGHTGBM_AVAILABLE:
        print("[WARN] LightGBM is not available. Install with: pip install lightgbm")

    metrics_df, preds_df, fold_info = run_scaffold_cv(
        X=X,
        df=df_valid,
        models=models,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        use_sample_weight=(not args.no_sample_weight),
        splitter=args.splitter,
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
        splitter=args.splitter,
    )

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
            p = (1 + (perm >= obs).sum()) / (len(perm) + 1) if len(perm) else np.nan
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

    data_summary = pd.DataFrame([
        ["input_file", str(input_path)],
        ["n_records_input", len(df)],
        ["n_records_valid", len(df_valid)],
        ["n_positive", int((df_valid["model_label"] == 1).sum())],
        ["n_negative", int((df_valid["model_label"] == 0).sum())],
        ["n_scaffold_groups", len(scaffold_summary)],
        ["n_singleton_scaffolds", int((scaffold_summary["n"] == 1).sum())],
        ["largest_scaffold_group_size", int(scaffold_summary["n"].max())],
        ["radius", args.radius],
        ["n_bits", args.n_bits],
        ["use_chirality_ecfp", not args.no_chirality],
        ["use_chirality_scaffold", args.scaffold_chirality],
        ["n_splits", args.n_splits],
        ["n_repeats", args.n_repeats],
        ["n_valid_scaffold_folds", metrics_df["fold"].nunique()],
        ["n_permutations", args.n_permutations],
        ["sample_weight_used", not args.no_sample_weight],
        ["splitter", args.splitter],
        ["stratified_group_kfold_available", STRATIFIED_GROUP_KFOLD_AVAILABLE],
        ["lightgbm_available", LIGHTGBM_AVAILABLE],
    ], columns=["item", "value"])

    class_counts = (
        df_valid.groupby(["model_label", "faers_signal_class"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = output_dir / f"scaffold_split_ecfp_results_{ts}.xlsx"
    out_preds = output_dir / f"scaffold_split_ecfp_fold_predictions_{ts}.csv"
    out_metrics = output_dir / f"scaffold_split_ecfp_fold_metrics_{ts}.csv"
    out_summary = output_dir / f"scaffold_split_ecfp_model_summary_{ts}.csv"
    out_scaffolds = output_dir / f"scaffold_split_scaffold_summary_{ts}.csv"

    preds_df.to_csv(out_preds, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_metrics, index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")
    scaffold_summary.to_csv(out_scaffolds, index=False, encoding="utf-8-sig")

    print(f"[INFO] Writing Excel: {out_xlsx}")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        add_sheet(writer, data_summary, "01_data_summary")
        add_sheet(writer, class_counts, "02_class_counts")
        add_sheet(writer, scaffold_summary, "03_scaffold_summary")
        add_sheet(writer, fold_info, "04_fold_info")
        add_sheet(writer, summary_df, "05_metric_summary")
        add_sheet(writer, metrics_df, "06_fold_metrics")
        add_sheet(writer, preds_df, "07_fold_predictions")
        add_sheet(writer, observed_auc, "08_observed_auc")
        if not permutation_df.empty:
            add_sheet(writer, permutation_summary, "09_permutation_summary")
            add_sheet(writer, permutation_df, "10_permutation_raw")
        add_sheet(writer, df_valid, "11_model_ready_used")

    print("[INFO] Done.")
    print("===== Data summary =====")
    print(data_summary.to_string(index=False))
    print("===== Class counts =====")
    print(class_counts.to_string(index=False))
    print("===== Metric summary: ROC-AUC / PR-AUC / balanced accuracy =====")
    print(
        summary_df[
            summary_df["metric"].isin(["roc_auc", "pr_auc", "balanced_accuracy"])
        ].sort_values(["threshold_type", "metric", "mean"], ascending=[True, True, False]).to_string(index=False)
    )
    if not permutation_summary.empty:
        print("===== Permutation summary =====")
        print(permutation_summary.to_string(index=False))

    print("[INFO] Output files:")
    print(f"  {out_xlsx}")
    print(f"  {out_preds}")
    print(f"  {out_metrics}")
    print(f"  {out_summary}")
    print(f"  {out_scaffolds}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
