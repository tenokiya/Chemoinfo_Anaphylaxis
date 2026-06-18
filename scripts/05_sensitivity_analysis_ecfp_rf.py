#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
07_sensitivity_analysis_ecfp_rf_colab.py

目的:
    ECFP + Random Forestを主モデルとして、FAERS由来アナフィラキシー報告シグナル予測の
    感度解析を一括実行する。

感度解析シナリオ:
    1. main_all_labels
        - positive_high + positive_moderate vs negative_high + negative_moderate + negative_expanded
    2. no_negative_expanded
        - negative_expandedを除外
    3. positive_high_vs_all_negative
        - positive_moderateを除外し、positive_highのみを陽性とする
    4. strict_high_confidence
        - positive_high vs negative_high + negative_moderate
    5. no_low_weight_labels
        - label_weight >= 0.8 のみ
          実質的に positive_high + negative_high + negative_moderate

評価:
    - Random split: repeated stratified K-fold
    - Scaffold split: Bemis–Murcko scaffold group split
    - fixed 0.5 threshold
    - train-fold Youden threshold
    - optional permutation test
    - drug-level mean prediction probability
    - ECFP bit importance from Random Forest

入力:
    output/gnn_dataset_model_ready_colab_YYYYMMDD_HHMMSS.csv

出力:
    output/sensitivity_ecfp_rf_results_YYYYMMDD_HHMMSS.xlsx
    output/sensitivity_ecfp_rf_metric_summary_YYYYMMDD_HHMMSS.csv
    output/sensitivity_ecfp_rf_fold_metrics_YYYYMMDD_HHMMSS.csv
    output/sensitivity_ecfp_rf_drug_probability_summary_YYYYMMDD_HHMMSS.csv
    output/sensitivity_ecfp_rf_bit_importance_YYYYMMDD_HHMMSS.csv

実行例:
    python 07_sensitivity_analysis_ecfp_rf_colab.py \
      --input output/gnn_dataset_model_ready_colab_20260521_085012.csv \
      --output-dir output \
      --n-splits 5 \
      --n-repeats 20 \
      --n-permutations 100

注意:
    小規模・弱教師ラベルの探索的解析である。
    シナリオによっては陽性/陰性数が少ないため、解釈には注意する。
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

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


def mol_to_ecfp_bitinfo(
    smiles: Any,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[int, List[Tuple[int, int]]]]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None, {}
    bit_info: Dict[int, List[Tuple[int, int]]] = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
        useChirality=use_chirality,
        bitInfo=bit_info,
    )
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, bit_info


def get_scaffold_key(row: pd.Series, include_chirality: bool = False) -> str:
    smiles = row.get("canonical_smiles")
    mol = smiles_to_mol(smiles)
    if mol is None:
        return f"INVALID__{norm_text(row.get('analysis_drug_name'))}"

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=include_chirality,
        )
    except Exception:
        scaffold = ""

    scaffold = "" if scaffold is None else str(scaffold).strip()

    if scaffold == "":
        inchikey = norm_text(row.get("inchikey"))
        if inchikey:
            return f"NO_SCAFFOLD__{inchikey}"
        smi = norm_text(smiles)
        if smi:
            return f"NO_SCAFFOLD__{smi}"
        return f"NO_SCAFFOLD__{norm_text(row.get('analysis_drug_name'))}"

    return scaffold


def make_ecfp_matrix(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[int, Dict[int, List[Tuple[int, int]]]]]:
    fps = []
    valid_idx = []
    failed_idx = []
    bitinfo_by_row: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}

    for i, row in df.iterrows():
        fp, bit_info = mol_to_ecfp_bitinfo(
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
            bitinfo_by_row[len(valid_idx) - 1] = bit_info

    if not fps:
        raise RuntimeError("No valid ECFP fingerprints were generated.")

    out_df = df.loc[valid_idx].reset_index(drop=True).copy()
    X = np.vstack(fps).astype(np.float32)

    if failed_idx:
        print(f"[WARN] Failed SMILES rows: {len(failed_idx)}")
        cols = [c for c in ["analysis_drug_name", smiles_col] if c in df.columns]
        print(df.loc[failed_idx, cols].head(20).to_string(index=False))

    return X, out_df, bitinfo_by_row


def get_rf(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )


def predict_proba_positive(model: Any, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] > 1:
        return proba[:, 1]
    return proba.ravel()


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
    group_cols = ["scenario", "split_type", "threshold_type"]
    for keys, g in metrics_df.groupby(group_cols):
        scenario, split_type, threshold_type = keys
        for metric in metric_cols:
            x = pd.to_numeric(g[metric], errors="coerce").dropna()
            rows.append({
                "scenario": scenario,
                "split_type": split_type,
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
# 2. Scenarios
# ============================================================

def assign_scenarios(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    scenarios: Dict[str, pd.DataFrame] = {}

    scenarios["main_all_labels"] = df.copy()

    scenarios["no_negative_expanded"] = df[
        ~df["faers_signal_class"].eq("negative_expanded")
    ].copy()

    scenarios["positive_high_vs_all_negative"] = df[
        df["faers_signal_class"].eq("positive_high") |
        df["faers_signal_class"].str.startswith("negative")
    ].copy()

    scenarios["strict_high_confidence"] = df[
        df["faers_signal_class"].eq("positive_high") |
        df["faers_signal_class"].isin(["negative_high", "negative_moderate"])
    ].copy()

    scenarios["no_low_weight_labels"] = df[
        pd.to_numeric(df["label_weight"], errors="coerce") >= 0.8
    ].copy()

    # model_labelは元のまま維持する。
    # 各scenarioで陽性/陰性が2クラス揃わない場合は後でskipする。
    return scenarios


def scenario_summary(scenarios: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, sdf in scenarios.items():
        rows.append({
            "scenario": name,
            "n": len(sdf),
            "n_positive": int((sdf["model_label"] == 1).sum()),
            "n_negative": int((sdf["model_label"] == 0).sum()),
            "classes": " | ".join(
                f"{k}:{v}" for k, v in sdf["faers_signal_class"].value_counts().sort_index().items()
            ),
        })
    return pd.DataFrame(rows)


# ============================================================
# 3. Splitters
# ============================================================

def get_random_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> List[Tuple[int, np.ndarray, np.ndarray, str]]:
    min_class = int(pd.Series(y).value_counts().min())
    n_splits_eff = min(n_splits, min_class)
    if n_splits_eff < 2:
        return []

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits_eff,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    splits = []
    for fold_id, (tr, te) in enumerate(cv.split(X, y), start=1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        splits.append((fold_id, tr, te, "random_repeated_stratified"))
    return splits


def get_sgkf_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> List[Tuple[int, np.ndarray, np.ndarray, str]]:
    if not STRATIFIED_GROUP_KFOLD_AVAILABLE:
        return []

    unique_groups = np.unique(groups)
    min_class = int(pd.Series(y).value_counts().min())
    n_splits_eff = min(n_splits, len(unique_groups), min_class)
    if n_splits_eff < 2:
        return []

    splits = []
    fold_global = 0
    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits_eff,
            shuffle=True,
            random_state=random_state + rep,
        )
        for tr, te in sgkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            fold_global += 1
            splits.append((fold_global, tr, te, "scaffold_stratified_group"))
    return splits


def greedy_group_split_once(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame({"idx": np.arange(len(y)), "y": y, "group": groups})
    group_stats = (
        df.groupby("group")
        .agg(n=("idx", "count"), n_pos=("y", "sum"))
        .reset_index()
    )
    group_stats["n_neg"] = group_stats["n"] - group_stats["n_pos"]
    group_stats["_rand"] = rng.random(len(group_stats))
    group_stats = group_stats.sort_values(["n", "_rand"], ascending=[False, True])

    total_pos = int(y.sum())
    total_neg = int(len(y) - y.sum())
    total_n = len(y)

    n_splits_eff = min(n_splits, len(group_stats), total_pos, total_neg)
    if n_splits_eff < 2:
        return []

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

    all_idx = np.arange(len(y))
    splits = []
    for k in range(n_splits_eff):
        test_groups = set(folds[k]["groups"])
        te = df[df["group"].isin(test_groups)]["idx"].values
        tr = np.setdiff1d(all_idx, te)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        splits.append((tr, te))
    return splits


def get_greedy_group_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> List[Tuple[int, np.ndarray, np.ndarray, str]]:
    splits = []
    fold_global = 0
    for rep in range(n_repeats):
        one = greedy_group_split_once(y, groups, n_splits, random_state + rep)
        for tr, te in one:
            fold_global += 1
            splits.append((fold_global, tr, te, "scaffold_greedy_group"))
    return splits


def get_scaffold_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> List[Tuple[int, np.ndarray, np.ndarray, str]]:
    splits = get_sgkf_splits(X, y, groups, n_splits, n_repeats, random_state)
    if not splits:
        splits = get_greedy_group_splits(y, groups, n_splits, n_repeats, random_state)
    return splits


# ============================================================
# 4. Evaluation
# ============================================================

def evaluate_scenario(
    scenario_name: str,
    sdf: pd.DataFrame,
    radius: int,
    n_bits: int,
    use_chirality: bool,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    use_sample_weight: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, vdf, bitinfo_by_row = make_ecfp_matrix(
        sdf,
        smiles_col="canonical_smiles",
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality,
    )

    vdf["model_label"] = vdf["model_label"].astype(int)
    vdf["label_weight"] = vdf["label_weight"].astype(float)
    vdf["scaffold_key"] = vdf.apply(lambda r: get_scaffold_key(r), axis=1)

    y = vdf["model_label"].values
    groups = vdf["scaffold_key"].astype(str).values
    weights = vdf["label_weight"].values if use_sample_weight else None

    # Skip if not enough data.
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        print(f"[WARN] Scenario skipped due to insufficient class counts: {scenario_name}")
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    split_sets: List[Tuple[str, List[Tuple[int, np.ndarray, np.ndarray, str]]]] = []
    split_sets.append((
        "random",
        get_random_splits(X, y, n_splits, n_repeats, random_state),
    ))
    split_sets.append((
        "scaffold",
        get_scaffold_splits(X, y, groups, n_splits, n_repeats, random_state),
    ))

    rf = get_rf(random_state=random_state)

    metric_rows = []
    pred_rows = []
    fold_rows = []

    for split_type, splits in split_sets:
        if not splits:
            print(f"[WARN] No valid splits for {scenario_name} / {split_type}")
            continue

        for fold_id, tr, te, split_method in splits:
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            w_tr = weights[tr] if weights is not None else None
            w_te = weights[te] if weights is not None else None

            model = clone(rf)
            if w_tr is None:
                model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr, sample_weight=w_tr)

            p_tr = predict_proba_positive(model, X_tr)
            p_te = predict_proba_positive(model, X_te)
            thr_youden = choose_youden_threshold(y_tr, p_tr)

            fold_rows.append({
                "scenario": scenario_name,
                "split_type": split_type,
                "fold": fold_id,
                "split_method": split_method,
                "n_train": len(tr),
                "n_test": len(te),
                "n_train_pos": int(y_tr.sum()),
                "n_train_neg": int(len(tr) - y_tr.sum()),
                "n_test_pos": int(y_te.sum()),
                "n_test_neg": int(len(te) - y_te.sum()),
                "n_train_scaffolds": len(set(groups[tr])),
                "n_test_scaffolds": len(set(groups[te])),
            })

            for threshold_type, thr in [("fixed_0.5", 0.5), ("train_youden", thr_youden)]:
                m = binary_metrics(y_te, p_te, threshold=float(thr), sample_weight=w_te)
                m.update({
                    "scenario": scenario_name,
                    "split_type": split_type,
                    "fold": fold_id,
                    "split_method": split_method,
                    "threshold_type": threshold_type,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "n_train_pos": int(y_tr.sum()),
                    "n_test_pos": int(y_te.sum()),
                })
                metric_rows.append(m)

            for local_idx, global_idx in enumerate(te):
                row = vdf.iloc[global_idx]
                pred_rows.append({
                    "scenario": scenario_name,
                    "split_type": split_type,
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

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.DataFrame(pred_rows)
    fold_df = pd.DataFrame(fold_rows)

    # Full-data RF for ECFP bit importance.
    bitimp_df = compute_bit_importance(
        scenario_name=scenario_name,
        X=X,
        df=vdf,
        random_state=random_state,
        sample_weight=weights,
        bitinfo_by_row=bitinfo_by_row,
        top_n=200,
    )

    scaffold_summary = (
        vdf.groupby("scaffold_key", dropna=False)
        .agg(
            scenario=("analysis_drug_name", lambda x: scenario_name),
            n=("analysis_drug_name", "count"),
            n_pos=("model_label", lambda x: int((x == 1).sum())),
            n_neg=("model_label", lambda x: int((x == 0).sum())),
            drugs=("analysis_drug_name", lambda x: " | ".join(map(str, x))),
            classes=("faers_signal_class", lambda x: " | ".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .sort_values(["n", "n_pos"], ascending=[False, False])
    )
    scaffold_summary["pos_rate"] = scaffold_summary["n_pos"] / scaffold_summary["n"]

    return metrics_df, preds_df, fold_df, bitimp_df, scaffold_summary


def compute_bit_importance(
    scenario_name: str,
    X: np.ndarray,
    df: pd.DataFrame,
    random_state: int,
    sample_weight: Optional[np.ndarray],
    bitinfo_by_row: Dict[int, Dict[int, List[Tuple[int, int]]]],
    top_n: int = 200,
) -> pd.DataFrame:
    y = df["model_label"].astype(int).values

    model = get_rf(random_state=random_state)
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=sample_weight)

    importances = model.feature_importances_
    bit_ids = np.arange(X.shape[1])

    pos_mask = y == 1
    neg_mask = y == 0

    pos_freq = X[pos_mask].mean(axis=0) if pos_mask.sum() > 0 else np.zeros(X.shape[1])
    neg_freq = X[neg_mask].mean(axis=0) if neg_mask.sum() > 0 else np.zeros(X.shape[1])
    freq_diff = pos_freq - neg_freq

    top_idx = np.argsort(importances)[::-1][:top_n]

    rows = []
    for bit in top_idx:
        bit = int(bit)
        on_idx = np.where(X[:, bit] > 0)[0]
        pos_drugs = df.iloc[on_idx][df.iloc[on_idx]["model_label"].astype(int).eq(1)]["analysis_drug_name"].tolist()
        neg_drugs = df.iloc[on_idx][df.iloc[on_idx]["model_label"].astype(int).eq(0)]["analysis_drug_name"].tolist()

        # bitInfo examples
        examples = []
        for i in on_idx[:10]:
            info = bitinfo_by_row.get(int(i), {}).get(bit, [])
            if info:
                examples.append(f"{df.iloc[i].get('analysis_drug_name')}:{info[:3]}")
            else:
                examples.append(f"{df.iloc[i].get('analysis_drug_name')}:NA")

        rows.append({
            "scenario": scenario_name,
            "bit": bit,
            "importance": float(importances[bit]),
            "pos_freq": float(pos_freq[bit]),
            "neg_freq": float(neg_freq[bit]),
            "pos_minus_neg_freq": float(freq_diff[bit]),
            "n_on": int(len(on_idx)),
            "n_pos_on": int(len(pos_drugs)),
            "n_neg_on": int(len(neg_drugs)),
            "example_positive_drugs": " | ".join(map(str, pos_drugs[:20])),
            "example_negative_drugs": " | ".join(map(str, neg_drugs[:20])),
            "bitinfo_examples": " | ".join(examples),
        })

    return pd.DataFrame(rows)


def summarize_drug_probabilities(preds_df: pd.DataFrame) -> pd.DataFrame:
    if preds_df.empty:
        return pd.DataFrame()

    g = (
        preds_df.groupby(
            ["scenario", "split_type", "analysis_drug_name", "model_label", "faers_signal_class"],
            dropna=False,
        )
        .agg(
            mean_prob_positive=("prob_positive", "mean"),
            sd_prob_positive=("prob_positive", "std"),
            median_prob_positive=("prob_positive", "median"),
            n_predictions=("prob_positive", "count"),
            mean_pred_0_5=("pred_0.5", "mean"),
            mean_pred_train_youden=("pred_train_youden", "mean"),
        )
        .reset_index()
    )

    g["potential_false_positive"] = (g["model_label"].eq(0)) & (g["mean_prob_positive"] >= 0.6)
    g["potential_false_negative"] = (g["model_label"].eq(1)) & (g["mean_prob_positive"] <= 0.4)

    return g.sort_values(["scenario", "split_type", "mean_prob_positive"], ascending=[True, True, False])


def run_permutation_for_scenarios(
    scenarios: Dict[str, pd.DataFrame],
    radius: int,
    n_bits: int,
    use_chirality: bool,
    n_splits: int,
    n_permutations: int,
    random_state: int,
    use_sample_weight: bool,
) -> pd.DataFrame:
    if n_permutations <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    rows = []

    for scenario_name, sdf in scenarios.items():
        try:
            X, vdf, _ = make_ecfp_matrix(
                sdf,
                smiles_col="canonical_smiles",
                radius=radius,
                n_bits=n_bits,
                use_chirality=use_chirality,
            )
            vdf["model_label"] = vdf["model_label"].astype(int)
            vdf["label_weight"] = vdf["label_weight"].astype(float)
            vdf["scaffold_key"] = vdf.apply(lambda r: get_scaffold_key(r), axis=1)

            y_orig = vdf["model_label"].values
            if len(np.unique(y_orig)) < 2 or pd.Series(y_orig).value_counts().min() < 2:
                continue

            groups = vdf["scaffold_key"].astype(str).values
            weights = vdf["label_weight"].values if use_sample_weight else None

            # observed random and scaffold AUC, fixed threshold-independent.
            for split_type in ["random", "scaffold"]:
                if split_type == "random":
                    splits = get_random_splits(X, y_orig, n_splits, 1, random_state)
                else:
                    splits = get_scaffold_splits(X, y_orig, groups, n_splits, 1, random_state)

                observed = estimate_auc_once(X, y_orig, weights, splits, random_state)
                perm_aucs = []

                for p in range(1, n_permutations + 1):
                    y_perm = rng.permutation(y_orig)
                    if split_type == "random":
                        perm_splits = get_random_splits(X, y_perm, n_splits, 1, random_state + p)
                    else:
                        perm_splits = get_scaffold_splits(X, y_perm, groups, n_splits, 1, random_state + p)
                    auc = estimate_auc_once(X, y_perm, weights, perm_splits, random_state + p)
                    if np.isfinite(auc):
                        perm_aucs.append(auc)

                perm_aucs = np.asarray(perm_aucs, dtype=float)
                pval = (1 + np.sum(perm_aucs >= observed)) / (len(perm_aucs) + 1) if len(perm_aucs) else np.nan

                rows.append({
                    "scenario": scenario_name,
                    "split_type": split_type,
                    "observed_roc_auc_mean": observed,
                    "n_permutations": int(len(perm_aucs)),
                    "permutation_p_value": float(pval) if np.isfinite(pval) else np.nan,
                    "perm_roc_auc_mean": float(np.mean(perm_aucs)) if len(perm_aucs) else np.nan,
                    "perm_roc_auc_q95": float(np.quantile(perm_aucs, 0.95)) if len(perm_aucs) else np.nan,
                })

        except Exception as e:
            rows.append({
                "scenario": scenario_name,
                "split_type": "ERROR",
                "observed_roc_auc_mean": np.nan,
                "n_permutations": 0,
                "permutation_p_value": np.nan,
                "perm_roc_auc_mean": np.nan,
                "perm_roc_auc_q95": np.nan,
                "error": str(e),
            })

    return pd.DataFrame(rows)


def estimate_auc_once(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray],
    splits: List[Tuple[int, np.ndarray, np.ndarray, str]],
    random_state: int,
) -> float:
    aucs = []
    rf = get_rf(random_state=random_state)
    for _fold_id, tr, te, _method in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model = clone(rf)
        w_tr = weights[tr] if weights is not None else None
        w_te = weights[te] if weights is not None else None
        if w_tr is None:
            model.fit(X[tr], y[tr])
        else:
            model.fit(X[tr], y[tr], sample_weight=w_tr)
        p = predict_proba_positive(model, X[te])
        aucs.append(roc_auc_score(y[te], p, sample_weight=w_te))
    return float(np.mean(aucs)) if aucs else np.nan


# ============================================================
# 5. Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
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

    df["model_label"] = df["model_label"].astype(int)
    df["label_weight"] = df["label_weight"].astype(float)

    scenarios = assign_scenarios(df)
    scen_summary = scenario_summary(scenarios)
    print("===== Scenario summary =====")
    print(scen_summary.to_string(index=False))

    all_metrics = []
    all_preds = []
    all_folds = []
    all_bits = []
    all_scaffolds = []

    for scenario_name, sdf in scenarios.items():
        print(f"[INFO] Evaluating scenario: {scenario_name}")
        print(sdf["model_label"].value_counts().to_string())

        # skip if too small
        vc = sdf["model_label"].value_counts()
        if len(vc) < 2 or vc.min() < 2:
            print(f"[WARN] Skip {scenario_name}: insufficient class counts")
            continue

        metrics_df, preds_df, fold_df, bitimp_df, scaffold_df = evaluate_scenario(
            scenario_name=scenario_name,
            sdf=sdf,
            radius=args.radius,
            n_bits=args.n_bits,
            use_chirality=(not args.no_chirality),
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.random_state,
            use_sample_weight=(not args.no_sample_weight),
        )

        if not metrics_df.empty:
            all_metrics.append(metrics_df)
        if not preds_df.empty:
            all_preds.append(preds_df)
        if not fold_df.empty:
            all_folds.append(fold_df)
        if not bitimp_df.empty:
            all_bits.append(bitimp_df)
        if not scaffold_df.empty:
            all_scaffolds.append(scaffold_df)

    metrics_all = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    preds_all = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    folds_all = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    bits_all = pd.concat(all_bits, ignore_index=True) if all_bits else pd.DataFrame()
    scaffolds_all = pd.concat(all_scaffolds, ignore_index=True) if all_scaffolds else pd.DataFrame()

    metric_summary = summarize_metrics(metrics_all) if not metrics_all.empty else pd.DataFrame()
    drug_prob_summary = summarize_drug_probabilities(preds_all)

    permutation_summary = run_permutation_for_scenarios(
        scenarios=scenarios,
        radius=args.radius,
        n_bits=args.n_bits,
        use_chirality=(not args.no_chirality),
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        use_sample_weight=(not args.no_sample_weight),
    )

    data_summary = pd.DataFrame([
        ["input_file", str(input_path)],
        ["n_records_input", len(df)],
        ["radius", args.radius],
        ["n_bits", args.n_bits],
        ["use_chirality", not args.no_chirality],
        ["n_splits", args.n_splits],
        ["n_repeats", args.n_repeats],
        ["n_permutations", args.n_permutations],
        ["sample_weight_used", not args.no_sample_weight],
        ["n_scenarios", len(scenarios)],
    ], columns=["item", "value"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = output_dir / f"sensitivity_ecfp_rf_results_{ts}.xlsx"
    out_metric_summary = output_dir / f"sensitivity_ecfp_rf_metric_summary_{ts}.csv"
    out_fold_metrics = output_dir / f"sensitivity_ecfp_rf_fold_metrics_{ts}.csv"
    out_pred_summary = output_dir / f"sensitivity_ecfp_rf_drug_probability_summary_{ts}.csv"
    out_bits = output_dir / f"sensitivity_ecfp_rf_bit_importance_{ts}.csv"

    metric_summary.to_csv(out_metric_summary, index=False, encoding="utf-8-sig")
    metrics_all.to_csv(out_fold_metrics, index=False, encoding="utf-8-sig")
    drug_prob_summary.to_csv(out_pred_summary, index=False, encoding="utf-8-sig")
    bits_all.to_csv(out_bits, index=False, encoding="utf-8-sig")

    print(f"[INFO] Writing Excel: {out_xlsx}")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        add_sheet(writer, data_summary, "01_data_summary")
        add_sheet(writer, scen_summary, "02_scenario_summary")
        add_sheet(writer, metric_summary, "03_metric_summary")
        add_sheet(writer, permutation_summary, "04_permutation_summary")
        add_sheet(writer, drug_prob_summary, "05_drug_probability")
        add_sheet(writer, bits_all, "06_bit_importance")
        add_sheet(writer, scaffolds_all, "07_scaffold_summary")
        add_sheet(writer, folds_all, "08_fold_info")
        add_sheet(writer, metrics_all, "09_fold_metrics")
        add_sheet(writer, preds_all, "10_fold_predictions")
        add_sheet(writer, df, "11_input_model_ready")

    print("[INFO] Done.")
    print("===== Metric summary: ROC-AUC / PR-AUC / balanced accuracy =====")
    if not metric_summary.empty:
        print(
            metric_summary[
                metric_summary["metric"].isin(["roc_auc", "pr_auc", "balanced_accuracy"])
            ].sort_values(["scenario", "split_type", "threshold_type", "metric"]).to_string(index=False)
        )
    print("===== Permutation summary =====")
    if not permutation_summary.empty:
        print(permutation_summary.to_string(index=False))

    print("[INFO] Output files:")
    print(f"  {out_xlsx}")
    print(f"  {out_metric_summary}")
    print(f"  {out_fold_metrics}")
    print(f"  {out_pred_summary}")
    print(f"  {out_bits}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
