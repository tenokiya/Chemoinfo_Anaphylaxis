#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
11_treatment_confounding_sensitivity_colab.py

目的:
    FAERS由来アナフィラキシー報告シグナルモデルにおいて、
    PS/SSであっても治療薬・救急薬として混入し得る薬剤を label-review 候補としてタグ付けし、
    それらを除外した場合に ECFP + Random Forest の性能が維持されるかを検証する。

背景:
    ステロイド、抗ヒスタミン薬、アドレナリン/昇圧薬、気管支拡張薬などは、
    アナフィラキシーやアレルギー反応の治療薬・併用薬としてFAERS症例に混入し得る。
    role_cod = PS/SS であっても因果確定ではないため、これらを label-review または NA とする
    感度解析が必要である。

入力:
    output/gnn_dataset_model_ready_colab_*.csv

任意入力:
    input/treatment_confounding_drug_rules.csv
    列:
      analysis_drug_name
      confounding_category
      review_level
      comment

    このファイルが存在する場合、内蔵ルールに追加・上書きする。
    review_level は以下を推奨:
      corticosteroid
      antihistamine_h1
      antihistamine_h2
      adrenergic_rescue
      bronchodilator_or_asthma_context
      other_treatment_context

出力:
    output_treatment_sensitivity/
      treatment_confounded_model_ready_YYYYMMDD_HHMMSS.csv
      treatment_confounding_review_candidates_YYYYMMDD_HHMMSS.xlsx
      treatment_sensitivity_results_YYYYMMDD_HHMMSS.xlsx
      treatment_sensitivity_metric_summary_YYYYMMDD_HHMMSS.csv
      treatment_sensitivity_fold_metrics_YYYYMMDD_HHMMSS.csv
      treatment_sensitivity_predictions_YYYYMMDD_HHMMSS.csv
      treatment_confounding_rule_template.csv

感度解析シナリオ:
    1. original_all
       元のmodel_ready全体
    2. exclude_corticosteroids_all
       corticosteroid候補を全て除外
    3. exclude_antihistamines_all
       H1/H2抗ヒスタミン薬候補を全て除外
    4. exclude_core_treatment_all
       corticosteroid + H1/H2 antihistamine + adrenergic_rescue を除外
    5. exclude_broad_treatment_all
       core + bronchodilator/asthma context + other_treatment_context を除外
    6. exclude_core_treatment_positive_only
       positive labelのうちcore治療薬候補だけを除外

評価:
    - ECFP4, radius=2, nBits=2048, chirality=True
    - Random Forest, 500 trees, min_samples_leaf=2
    - label_weight を sample_weight として使用
    - repeated stratified random split
    - Bemis–Murcko scaffold split
    - ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity, MCC など

実行例:
    python 11_treatment_confounding_sensitivity_colab.py \
      --input output/gnn_dataset_model_ready_colab_20260521_085012.csv \
      --output-dir output_treatment_sensitivity \
      --n-splits 5 \
      --n-repeats 20

注意:
    この解析は、治療薬混入によるラベル汚染の影響を評価するための感度解析である。
    除外対象は「真に安全/危険でない」と断定するものではなく、label-review対象として扱う。
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

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
from sklearn.model_selection import RepeatedStratifiedKFold

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
# 1. Utility functions
# ============================================================

def latest_file(output_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(output_dir.glob(pattern))
    return files[-1] if files else None


def normalize_drug_name(x: Any) -> str:
    if pd.isna(x):
        return ""
    x = str(x).upper().strip()
    x = re.sub(r"\s+", " ", x)
    x = x.replace("−", "-")
    return x


def sanitize_filename(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))[:120]


def smiles_to_mol(smiles: Any):
    if pd.isna(smiles) or str(smiles).strip() == "":
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def smiles_to_ecfp(smiles: Any, radius: int, n_bits: int, use_chirality: bool) -> Optional[np.ndarray]:
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
    radius: int,
    n_bits: int,
    use_chirality: bool,
) -> Tuple[np.ndarray, pd.DataFrame]:
    fps = []
    valid = []
    failed = []

    for i, row in df.iterrows():
        fp = smiles_to_ecfp(row.get("canonical_smiles"), radius, n_bits, use_chirality)
        if fp is None:
            failed.append(i)
        else:
            fps.append(fp)
            valid.append(i)

    if not fps:
        raise RuntimeError("No valid ECFP fingerprints generated.")

    X = np.vstack(fps).astype(np.float32)
    vdf = df.loc[valid].reset_index(drop=True).copy()

    if failed:
        print(f"[WARN] Failed SMILES rows: {len(failed)}")
        cols = [c for c in ["analysis_drug_name", "canonical_smiles"] if c in df.columns]
        print(df.loc[failed, cols].to_string(index=False))

    return X, vdf


def get_scaffold_key(row: pd.Series) -> str:
    mol = smiles_to_mol(row.get("canonical_smiles"))
    if mol is None:
        return "INVALID__" + normalize_drug_name(row.get("analysis_drug_name"))

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ""

    scaffold = "" if scaffold is None else str(scaffold).strip()

    if scaffold == "":
        inchikey = normalize_drug_name(row.get("inchikey"))
        if inchikey:
            return "NO_SCAFFOLD__" + inchikey
        return "NO_SCAFFOLD__" + normalize_drug_name(row.get("canonical_smiles"))

    return scaffold


def get_rf(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )


def predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()


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
    threshold: float,
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


# ============================================================
# 2. Treatment-confounding rules
# ============================================================

BUILTIN_RULES = [
    # corticosteroids
    ("DEXAMETHASONE", "corticosteroid", "core", "systemic/topical corticosteroid; potential treatment-context confounding"),
    ("DEXAMETHASONE ACETATE", "corticosteroid", "core", "corticosteroid ester"),
    ("DEXAMETHASONE SODIUM PHOSPHATE", "corticosteroid", "core", "corticosteroid ester/salt"),
    ("METHYLPREDNISOLONE", "corticosteroid", "core", "systemic corticosteroid"),
    ("METHYLPREDNISOLONE SODIUM SUCCINATE", "corticosteroid", "core", "systemic corticosteroid ester/salt"),
    ("PREDNISOLONE", "corticosteroid", "core", "systemic corticosteroid"),
    ("PREDNISONE", "corticosteroid", "core", "systemic corticosteroid"),
    ("HYDROCORTISONE", "corticosteroid", "core", "systemic corticosteroid"),
    ("HYDROCORTISONE SODIUM SUCCINATE", "corticosteroid", "core", "systemic corticosteroid ester/salt"),
    ("BUDESONIDE", "corticosteroid", "core", "corticosteroid; allergy/asthma context possible"),
    ("BETAMETHASONE", "corticosteroid", "core", "corticosteroid"),
    ("TRIAMCINOLONE", "corticosteroid", "core", "corticosteroid"),
    ("FLUDROCORTISONE", "corticosteroid", "core", "corticosteroid"),
    ("CORTISONE", "corticosteroid", "core", "corticosteroid"),
    ("CLOBETASOL", "corticosteroid", "core", "corticosteroid"),
    ("MOMETASONE", "corticosteroid", "core", "corticosteroid"),
    ("FLUTICASONE", "corticosteroid", "core", "corticosteroid"),

    # H1 antihistamines
    ("DIPHENHYDRAMINE", "antihistamine_h1", "core", "H1 antihistamine; anaphylaxis/allergy treatment context"),
    ("CHLORPHENIRAMINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("DEXCHLORPHENIRAMINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("HYDROXYZINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("PROMETHAZINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("CETIRIZINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("LEVOCETIRIZINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("LORATADINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("DESLORATADINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("FEXOFENADINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("RUPATADINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("BILASTINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("DOXYLAMINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("MECLIZINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("CYPROHEPTADINE", "antihistamine_h1", "core", "H1 antihistamine"),
    ("CLEMASTINE", "antihistamine_h1", "core", "H1 antihistamine"),

    # H2 blockers
    ("FAMOTIDINE", "antihistamine_h2", "core", "H2 blocker; allergy/anaphylaxis treatment context possible"),
    ("RANITIDINE", "antihistamine_h2", "core", "H2 blocker"),
    ("CIMETIDINE", "antihistamine_h2", "core", "H2 blocker"),
    ("NIZATIDINE", "antihistamine_h2", "core", "H2 blocker"),

    # adrenergic rescue / shock treatment context
    ("EPINEPHRINE", "adrenergic_rescue", "core", "first-line anaphylaxis rescue medication"),
    ("ADRENALINE", "adrenergic_rescue", "core", "first-line anaphylaxis rescue medication"),
    ("NOREPINEPHRINE", "adrenergic_rescue", "core", "vasopressor/shock treatment context"),
    ("NORADRENALINE", "adrenergic_rescue", "core", "vasopressor/shock treatment context"),
    ("PHENYLEPHRINE", "adrenergic_rescue", "core", "vasopressor/shock treatment context"),
    ("DOPAMINE", "adrenergic_rescue", "core", "vasopressor/shock treatment context"),
    ("DOBUTAMINE", "adrenergic_rescue", "core", "vasopressor/shock treatment context"),

    # bronchodilator/asthma context
    ("ALBUTEROL", "bronchodilator_or_asthma_context", "broad", "bronchodilator/allergy-asthma context"),
    ("SALBUTAMOL", "bronchodilator_or_asthma_context", "broad", "bronchodilator/allergy-asthma context"),
    ("LEVALBUTEROL", "bronchodilator_or_asthma_context", "broad", "bronchodilator/allergy-asthma context"),
    ("TERBUTALINE", "bronchodilator_or_asthma_context", "broad", "bronchodilator/allergy-asthma context"),
    ("IPRATROPIUM", "bronchodilator_or_asthma_context", "broad", "bronchodilator/allergy-asthma context"),
    ("TIOTROPIUM", "bronchodilator_or_asthma_context", "broad", "bronchodilator/asthma-COPD context"),
    ("AMINOPHYLLINE", "bronchodilator_or_asthma_context", "broad", "bronchodilator/asthma context"),
    ("THEOPHYLLINE", "bronchodilator_or_asthma_context", "broad", "bronchodilator/asthma context"),
    ("MONTELUKAST", "bronchodilator_or_asthma_context", "broad", "asthma/allergy context"),
    ("ZAFIRLUKAST", "bronchodilator_or_asthma_context", "broad", "asthma/allergy context"),
    ("CROMOLYN", "bronchodilator_or_asthma_context", "broad", "mast-cell stabilizer/allergy context"),
]


def build_rule_table(optional_rule_path: Optional[Path]) -> pd.DataFrame:
    rules = pd.DataFrame(
        BUILTIN_RULES,
        columns=["analysis_drug_name", "confounding_category", "review_level", "comment"]
    )
    rules["analysis_drug_name_norm"] = rules["analysis_drug_name"].map(normalize_drug_name)

    if optional_rule_path is not None and optional_rule_path.exists():
        user_rules = pd.read_csv(optional_rule_path)
        required = ["analysis_drug_name", "confounding_category", "review_level"]
        missing = [c for c in required if c not in user_rules.columns]
        if missing:
            raise ValueError(f"User rule file missing columns: {missing}")

        if "comment" not in user_rules.columns:
            user_rules["comment"] = ""

        user_rules = user_rules[["analysis_drug_name", "confounding_category", "review_level", "comment"]].copy()
        user_rules["analysis_drug_name_norm"] = user_rules["analysis_drug_name"].map(normalize_drug_name)

        # User rules override built-in exact names.
        rules = rules[~rules["analysis_drug_name_norm"].isin(set(user_rules["analysis_drug_name_norm"]))].copy()
        rules = pd.concat([rules, user_rules], ignore_index=True)

    rules = rules.drop_duplicates("analysis_drug_name_norm", keep="last").reset_index(drop=True)
    return rules


def match_rule(name_norm: str, rules: pd.DataFrame) -> Tuple[bool, str, str, str, str]:
    """
    Returns:
      is_confounded, category, review_level, matched_rule, comment

    Matching:
      exact match
      or startswith rule + space/hyphen to capture salts/esters
    """
    if not name_norm:
        return False, "", "", "", ""

    # Exact
    exact = rules[rules["analysis_drug_name_norm"].eq(name_norm)]
    if len(exact):
        r = exact.iloc[0]
        return True, r["confounding_category"], r["review_level"], r["analysis_drug_name"], r.get("comment", "")

    # Prefix match for salt/ester names.
    candidates = []
    for _, r in rules.iterrows():
        rn = str(r["analysis_drug_name_norm"])
        if name_norm.startswith(rn + " ") or name_norm.startswith(rn + "-"):
            candidates.append((len(rn), r))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        r = candidates[0][1]
        return True, r["confounding_category"], r["review_level"], r["analysis_drug_name"], r.get("comment", "")

    return False, "", "", "", ""


def tag_treatment_confounding(df: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["analysis_drug_name_norm"] = out["analysis_drug_name"].map(normalize_drug_name)

    tagged = out["analysis_drug_name_norm"].apply(lambda x: match_rule(x, rules))
    out["treatment_confounded_candidate"] = [t[0] for t in tagged]
    out["treatment_confounded_category"] = [t[1] for t in tagged]
    out["treatment_review_level"] = [t[2] for t in tagged]
    out["matched_treatment_rule"] = [t[3] for t in tagged]
    out["treatment_review_comment"] = [t[4] for t in tagged]

    out["is_corticosteroid_candidate"] = out["treatment_confounded_category"].eq("corticosteroid")
    out["is_antihistamine_candidate"] = out["treatment_confounded_category"].isin(["antihistamine_h1", "antihistamine_h2"])
    out["is_core_treatment_candidate"] = out["treatment_confounded_category"].isin([
        "corticosteroid",
        "antihistamine_h1",
        "antihistamine_h2",
        "adrenergic_rescue",
    ])
    out["is_broad_treatment_candidate"] = out["treatment_confounded_candidate"]

    out["recommended_label_status"] = np.where(
        out["is_core_treatment_candidate"] & out["model_label"].astype(int).eq(1),
        "label_review_positive_core_treatment",
        np.where(
            out["is_broad_treatment_candidate"],
            "label_review_treatment_context",
            "retain"
        )
    )

    return out


# ============================================================
# 3. Splitters
# ============================================================

def get_random_splits(X, y, n_splits, n_repeats, random_state):
    min_class = int(pd.Series(y).value_counts().min())
    n_splits_eff = min(n_splits, min_class)
    if n_splits_eff < 2:
        return []

    cv = RepeatedStratifiedKFold(n_splits=n_splits_eff, n_repeats=n_repeats, random_state=random_state)
    splits = []
    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        splits.append((fold, tr, te, "random_repeated_stratified"))
    return splits


def get_sgkf_splits(X, y, groups, n_splits, n_repeats, random_state):
    if not STRATIFIED_GROUP_KFOLD_AVAILABLE:
        return []

    min_class = int(pd.Series(y).value_counts().min())
    n_splits_eff = min(n_splits, len(np.unique(groups)), min_class)
    if n_splits_eff < 2:
        return []

    splits = []
    fold_global = 0
    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state + rep)
        for tr, te in sgkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            fold_global += 1
            splits.append((fold_global, tr, te, "scaffold_stratified_group"))
    return splits


def greedy_group_split_once(y, groups, n_splits, random_state):
    rng = np.random.default_rng(random_state)
    d = pd.DataFrame({"idx": np.arange(len(y)), "y": y, "group": groups})
    gs = d.groupby("group").agg(n=("idx", "count"), n_pos=("y", "sum")).reset_index()
    gs["n_neg"] = gs["n"] - gs["n_pos"]
    gs["_rand"] = rng.random(len(gs))
    gs = gs.sort_values(["n", "_rand"], ascending=[False, True])

    n_splits_eff = min(n_splits, len(gs), int(y.sum()), int(len(y) - y.sum()))
    if n_splits_eff < 2:
        return []

    total_n = len(y)
    total_pos = int(y.sum())
    total_neg = int(len(y) - y.sum())

    target_n = total_n / n_splits_eff
    target_pos = total_pos / n_splits_eff
    target_neg = total_neg / n_splits_eff

    folds = [{"groups": [], "n": 0, "pos": 0, "neg": 0} for _ in range(n_splits_eff)]

    for _, r in gs.iterrows():
        best = None
        best_score = None
        for k in range(n_splits_eff):
            n_new = folds[k]["n"] + int(r["n"])
            p_new = folds[k]["pos"] + int(r["n_pos"])
            ng_new = folds[k]["neg"] + int(r["n_neg"])
            score = (
                abs(n_new - target_n) / max(target_n, 1)
                + abs(p_new - target_pos) / max(target_pos, 1)
                + abs(ng_new - target_neg) / max(target_neg, 1)
            )
            if best_score is None or score < best_score:
                best_score = score
                best = k

        folds[best]["groups"].append(r["group"])
        folds[best]["n"] += int(r["n"])
        folds[best]["pos"] += int(r["n_pos"])
        folds[best]["neg"] += int(r["n_neg"])

    all_idx = np.arange(len(y))
    splits = []
    for k in range(n_splits_eff):
        test_groups = set(folds[k]["groups"])
        te = d[d["group"].isin(test_groups)]["idx"].values
        tr = np.setdiff1d(all_idx, te)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        splits.append((tr, te))
    return splits


def get_greedy_splits(y, groups, n_splits, n_repeats, random_state):
    splits = []
    fold_global = 0
    for rep in range(n_repeats):
        one = greedy_group_split_once(y, groups, n_splits, random_state + rep)
        for tr, te in one:
            fold_global += 1
            splits.append((fold_global, tr, te, "scaffold_greedy_group"))
    return splits


def get_scaffold_splits(X, y, groups, n_splits, n_repeats, random_state):
    splits = get_sgkf_splits(X, y, groups, n_splits, n_repeats, random_state)
    if not splits:
        splits = get_greedy_splits(y, groups, n_splits, n_repeats, random_state)
    return splits


# ============================================================
# 4. Scenario evaluation
# ============================================================

def make_scenarios(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    scenarios = {}
    scenarios["original_all"] = df.copy()
    scenarios["exclude_corticosteroids_all"] = df[~df["is_corticosteroid_candidate"]].copy()
    scenarios["exclude_antihistamines_all"] = df[~df["is_antihistamine_candidate"]].copy()
    scenarios["exclude_core_treatment_all"] = df[~df["is_core_treatment_candidate"]].copy()
    scenarios["exclude_broad_treatment_all"] = df[~df["is_broad_treatment_candidate"]].copy()
    scenarios["exclude_core_treatment_positive_only"] = df[
        ~(df["is_core_treatment_candidate"] & df["model_label"].astype(int).eq(1))
    ].copy()
    return scenarios


def scenario_counts(scenarios: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, sdf in scenarios.items():
        rows.append({
            "scenario": name,
            "n": len(sdf),
            "n_positive": int((sdf["model_label"].astype(int) == 1).sum()),
            "n_negative": int((sdf["model_label"].astype(int) == 0).sum()),
            "n_treatment_confounded_remaining": int(sdf["treatment_confounded_candidate"].sum()),
            "n_core_treatment_remaining": int(sdf["is_core_treatment_candidate"].sum()),
            "label_classes": " | ".join(f"{k}:{v}" for k, v in sdf["faers_signal_class"].value_counts().sort_index().items()),
        })
    return pd.DataFrame(rows)


def evaluate_one_scenario(
    scenario_name: str,
    sdf: pd.DataFrame,
    radius: int,
    n_bits: int,
    use_chirality: bool,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    use_sample_weight: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if len(sdf) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    vc = sdf["model_label"].astype(int).value_counts()
    if len(vc) < 2 or vc.min() < 2:
        print(f"[WARN] Skipping {scenario_name}: insufficient class counts.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X, vdf = make_ecfp_matrix(sdf, radius, n_bits, use_chirality)
    vdf["model_label"] = vdf["model_label"].astype(int)
    vdf["label_weight"] = vdf["label_weight"].astype(float)
    vdf["scaffold_key"] = vdf.apply(get_scaffold_key, axis=1)

    y = vdf["model_label"].values
    w = vdf["label_weight"].values if use_sample_weight else None
    groups = vdf["scaffold_key"].astype(str).values

    split_sets = [
        ("random", get_random_splits(X, y, n_splits, n_repeats, random_state)),
        ("scaffold", get_scaffold_splits(X, y, groups, n_splits, n_repeats, random_state)),
    ]

    rf = get_rf(random_state)
    metric_rows = []
    pred_rows = []
    fold_rows = []

    for split_type, splits in split_sets:
        if not splits:
            print(f"[WARN] No valid splits for {scenario_name} / {split_type}")
            continue

        for fold, tr, te, split_method in splits:
            model = clone(rf)
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            wtr = w[tr] if w is not None else None
            wte = w[te] if w is not None else None

            if wtr is None:
                model.fit(Xtr, ytr)
            else:
                model.fit(Xtr, ytr, sample_weight=wtr)

            ptr = predict_proba_positive(model, Xtr)
            pte = predict_proba_positive(model, Xte)
            thr_youden = choose_youden_threshold(ytr, ptr)

            fold_rows.append({
                "scenario": scenario_name,
                "split_type": split_type,
                "fold": fold,
                "split_method": split_method,
                "n_train": len(tr),
                "n_test": len(te),
                "n_train_pos": int(ytr.sum()),
                "n_train_neg": int(len(ytr) - ytr.sum()),
                "n_test_pos": int(yte.sum()),
                "n_test_neg": int(len(yte) - yte.sum()),
                "n_train_scaffolds": len(set(groups[tr])),
                "n_test_scaffolds": len(set(groups[te])),
            })

            for threshold_type, thr in [("fixed_0.5", 0.5), ("train_youden", thr_youden)]:
                m = binary_metrics(yte, pte, float(thr), sample_weight=wte)
                m.update({
                    "scenario": scenario_name,
                    "split_type": split_type,
                    "fold": fold,
                    "split_method": split_method,
                    "threshold_type": threshold_type,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "n_train_pos": int(ytr.sum()),
                    "n_test_pos": int(yte.sum()),
                })
                metric_rows.append(m)

            for local, global_idx in enumerate(te):
                row = vdf.iloc[global_idx]
                pred_rows.append({
                    "scenario": scenario_name,
                    "split_type": split_type,
                    "fold": fold,
                    "split_method": split_method,
                    "analysis_drug_name": row.get("analysis_drug_name"),
                    "model_label": int(row.get("model_label")),
                    "label_weight": float(row.get("label_weight")),
                    "faers_signal_class": row.get("faers_signal_class"),
                    "treatment_confounded_candidate": bool(row.get("treatment_confounded_candidate")),
                    "treatment_confounded_category": row.get("treatment_confounded_category"),
                    "recommended_label_status": row.get("recommended_label_status"),
                    "scaffold_key": row.get("scaffold_key"),
                    "canonical_smiles": row.get("canonical_smiles"),
                    "prob_positive": float(pte[local]),
                    "pred_0.5": int(pte[local] >= 0.5),
                    "pred_train_youden": int(pte[local] >= thr_youden),
                    "train_youden_threshold": float(thr_youden),
                })

    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    metric_cols = [
        "roc_auc", "pr_auc", "balanced_accuracy", "accuracy",
        "sensitivity", "specificity", "precision", "f1", "mcc", "brier"
    ]

    rows = []
    for (scenario, split_type, threshold_type), g in metrics.groupby(["scenario", "split_type", "threshold_type"]):
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


def aggregate_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()

    out = (
        preds.groupby(
            [
                "scenario", "split_type", "analysis_drug_name", "model_label",
                "faers_signal_class", "treatment_confounded_candidate",
                "treatment_confounded_category", "recommended_label_status"
            ],
            dropna=False
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

    return out


# ============================================================
# 5. Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--search-dir", default="output")
    parser.add_argument("--output-dir", default="output_treatment_sensitivity")
    parser.add_argument("--rule-file", default="input/treatment_confounding_drug_rules.csv")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--no-chirality", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-sample-weight", action="store_true")
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        print("[ERROR] RDKit is not available. Install with: pip install rdkit", file=sys.stderr)
        return 1

    search_dir = Path(args.search_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input is None:
        input_path = latest_file(search_dir, "gnn_dataset_model_ready_colab_*.csv")
        if input_path is None:
            input_path = latest_file(search_dir, "gnn_dataset_model_ready_*.csv")
        if input_path is None:
            print("[ERROR] model_ready CSV not found. Specify --input.", file=sys.stderr)
            return 1
    else:
        input_path = Path(args.input)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    print(f"[INFO] Loading: {input_path}")
    df0 = pd.read_csv(input_path)
    df0["model_label"] = df0["model_label"].astype(int)
    df0["label_weight"] = df0["label_weight"].astype(float)

    rule_file = Path(args.rule_file)
    rules = build_rule_table(rule_file if rule_file.exists() else None)

    tagged = tag_treatment_confounding(df0, rules)

    print("===== Treatment-confounding candidates detected =====")
    cols = [
        "analysis_drug_name", "model_label", "faers_signal_class",
        "treatment_confounded_category", "treatment_review_level",
        "recommended_label_status"
    ]
    print(tagged[tagged["treatment_confounded_candidate"]][cols].to_string(index=False))

    scenarios = make_scenarios(tagged)
    scen_counts = scenario_counts(scenarios)

    print("===== Scenario counts =====")
    print(scen_counts.to_string(index=False))

    all_metrics = []
    all_preds = []
    all_folds = []

    for scenario, sdf in scenarios.items():
        print(f"[INFO] Evaluating scenario: {scenario}")
        print(sdf["model_label"].value_counts().sort_index().to_string())

        metrics, preds, folds = evaluate_one_scenario(
            scenario_name=scenario,
            sdf=sdf,
            radius=args.radius,
            n_bits=args.n_bits,
            use_chirality=(not args.no_chirality),
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.random_state,
            use_sample_weight=(not args.no_sample_weight),
        )

        if not metrics.empty:
            all_metrics.append(metrics)
        if not preds.empty:
            all_preds.append(preds)
        if not folds.empty:
            all_folds.append(folds)

    metrics_all = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    preds_all = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    folds_all = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()

    metric_summary = summarize_metrics(metrics_all)
    pred_summary = aggregate_predictions(preds_all)

    category_counts = (
        tagged.groupby(["treatment_confounded_candidate", "treatment_confounded_category", "model_label", "faers_signal_class"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["treatment_confounded_candidate", "treatment_confounded_category", "model_label", "faers_signal_class"])
    )

    review_candidates = tagged[tagged["treatment_confounded_candidate"]].copy()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_tagged = output_dir / f"treatment_confounded_model_ready_{ts}.csv"
    out_review_xlsx = output_dir / f"treatment_confounding_review_candidates_{ts}.xlsx"
    out_results_xlsx = output_dir / f"treatment_sensitivity_results_{ts}.xlsx"
    out_metric_summary = output_dir / f"treatment_sensitivity_metric_summary_{ts}.csv"
    out_fold_metrics = output_dir / f"treatment_sensitivity_fold_metrics_{ts}.csv"
    out_predictions = output_dir / f"treatment_sensitivity_predictions_{ts}.csv"
    out_pred_summary = output_dir / f"treatment_sensitivity_drug_probability_summary_{ts}.csv"
    out_rule_template = output_dir / "treatment_confounding_rule_template.csv"

    tagged.to_csv(out_tagged, index=False, encoding="utf-8-sig")
    metric_summary.to_csv(out_metric_summary, index=False, encoding="utf-8-sig")
    metrics_all.to_csv(out_fold_metrics, index=False, encoding="utf-8-sig")
    preds_all.to_csv(out_predictions, index=False, encoding="utf-8-sig")
    pred_summary.to_csv(out_pred_summary, index=False, encoding="utf-8-sig")
    rules[["analysis_drug_name", "confounding_category", "review_level", "comment"]].to_csv(out_rule_template, index=False, encoding="utf-8-sig")

    data_summary = pd.DataFrame([
        ["input_file", str(input_path)],
        ["rule_file_used", str(rule_file) if rule_file.exists() else "built-in rules only"],
        ["n_model_ready", len(tagged)],
        ["n_positive", int((tagged["model_label"] == 1).sum())],
        ["n_negative", int((tagged["model_label"] == 0).sum())],
        ["n_treatment_confounded_candidates", int(tagged["treatment_confounded_candidate"].sum())],
        ["n_core_treatment_candidates", int(tagged["is_core_treatment_candidate"].sum())],
        ["n_corticosteroid_candidates", int(tagged["is_corticosteroid_candidate"].sum())],
        ["n_antihistamine_candidates", int(tagged["is_antihistamine_candidate"].sum())],
        ["n_splits", args.n_splits],
        ["n_repeats", args.n_repeats],
        ["sample_weight_used", not args.no_sample_weight],
    ], columns=["item", "value"])

    with pd.ExcelWriter(out_review_xlsx, engine="openpyxl") as writer:
        sheets = {
            "01_review_candidates": review_candidates,
            "02_category_counts": category_counts,
            "03_rules": rules,
            "04_all_tagged_model_ready": tagged,
        }
        for sheet, dat in sheets.items():
            dat.to_excel(writer, sheet_name=sheet[:31], index=False)
            ws = writer.sheets[sheet[:31]]
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 70)

    with pd.ExcelWriter(out_results_xlsx, engine="openpyxl") as writer:
        sheets = {
            "01_data_summary": data_summary,
            "02_scenario_counts": scen_counts,
            "03_category_counts": category_counts,
            "04_metric_summary": metric_summary,
            "05_fold_metrics": metrics_all,
            "06_drug_probability_summary": pred_summary,
            "07_predictions": preds_all,
            "08_review_candidates": review_candidates,
            "09_all_tagged_model_ready": tagged,
            "10_rules": rules,
        }
        for sheet, dat in sheets.items():
            if dat is None or dat.empty:
                dat = pd.DataFrame({"note": ["No data available"]})
            dat.to_excel(writer, sheet_name=sheet[:31], index=False)
            ws = writer.sheets[sheet[:31]]
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 70)

    print("[INFO] Done.")
    print("===== Output files =====")
    for p in [
        out_tagged,
        out_review_xlsx,
        out_results_xlsx,
        out_metric_summary,
        out_fold_metrics,
        out_predictions,
        out_pred_summary,
        out_rule_template,
    ]:
        print(p)

    print("===== Key metric summary: ROC-AUC / PR-AUC / balanced accuracy =====")
    if not metric_summary.empty:
        print(
            metric_summary[
                metric_summary["metric"].isin(["roc_auc", "pr_auc", "balanced_accuracy"])
            ].sort_values(["scenario", "split_type", "threshold_type", "metric"]).to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
