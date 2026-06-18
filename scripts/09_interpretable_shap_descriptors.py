#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
10_interpretable_shap_descriptors_colab.py

目的:
    ECFP bitではなく、直接的に意味を解釈しやすいRDKit記述子を用いて
    Random Forestモデルを再学習し、SHAP beeswarm / bar plot / waterfall plotを作成する。

背景:
    ECFPのSHAP beeswarmは Bit 378, Bit 1160 のような表示になり、
    そのままでは化学的意味が分かりにくい。
    そこで本スクリプトでは、以下のような解釈可能な特徴量を使う。

    例:
      MolWt
      TPSA
      MolLogP
      NumHAcceptors
      NumHDonors
      NumAromaticRings
      FractionCSP3
      fr_benzene
      fr_phenol
      fr_aniline
      fr_halogen
      fr_ester
      fr_amide
      fr_urea
      fr_sulfonamd
      fr_NH2
      fr_quatN

入力:
    output/gnn_dataset_model_ready_colab_YYYYMMDD_HHMMSS.csv

出力:
    output_interpretable_shap/
      interpretable_shap_summary_YYYYMMDD_HHMMSS.xlsx
      interpretable_feature_matrix_YYYYMMDD_HHMMSS.csv
      interpretable_shap_feature_importance_YYYYMMDD_HHMMSS.csv
      interpretable_shap_predictions_YYYYMMDD_HHMMSS.csv

      figure_interpretable_shap_beeswarm.png/pdf/svg
      figure_interpretable_shap_bar.png/pdf/svg
      figure_interpretable_shap_probability_distribution.png/pdf/svg
      waterfall_plots/*.png/pdf/svg

特徴:
    - RDKit記述子名がそのまま表示されるため、beeswarmが解釈しやすい
    - Random Forest + TreeSHAP
    - label_weightをsample_weightとして使用
    - high-resolution PNG + PDF/SVG出力
    - モデル性能値は全データ再学習後のapparent値であり、性能評価用ではない

注意:
    - この解析は「解釈用モデル」であり、主性能評価はECFPモデルのrandom/scaffold splitを優先する。
    - RDKit記述子間には相関があるため、SHAP寄与は因果効果ではない。
    - 物性・部分構造記述子はECFPより単純化されており、性能より解釈性を優先する。

実行例:
    python 10_interpretable_shap_descriptors_colab.py \
      --input output/gnn_dataset_model_ready_colab_20260521_085012.csv \
      --output-dir output_interpretable_shap \
      --top-features 30 \
      --top-waterfalls 12
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import Crippen
    from rdkit.Chem import Lipinski
    from rdkit.Chem import Fragments
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


# ============================================================
# 0. Style and utilities
# ============================================================

def set_style(font_size: int = 11) -> None:
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size + 1,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size + 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def save_fig(fig, base_path: Path, dpi: int = 600) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def latest_file(output_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(output_dir.glob(pattern))
    return files[-1] if files else None


def sanitize_filename(x: str) -> str:
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x).strip())
    return x[:120]


def smiles_to_mol(smiles: Any):
    if pd.isna(smiles) or str(smiles).strip() == "":
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


# ============================================================
# 1. Interpretable descriptors
# ============================================================

def descriptor_category(name: str) -> str:
    if name.startswith("fr_"):
        return "RDKit_fragment_count"
    if name in ["MolWt", "HeavyAtomMolWt", "ExactMolWt"]:
        return "molecular_weight"
    if name in ["MolLogP", "MolMR"]:
        return "lipophilicity_refractivity"
    if name in ["TPSA"]:
        return "polar_surface_area"
    if name.startswith("Num") or name.startswith("NHOH") or name.startswith("NOCount"):
        return "atom_bond_ring_count"
    if name in ["FractionCSP3"]:
        return "saturation"
    if "Chi" in name or "Kappa" in name or "HallKier" in name:
        return "topological_shape"
    if "VSA" in name or "EState" in name:
        return "surface_estate"
    return "other_rdkit_descriptor"


def descriptor_plain_description(name: str) -> str:
    # Minimal human-readable descriptions for commonly important descriptors.
    mp = {
        "MolWt": "Molecular weight",
        "ExactMolWt": "Exact molecular weight",
        "HeavyAtomMolWt": "Heavy-atom molecular weight",
        "MolLogP": "Calculated octanol/water partition coefficient",
        "MolMR": "Molar refractivity",
        "TPSA": "Topological polar surface area",
        "NumHAcceptors": "Number of hydrogen-bond acceptors",
        "NumHDonors": "Number of hydrogen-bond donors",
        "NumAromaticRings": "Number of aromatic rings",
        "NumAliphaticRings": "Number of aliphatic rings",
        "NumSaturatedRings": "Number of saturated rings",
        "NumHeteroatoms": "Number of heteroatoms",
        "NumRotatableBonds": "Number of rotatable bonds",
        "FractionCSP3": "Fraction of sp3 carbons",
        "RingCount": "Number of rings",
        "HeavyAtomCount": "Number of heavy atoms",
        "NHOHCount": "Number of NH or OH groups",
        "NOCount": "Number of nitrogen or oxygen atoms",
    }
    if name in mp:
        return mp[name]
    if name.startswith("fr_"):
        return "RDKit fragment count: " + name.replace("fr_", "")
    return name


def get_descriptor_functions(feature_set: str = "all") -> List[Tuple[str, Any]]:
    """
    feature_set:
      - all: all RDKit descriptors from Descriptors.descList
      - core: hand-picked physicochemical descriptors + fragment counts
      - fragments: RDKit fr_* fragment count descriptors only
    """
    all_desc = list(Descriptors.descList)

    if feature_set == "all":
        return all_desc

    if feature_set == "fragments":
        return [(name, fn) for name, fn in all_desc if name.startswith("fr_")]

    if feature_set == "core":
        core_names = {
            "MolWt",
            "ExactMolWt",
            "HeavyAtomMolWt",
            "MolLogP",
            "MolMR",
            "TPSA",
            "NumHAcceptors",
            "NumHDonors",
            "NumHeteroatoms",
            "NumRotatableBonds",
            "NumAromaticRings",
            "NumAliphaticRings",
            "NumSaturatedRings",
            "RingCount",
            "HeavyAtomCount",
            "NHOHCount",
            "NOCount",
            "FractionCSP3",
            "BalabanJ",
            "BertzCT",
        }
        selected = [(name, fn) for name, fn in all_desc if name in core_names or name.startswith("fr_")]
        return selected

    raise ValueError(f"Unknown feature_set: {feature_set}")


def calculate_descriptors_for_mol(mol, desc_funcs: List[Tuple[str, Any]]) -> Dict[str, float]:
    vals = {}
    for name, fn in desc_funcs:
        try:
            v = fn(mol)
            if v is None:
                vals[name] = np.nan
            else:
                vals[name] = float(v)
        except Exception:
            vals[name] = np.nan
    return vals


def build_descriptor_matrix(
    df: pd.DataFrame,
    feature_set: str,
    min_nonzero: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Any]]:
    desc_funcs = get_descriptor_functions(feature_set)
    rows = []
    valid_idx = []
    mols = []
    failed = []

    for i, row in df.iterrows():
        mol = smiles_to_mol(row.get("canonical_smiles"))
        if mol is None:
            failed.append(i)
            continue

        vals = calculate_descriptors_for_mol(mol, desc_funcs)
        rows.append(vals)
        valid_idx.append(i)
        mols.append(mol)

    if not rows:
        raise RuntimeError("No valid RDKit molecules.")

    X = pd.DataFrame(rows)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop descriptors entirely missing.
    X = X.dropna(axis=1, how="all")

    # Drop constant or near-constant all-zero descriptors.
    keep_cols = []
    for col in X.columns:
        s = pd.to_numeric(X[col], errors="coerce")
        nunique = s.dropna().nunique()
        nonzero = (s.fillna(0) != 0).sum()
        if nunique >= 2 and nonzero >= min_nonzero:
            keep_cols.append(col)

    X = X[keep_cols].copy()

    vdf = df.loc[valid_idx].reset_index(drop=True).copy()

    if failed:
        print("[WARN] Failed RDKit parsing rows:")
        cols = [c for c in ["analysis_drug_name", "canonical_smiles"] if c in df.columns]
        print(df.loc[failed, cols].to_string(index=False))

    return X, vdf, mols


# ============================================================
# 2. SHAP handling
# ============================================================

def get_positive_class_shap_values(explainer, X_arr: np.ndarray) -> Tuple[np.ndarray, float]:
    sv = explainer.shap_values(X_arr)
    ev = explainer.expected_value

    if isinstance(sv, list):
        shap_pos = np.asarray(sv[1])
        expected_pos = float(ev[1]) if isinstance(ev, (list, tuple, np.ndarray)) else float(ev)
        return shap_pos, expected_pos

    sv_arr = np.asarray(sv)
    if sv_arr.ndim == 3:
        if sv_arr.shape[2] >= 2:
            shap_pos = sv_arr[:, :, 1]
        else:
            shap_pos = sv_arr[:, :, 0]
    else:
        shap_pos = sv_arr

    if isinstance(ev, (list, tuple, np.ndarray)):
        ev_arr = np.asarray(ev).ravel()
        expected_pos = float(ev_arr[1]) if len(ev_arr) >= 2 else float(ev_arr[0])
    else:
        expected_pos = float(ev)

    return shap_pos, expected_pos


def summarize_shap_features(
    X_df: pd.DataFrame,
    shap_values: np.ndarray,
    df: pd.DataFrame,
    top_n: int = 200,
) -> pd.DataFrame:
    y = df["model_label"].astype(int).values
    pos_mask = y == 1
    neg_mask = y == 0

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)

    rows = []
    for idx in np.argsort(mean_abs)[::-1][:top_n]:
        feat = X_df.columns[int(idx)]
        vals = pd.to_numeric(X_df[feat], errors="coerce")
        pos_vals = vals[pos_mask]
        neg_vals = vals[neg_mask]

        rows.append({
            "feature": feat,
            "feature_description": descriptor_plain_description(feat),
            "feature_category": descriptor_category(feat),
            "mean_abs_shap": float(mean_abs[idx]),
            "mean_shap": float(mean_shap[idx]),
            "mean_value_all": float(vals.mean()),
            "mean_value_positive": float(pos_vals.mean()) if len(pos_vals) else np.nan,
            "mean_value_negative": float(neg_vals.mean()) if len(neg_vals) else np.nan,
            "positive_minus_negative_mean": float(pos_vals.mean() - neg_vals.mean()) if len(pos_vals) and len(neg_vals) else np.nan,
            "nonzero_count": int((vals.fillna(0) != 0).sum()),
        })

    return pd.DataFrame(rows)


# ============================================================
# 3. Plotting
# ============================================================

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    expected_value: float,
    out_dir: Path,
    max_display: int,
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(expected_value, X_df.shape[0]),
        data=X_df.values,
        feature_names=list(X_df.columns),
    )

    # Beeswarm
    plt.figure(figsize=(10.5, max(7, 0.34 * max_display)))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title("SHAP beeswarm for interpretable RDKit descriptors")
    plt.tight_layout()
    fig = plt.gcf()
    save_fig(fig, out_dir / "figure_interpretable_shap_beeswarm", dpi=dpi)

    # Bar
    plt.figure(figsize=(10.0, max(6, 0.30 * max_display)))
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title("Mean absolute SHAP values for interpretable RDKit descriptors")
    plt.tight_layout()
    fig = plt.gcf()
    save_fig(fig, out_dir / "figure_interpretable_shap_bar", dpi=dpi)


def plot_probability_distribution(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    df = df.copy()
    df["label_name"] = df["model_label"].map({1: "Positive label", 0: "Negative label", 1.0: "Positive label", 0.0: "Negative label"})
    groups = ["Negative label", "Positive label"]
    data = [df[df["label_name"].eq(g)]["final_descriptor_rf_prob_positive"].dropna().values for g in groups]

    ax.boxplot(data, labels=groups, showmeans=True, patch_artist=False, widths=0.45)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=30, alpha=0.75, edgecolor="black", linewidth=0.4)

    ax.axhline(0.5, linestyle=":", linewidth=1.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final descriptor model predicted probability for positive label")
    ax.set_xlabel("Observed weak label")
    ax.set_title("Final descriptor-based Random Forest predicted probabilities")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    save_fig(fig, out_dir / "figure_interpretable_shap_probability_distribution", dpi=dpi)


def plot_waterfalls(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    df: pd.DataFrame,
    expected_value: float,
    out_dir: Path,
    max_display: int,
    top_n: int,
    explicit_drugs: List[str],
    dpi: int,
) -> pd.DataFrame:
    water_dir = out_dir / "waterfall_plots"
    water_dir.mkdir(parents=True, exist_ok=True)

    # Select examples: high prob negatives, low prob positives, high prob positives, explicit drugs.
    selected = []

    if explicit_drugs:
        name_norm = {re.sub(r"\s+", " ", x.strip().upper()) for x in explicit_drugs}
        for i, name in enumerate(df["analysis_drug_name"].astype(str)):
            if re.sub(r"\s+", " ", name.strip().upper()) in name_norm:
                selected.append((i, "user_specified"))

    buckets = [
        (df[df["model_label"].eq(0)].sort_values("final_descriptor_rf_prob_positive", ascending=False).head(max(2, top_n // 4)), "high_prob_negative"),
        (df[df["model_label"].eq(1)].sort_values("final_descriptor_rf_prob_positive", ascending=True).head(max(2, top_n // 4)), "low_prob_positive"),
        (df[df["model_label"].eq(1)].sort_values("final_descriptor_rf_prob_positive", ascending=False).head(max(2, top_n // 4)), "high_prob_positive"),
        (df[df["model_label"].eq(0)].sort_values("final_descriptor_rf_prob_positive", ascending=True).head(max(2, top_n // 4)), "low_prob_negative"),
    ]

    for sub, reason in buckets:
        for idx in sub.index:
            selected.append((int(idx), reason))

    # Deduplicate.
    selected_unique = []
    seen = set()
    for idx, reason in selected:
        if idx not in seen:
            selected_unique.append((idx, reason))
            seen.add(idx)
        if len(selected_unique) >= top_n:
            break

    rows = []
    for idx, reason in selected_unique:
        row = df.iloc[idx]
        exp = shap.Explanation(
            values=shap_values[idx],
            base_values=expected_value,
            data=X_df.iloc[idx].values,
            feature_names=list(X_df.columns),
        )

        drug = str(row["analysis_drug_name"])
        p = float(row["final_descriptor_rf_prob_positive"])
        label = int(row["model_label"])
        cls = str(row.get("faers_signal_class", ""))

        plt.figure(figsize=(9.5, 6.5))
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        plt.title(f"{drug} | label={label} | p={p:.2f}")
        plt.tight_layout()
        fig = plt.gcf()

        base = f"{sanitize_filename(drug)}__label{label}__p{p:.2f}__{sanitize_filename(reason)}"
        save_fig(fig, water_dir / base, dpi=dpi)

        rows.append({
            "analysis_drug_name": drug,
            "model_label": label,
            "faers_signal_class": cls,
            "final_descriptor_rf_prob_positive": p,
            "selection_reason": reason,
            "waterfall_file_stem": base,
        })

    return pd.DataFrame(rows)


# ============================================================
# 4. Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="model_ready CSV. If omitted, latest output/gnn_dataset_model_ready_colab_*.csv is used.")
    parser.add_argument("--search-dir", default="output")
    parser.add_argument("--output-dir", default="output_interpretable_shap")
    parser.add_argument("--feature-set", default="core", choices=["core", "all", "fragments"])
    parser.add_argument("--min-nonzero", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-features", type=int, default=30)
    parser.add_argument("--top-waterfalls", type=int, default=12)
    parser.add_argument("--waterfall-features", type=int, default=15)
    parser.add_argument("--drugs", default="", help='Comma-separated drug names to force waterfall plots, e.g. "AZACITIDINE,ONDANSETRON"')
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font-size", type=int, default=11)
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        print("[ERROR] RDKit is not available. Install with: pip install rdkit", file=sys.stderr)
        return 1
    if not SHAP_AVAILABLE:
        print("[ERROR] SHAP is not available. Install with: pip install shap", file=sys.stderr)
        return 1

    set_style(args.font_size)

    search_dir = Path(args.search_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    required = ["analysis_drug_name", "canonical_smiles", "model_label", "label_weight", "faers_signal_class"]
    missing = [c for c in required if c not in df0.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}", file=sys.stderr)
        return 1

    df0["model_label"] = df0["model_label"].astype(int)
    df0["label_weight"] = df0["label_weight"].astype(float)

    print(f"[INFO] Building RDKit descriptor matrix: feature_set={args.feature_set}")
    X_raw, df, mols = build_descriptor_matrix(
        df0,
        feature_set=args.feature_set,
        min_nonzero=args.min_nonzero,
    )

    feature_names = list(X_raw.columns)
    print(f"[INFO] Valid compounds: {len(df)}")
    print(f"[INFO] Number of descriptors after filtering: {len(feature_names)}")
    print(df["model_label"].value_counts().sort_index().to_string())

    y = df["model_label"].astype(int).values
    w = df["label_weight"].astype(float).values

    # Impute missing values.
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X_raw)
    X_imp = pd.DataFrame(X_arr, columns=feature_names)

    print("[INFO] Training descriptor-based final Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=args.random_state,
        n_jobs=-1,
    )
    rf.fit(X_arr, y, sample_weight=w)

    prob = rf.predict_proba(X_arr)[:, 1]
    pred = (prob >= 0.5).astype(int)

    df = df.copy()
    df["final_descriptor_rf_prob_positive"] = prob
    df["final_descriptor_rf_pred_0_5"] = pred

    try:
        auc = roc_auc_score(y, prob, sample_weight=w)
        ap = average_precision_score(y, prob, sample_weight=w)
        bacc = balanced_accuracy_score(y, pred, sample_weight=w)
        acc = accuracy_score(y, pred, sample_weight=w)
    except Exception:
        auc, ap, bacc, acc = np.nan, np.nan, np.nan, np.nan

    print(f"[INFO] Apparent full-data ROC-AUC: {auc:.4f}")
    print(f"[INFO] Apparent full-data PR-AUC: {ap:.4f}")
    print("[NOTE] These are apparent full-data values for interpretation, not external performance estimates.")

    print("[INFO] Computing TreeSHAP values...")
    explainer = shap.TreeExplainer(rf)
    shap_values, expected_value = get_positive_class_shap_values(explainer, X_arr)

    if shap_values.shape != X_arr.shape:
        raise RuntimeError(f"Unexpected SHAP shape: {shap_values.shape}; expected {X_arr.shape}")

    shap_summary = summarize_shap_features(
        X_df=X_imp,
        shap_values=shap_values,
        df=df,
        top_n=300,
    )

    # Add top feature columns to predictions.
    df["shap_sum"] = shap_values.sum(axis=1)
    df["shap_expected_value"] = expected_value
    df["shap_expected_plus_sum"] = expected_value + df["shap_sum"]

    print("[INFO] Creating beeswarm and bar plots...")
    plot_shap_beeswarm(
        shap_values=shap_values,
        X_df=X_imp,
        expected_value=expected_value,
        out_dir=out_dir,
        max_display=args.top_features,
        dpi=args.dpi,
    )

    print("[INFO] Creating probability distribution plot...")
    plot_probability_distribution(df, out_dir, dpi=args.dpi)

    print("[INFO] Creating waterfall plots...")
    explicit_drugs = [x.strip() for x in args.drugs.split(",") if x.strip()]
    waterfall_selected = plot_waterfalls(
        shap_values=shap_values,
        X_df=X_imp,
        df=df,
        expected_value=expected_value,
        out_dir=out_dir,
        max_display=args.waterfall_features,
        top_n=args.top_waterfalls,
        explicit_drugs=explicit_drugs,
        dpi=args.dpi,
    )

    # Long SHAP matrix for top features.
    top_features = shap_summary["feature"].head(args.top_features).tolist()
    top_idx = [feature_names.index(f) for f in top_features if f in feature_names]
    shap_long_rows = []
    for i, row in df.iterrows():
        for feat, j in zip(top_features, top_idx):
            shap_long_rows.append({
                "analysis_drug_name": row["analysis_drug_name"],
                "model_label": int(row["model_label"]),
                "faers_signal_class": row["faers_signal_class"],
                "final_descriptor_rf_prob_positive": float(row["final_descriptor_rf_prob_positive"]),
                "feature": feat,
                "feature_description": descriptor_plain_description(feat),
                "feature_value": float(X_imp.iloc[i, j]),
                "shap_value": float(shap_values[i, j]),
                "abs_shap_value": float(abs(shap_values[i, j])),
            })
    shap_long = pd.DataFrame(shap_long_rows)

    # Descriptor metadata.
    feature_meta = pd.DataFrame([
        {
            "feature": f,
            "feature_description": descriptor_plain_description(f),
            "feature_category": descriptor_category(f),
        }
        for f in feature_names
    ])

    # Outputs.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = out_dir / f"interpretable_shap_summary_{ts}.xlsx"
    out_feature_matrix = out_dir / f"interpretable_feature_matrix_{ts}.csv"
    out_predictions = out_dir / f"interpretable_shap_predictions_{ts}.csv"
    out_importance = out_dir / f"interpretable_shap_feature_importance_{ts}.csv"
    out_shap_long = out_dir / f"interpretable_shap_long_top_features_{ts}.csv"
    out_waterfall = out_dir / f"interpretable_waterfall_selected_drugs_{ts}.csv"
    out_meta = out_dir / f"interpretable_feature_metadata_{ts}.csv"

    X_out = pd.concat(
        [
            df[["analysis_drug_name", "model_label", "label_weight", "faers_signal_class", "canonical_smiles"]].reset_index(drop=True),
            X_imp.reset_index(drop=True),
        ],
        axis=1,
    )

    X_out.to_csv(out_feature_matrix, index=False, encoding="utf-8-sig")
    df.to_csv(out_predictions, index=False, encoding="utf-8-sig")
    shap_summary.to_csv(out_importance, index=False, encoding="utf-8-sig")
    shap_long.to_csv(out_shap_long, index=False, encoding="utf-8-sig")
    waterfall_selected.to_csv(out_waterfall, index=False, encoding="utf-8-sig")
    feature_meta.to_csv(out_meta, index=False, encoding="utf-8-sig")

    data_summary = pd.DataFrame([
        ["input_file", str(input_path)],
        ["n_compounds_input", len(df0)],
        ["n_compounds_valid", len(df)],
        ["n_positive", int((df["model_label"] == 1).sum())],
        ["n_negative", int((df["model_label"] == 0).sum())],
        ["feature_set", args.feature_set],
        ["n_descriptors_after_filtering", len(feature_names)],
        ["min_nonzero", args.min_nonzero],
        ["rf_n_estimators", 500],
        ["rf_min_samples_leaf", 2],
        ["sample_weight_used", True],
        ["apparent_full_data_roc_auc", auc],
        ["apparent_full_data_pr_auc", ap],
        ["apparent_full_data_balanced_accuracy", bacc],
        ["apparent_full_data_accuracy", acc],
        ["expected_value_positive_class", expected_value],
    ], columns=["item", "value"])

    print(f"[INFO] Writing Excel: {out_xlsx}")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        sheets = {
            "01_data_summary": data_summary,
            "02_feature_importance": shap_summary,
            "03_predictions": df,
            "04_shap_long_top_features": shap_long,
            "05_waterfall_selected": waterfall_selected,
            "06_feature_metadata": feature_meta,
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

    readme = f"""Interpretable SHAP analysis using RDKit descriptors

Input:
  {input_path}

Model:
  RandomForestClassifier(n_estimators=500, min_samples_leaf=2)
  Features: RDKit descriptors, feature_set={args.feature_set}
  Sample weights: label_weight

Important interpretation:
  This is an interpretation-oriented descriptor model.
  It is not the primary performance model.
  The main performance model remains the ECFP-based Random Forest evaluated by random and scaffold splits.
  Beeswarm features are directly interpretable RDKit descriptors or fragment-count descriptors.
  SHAP values describe associations learned by the final model, not causal effects.

Main outputs:
  {out_xlsx.name}
  {out_feature_matrix.name}
  {out_predictions.name}
  {out_importance.name}
  {out_shap_long.name}
  figure_interpretable_shap_beeswarm.png/pdf/svg
  figure_interpretable_shap_bar.png/pdf/svg
  waterfall_plots/*.png/pdf/svg
"""
    (out_dir / f"README_interpretable_SHAP_{ts}.txt").write_text(readme, encoding="utf-8")

    print("[INFO] Done.")
    print("===== Output files =====")
    for p in [
        out_xlsx,
        out_feature_matrix,
        out_predictions,
        out_importance,
        out_shap_long,
        out_waterfall,
        out_meta,
        out_dir / "figure_interpretable_shap_beeswarm.png",
        out_dir / "figure_interpretable_shap_bar.png",
        out_dir / "figure_interpretable_shap_probability_distribution.png",
    ]:
        print(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
