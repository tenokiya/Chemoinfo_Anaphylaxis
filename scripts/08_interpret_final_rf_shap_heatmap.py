#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
09_interpret_final_rf_shap_heatmap_colab.py

目的:
    最終ECFP + Random Forestモデルを全model-readyデータで再学習し、
    SHAP解析および化学構造への原子レベルヒートマッピングを行う。

入力:
    output/gnn_dataset_model_ready_colab_*.csv

主な出力:
    output_interpretation/
      shap_interpretation_summary_*.xlsx
      shap_global_bit_importance_*.csv
      shap_predictions_*.csv
      shap_selected_drugs_*.csv
      shap_atom_contributions_*.csv
      shap_bit_contributions_selected_*.csv
      figure_shap_global_top_bits.png/pdf/svg
      figure_shap_probability_distribution.png/pdf/svg
      heatmaps_svg/*.svg
      heatmaps_png/*.png

注意:
    - 本解析は説明可能性・仮説生成用であり、因果的な構造アラートではありません。
    - ECFP bitはハッシュ化されており、bit collisionを含み得ます。
    - 原子寄与はactive ECFP bitのSHAP値をRDKit bitInfoのMorgan環境へ分配した近似値です。
    - 最終モデルは全データで再学習するため、ここでの予測確率は説明用です。
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


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


def normalize_name(x: Any) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().upper())


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


def mol_to_ecfp_and_bitinfo(mol, radius: int, n_bits: int, use_chirality: bool):
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


def build_ecfp_dataset(df: pd.DataFrame, radius: int, n_bits: int, use_chirality: bool):
    fps, valid_idx, bitinfos, mols, failed = [], [], [], [], []
    for i, row in df.iterrows():
        mol = smiles_to_mol(row.get("canonical_smiles"))
        if mol is None:
            failed.append(i)
            continue
        fp, bit_info = mol_to_ecfp_and_bitinfo(mol, radius, n_bits, use_chirality)
        fps.append(fp)
        valid_idx.append(i)
        bitinfos.append(bit_info)
        mols.append(mol)
    if not fps:
        raise RuntimeError("No valid molecules were parsed by RDKit.")
    vdf = df.loc[valid_idx].reset_index(drop=True).copy()
    X = np.vstack(fps).astype(np.float32)
    if failed:
        print("[WARN] Failed RDKit parsing rows:")
        cols = [c for c in ["analysis_drug_name", "canonical_smiles"] if c in df.columns]
        print(df.loc[failed, cols].to_string(index=False))
    return X, vdf, bitinfos, mols


def get_rf(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )


def get_positive_class_shap_values(explainer, X: np.ndarray):
    sv = explainer.shap_values(X)
    ev = explainer.expected_value
    if isinstance(sv, list):
        shap_pos = np.asarray(sv[1])
        expected_pos = float(ev[1]) if isinstance(ev, (list, tuple, np.ndarray)) else float(ev)
        return shap_pos, expected_pos
    sv_arr = np.asarray(sv)
    if sv_arr.ndim == 3:
        shap_pos = sv_arr[:, :, 1] if sv_arr.shape[2] >= 2 else sv_arr[:, :, 0]
    else:
        shap_pos = sv_arr
    if isinstance(ev, (list, tuple, np.ndarray)):
        ev_arr = np.asarray(ev).ravel()
        expected_pos = float(ev_arr[1]) if len(ev_arr) >= 2 else float(ev_arr[0])
    else:
        expected_pos = float(ev)
    return shap_pos, expected_pos


def summarize_global_shap(X, shap_values, df, n_bits, top_n=200):
    y = df["model_label"].astype(int).values
    pos_mask = y == 1
    neg_mask = y == 0
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_val = np.mean(shap_values, axis=0)
    pos_freq = X[pos_mask].mean(axis=0) if pos_mask.sum() > 0 else np.zeros(n_bits)
    neg_freq = X[neg_mask].mean(axis=0) if neg_mask.sum() > 0 else np.zeros(n_bits)
    all_freq = X.mean(axis=0)
    rows = []
    for bit in np.argsort(mean_abs)[::-1][:top_n]:
        bit = int(bit)
        on_idx = np.where(X[:, bit] > 0)[0]
        pos_drugs = df.iloc[on_idx][df.iloc[on_idx]["model_label"].astype(int).eq(1)]["analysis_drug_name"].astype(str).tolist()
        neg_drugs = df.iloc[on_idx][df.iloc[on_idx]["model_label"].astype(int).eq(0)]["analysis_drug_name"].astype(str).tolist()
        rows.append({
            "bit": bit,
            "mean_abs_shap": float(mean_abs[bit]),
            "mean_shap": float(mean_val[bit]),
            "all_freq": float(all_freq[bit]),
            "pos_freq": float(pos_freq[bit]),
            "neg_freq": float(neg_freq[bit]),
            "pos_minus_neg_freq": float(pos_freq[bit] - neg_freq[bit]),
            "n_on": int(len(on_idx)),
            "n_pos_on": int(len(pos_drugs)),
            "n_neg_on": int(len(neg_drugs)),
            "example_positive_drugs": " | ".join(pos_drugs[:20]),
            "example_negative_drugs": " | ".join(neg_drugs[:20]),
        })
    return pd.DataFrame(rows)


def atoms_for_morgan_env(mol, center_atom: int, radius: int) -> Set[int]:
    if radius == 0:
        return {int(center_atom)}
    env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(center_atom))
    atoms = {int(center_atom)}
    for bond_idx in env_bonds:
        bond = mol.GetBondWithIdx(int(bond_idx))
        atoms.add(int(bond.GetBeginAtomIdx()))
        atoms.add(int(bond.GetEndAtomIdx()))
    return atoms


def atom_contributions_from_bit_shap(mol, bit_info, bit_shap_values):
    n_atoms = mol.GetNumAtoms()
    atom_scores = np.zeros(n_atoms, dtype=float)
    rows = []
    for bit, envs in bit_info.items():
        if bit >= len(bit_shap_values):
            continue
        shap_val = float(bit_shap_values[int(bit)])
        if abs(shap_val) < 1e-12 or not envs:
            continue
        per_env_val = shap_val / len(envs)
        for center_atom, rad in envs:
            atoms = atoms_for_morgan_env(mol, int(center_atom), int(rad))
            if not atoms:
                continue
            per_atom_val = per_env_val / len(atoms)
            for a in atoms:
                atom_scores[int(a)] += per_atom_val
            rows.append({
                "bit": int(bit),
                "bit_shap": shap_val,
                "center_atom": int(center_atom),
                "radius": int(rad),
                "n_atoms_in_environment": int(len(atoms)),
                "environment_atoms": ",".join(map(str, sorted(atoms))),
                "per_environment_value": float(per_env_val),
                "per_atom_value": float(per_atom_val),
            })
    return atom_scores, pd.DataFrame(rows)


def score_to_color(score: float, max_abs: float):
    # positive toward positive class: red; negative toward negative class: blue; near zero: white/gray
    if max_abs <= 0 or not np.isfinite(score):
        return (0.92, 0.92, 0.92)
    v = max(min(score / max_abs, 1.0), -1.0)
    intensity = abs(v)
    if v >= 0:
        return (1.0, 1.0 - 0.70 * intensity, 1.0 - 0.70 * intensity)
    return (1.0 - 0.70 * intensity, 1.0 - 0.70 * intensity, 1.0)


def draw_atom_heatmap_svg(mol, atom_scores, out_svg: Path, width=560, height=410, legend=""):
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    mol2 = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol2)
    max_abs = float(np.max(np.abs(atom_scores))) if len(atom_scores) else 1.0
    if max_abs <= 0:
        max_abs = 1.0
    atoms = list(range(mol2.GetNumAtoms()))
    colors = {i: score_to_color(float(atom_scores[i]), max_abs) for i in atoms}
    radii = {i: 0.35 + 0.25 * min(abs(float(atom_scores[i])) / max_abs, 1.0) for i in atoms}
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.bondLineWidth = 1.5
    opts.legendFontSize = 14
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol2, legend=legend,
        highlightAtoms=atoms,
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    out_svg.write_text(drawer.GetDrawingText(), encoding="utf-8")


def draw_atom_heatmap_png(mol, atom_scores, out_png: Path, width=1400, height=1000, legend=""):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    mol2 = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol2)
    max_abs = float(np.max(np.abs(atom_scores))) if len(atom_scores) else 1.0
    if max_abs <= 0:
        max_abs = 1.0
    atoms = list(range(mol2.GetNumAtoms()))
    colors = {i: score_to_color(float(atom_scores[i]), max_abs) for i in atoms}
    radii = {i: 0.35 + 0.25 * min(abs(float(atom_scores[i])) / max_abs, 1.0) for i in atoms}
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.bondLineWidth = 3.0
    opts.legendFontSize = 38
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol2, legend=legend,
        highlightAtoms=atoms,
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    out_png.write_bytes(drawer.GetDrawingText())


def plot_global_shap(shap_global: pd.DataFrame, out_dir: Path, dpi: int, top_n: int = 30):
    if shap_global.empty:
        return
    df = shap_global.head(top_n).copy().sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.27 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["mean_abs_shap"], edgecolor="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Bit {int(b)}" for b in df["bit"]])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("ECFP bit")
    ax.set_title("Global SHAP importance of ECFP bits\nFinal Random Forest model")
    ax.grid(axis="x", linewidth=0.5, alpha=0.4)
    save_fig(fig, out_dir / "figure_shap_global_top_bits", dpi=dpi)


def plot_probability_distribution(df: pd.DataFrame, out_dir: Path, dpi: int):
    if df.empty:
        return
    tmp = df.copy()
    tmp["label_name"] = tmp["model_label"].map({0: "Negative label", 1: "Positive label"})
    groups = ["Negative label", "Positive label"]
    data = [tmp[tmp["label_name"].eq(g)]["final_rf_prob_positive"].dropna().values for g in groups]
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.boxplot(data, labels=groups, showmeans=True, patch_artist=False, widths=0.45)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=30, alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.axhline(0.5, linestyle=":", linewidth=1.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final model predicted probability for positive label")
    ax.set_xlabel("Observed weak label")
    ax.set_title("Final Random Forest predicted probabilities")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    save_fig(fig, out_dir / "figure_shap_probability_distribution", dpi=dpi)


def select_drugs_for_heatmaps(df: pd.DataFrame, explicit_drugs: List[str], top_n_total: int):
    df = df.copy()
    df["selection_reason"] = ""
    selected = []
    if explicit_drugs:
        df["_name_norm"] = df["analysis_drug_name"].map(normalize_name)
        for nm in [normalize_name(x) for x in explicit_drugs]:
            hit = df[df["_name_norm"].eq(nm)]
            if len(hit):
                idx = int(hit.index[0])
                selected.append(idx)
                df.loc[idx, "selection_reason"] = "user_specified"
    remaining = max(top_n_total - len(set(selected)), 0)
    per_bucket = max(2, remaining // 4) if remaining > 0 else 0
    buckets = [
        ("high_prob_negative", df[df["model_label"].eq(0)].sort_values("final_rf_prob_positive", ascending=False).head(per_bucket)),
        ("low_prob_positive", df[df["model_label"].eq(1)].sort_values("final_rf_prob_positive", ascending=True).head(per_bucket)),
        ("high_prob_positive", df[df["model_label"].eq(1)].sort_values("final_rf_prob_positive", ascending=False).head(per_bucket)),
        ("low_prob_negative", df[df["model_label"].eq(0)].sort_values("final_rf_prob_positive", ascending=True).head(per_bucket)),
    ]
    for reason, sub in buckets:
        for idx in sub.index:
            selected.append(int(idx))
            if df.loc[idx, "selection_reason"] == "":
                df.loc[idx, "selection_reason"] = reason
    unique = []
    seen = set()
    for idx in selected:
        if idx not in seen:
            unique.append(idx)
            seen.add(idx)
    if len(unique) < top_n_total:
        for idx in df.sort_values("final_rf_prob_positive", ascending=False).index:
            if int(idx) not in seen:
                unique.append(int(idx))
                seen.add(int(idx))
                if df.loc[idx, "selection_reason"] == "":
                    df.loc[idx, "selection_reason"] = "additional_high_probability"
            if len(unique) >= top_n_total:
                break
    out = df.loc[unique].copy()
    out = out.sort_values(["selection_reason", "final_rf_prob_positive"], ascending=[True, False]).reset_index(drop=False)
    out = out.rename(columns={"index": "row_index"})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--search-dir", default="output")
    parser.add_argument("--output-dir", default="output_interpretation")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--no-chirality", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-n-global-bits", type=int, default=200)
    parser.add_argument("--top-n-heatmaps", type=int, default=24)
    parser.add_argument("--drugs", default="")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font-size", type=int, default=11)
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        print("[ERROR] RDKit is not available. Install with: pip install rdkit", file=sys.stderr)
        return 1
    if not SHAP_AVAILABLE:
        print("[ERROR] shap is not available. Install with: pip install shap", file=sys.stderr)
        return 1

    set_style(args.font_size)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "heatmaps_svg").mkdir(exist_ok=True)
    (out_dir / "heatmaps_png").mkdir(exist_ok=True)

    if args.input is None:
        input_path = latest_file(Path(args.search_dir), "gnn_dataset_model_ready_colab_*.csv")
        if input_path is None:
            input_path = latest_file(Path(args.search_dir), "gnn_dataset_model_ready_*.csv")
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

    print("[INFO] Building ECFP matrix...")
    X, df, bitinfos, mols = build_ecfp_dataset(df0, args.radius, args.n_bits, not args.no_chirality)
    y = df["model_label"].astype(int).values
    w = df["label_weight"].astype(float).values

    print("[INFO] Training final Random Forest model on all model-ready compounds...")
    model = get_rf(args.random_state)
    model.fit(X, y, sample_weight=w)
    prob = model.predict_proba(X)[:, 1]
    df["final_rf_prob_positive"] = prob
    df["final_rf_pred_label_0_5"] = (prob >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, prob, sample_weight=w)
        ap = average_precision_score(y, prob, sample_weight=w)
    except Exception:
        auc, ap = np.nan, np.nan
    print(f"[INFO] Apparent full-data ROC-AUC: {auc:.4f}")
    print(f"[INFO] Apparent full-data PR-AUC: {ap:.4f}")
    print("[NOTE] Apparent full-data values are for interpretation, not external performance estimates.")

    print("[INFO] Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values, expected_value = get_positive_class_shap_values(explainer, X)
    if shap_values.shape != X.shape:
        raise RuntimeError(f"Unexpected SHAP shape: {shap_values.shape}; expected {X.shape}")
    df["shap_sum"] = shap_values.sum(axis=1)
    df["shap_expected_value"] = expected_value
    df["shap_expected_plus_sum"] = expected_value + df["shap_sum"]

    shap_global = summarize_global_shap(X, shap_values, df, args.n_bits, args.top_n_global_bits)
    selected = select_drugs_for_heatmaps(df, [x.strip() for x in args.drugs.split(",") if x.strip()], args.top_n_heatmaps)

    print("[INFO] Selected drugs for heatmaps:")
    print(selected[["analysis_drug_name", "model_label", "faers_signal_class", "final_rf_prob_positive", "selection_reason"]].to_string(index=False))

    atom_rows = []
    bit_rows = []
    print("[INFO] Creating atom-level heatmaps...")
    for _, srow in selected.iterrows():
        idx = int(srow["row_index"])
        mol = mols[idx]
        atom_scores, bit_contrib = atom_contributions_from_bit_shap(mol, bitinfos[idx], shap_values[idx])
        drug = str(srow["analysis_drug_name"])
        label = int(srow["model_label"])
        cls = str(srow["faers_signal_class"])
        p = float(srow["final_rf_prob_positive"])
        reason = str(srow["selection_reason"])
        legend = f"{drug} | label={label} | p={p:.2f}"
        base = f"{sanitize_filename(drug)}__label{label}__p{p:.2f}__{sanitize_filename(reason)}"
        svg_path = out_dir / "heatmaps_svg" / f"{base}.svg"
        png_path = out_dir / "heatmaps_png" / f"{base}.png"
        draw_atom_heatmap_svg(mol, atom_scores, svg_path, legend=legend)
        draw_atom_heatmap_png(mol, atom_scores, png_path, legend=legend)
        for atom_idx, score in enumerate(atom_scores):
            atom = mol.GetAtomWithIdx(int(atom_idx))
            atom_rows.append({
                "analysis_drug_name": drug,
                "model_label": label,
                "faers_signal_class": cls,
                "final_rf_prob_positive": p,
                "selection_reason": reason,
                "atom_index": int(atom_idx),
                "atom_symbol": atom.GetSymbol(),
                "atom_shap_contribution": float(score),
                "abs_atom_shap_contribution": float(abs(score)),
                "heatmap_svg": str(svg_path),
                "heatmap_png": str(png_path),
            })
        if not bit_contrib.empty:
            bit_contrib = bit_contrib.copy()
            bit_contrib["analysis_drug_name"] = drug
            bit_contrib["model_label"] = label
            bit_contrib["faers_signal_class"] = cls
            bit_contrib["final_rf_prob_positive"] = p
            bit_contrib["selection_reason"] = reason
            bit_rows.append(bit_contrib)

    atom_df = pd.DataFrame(atom_rows)
    bit_selected_df = pd.concat(bit_rows, ignore_index=True) if bit_rows else pd.DataFrame()

    plot_global_shap(shap_global, out_dir, args.dpi, min(30, args.top_n_global_bits))
    plot_probability_distribution(df, out_dir, args.dpi)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_predictions = out_dir / f"shap_predictions_{ts}.csv"
    out_global = out_dir / f"shap_global_bit_importance_{ts}.csv"
    out_selected = out_dir / f"shap_selected_drugs_{ts}.csv"
    out_atom = out_dir / f"shap_atom_contributions_{ts}.csv"
    out_bit_selected = out_dir / f"shap_bit_contributions_selected_{ts}.csv"
    out_xlsx = out_dir / f"shap_interpretation_summary_{ts}.xlsx"

    df.to_csv(out_predictions, index=False, encoding="utf-8-sig")
    shap_global.to_csv(out_global, index=False, encoding="utf-8-sig")
    selected.to_csv(out_selected, index=False, encoding="utf-8-sig")
    atom_df.to_csv(out_atom, index=False, encoding="utf-8-sig")
    bit_selected_df.to_csv(out_bit_selected, index=False, encoding="utf-8-sig")

    data_summary = pd.DataFrame([
        ["input_file", str(input_path)],
        ["n_compounds_input", len(df0)],
        ["n_compounds_valid", len(df)],
        ["n_positive", int((df["model_label"] == 1).sum())],
        ["n_negative", int((df["model_label"] == 0).sum())],
        ["radius", args.radius],
        ["n_bits", args.n_bits],
        ["use_chirality", not args.no_chirality],
        ["rf_n_estimators", 500],
        ["rf_min_samples_leaf", 2],
        ["sample_weight_used", True],
        ["apparent_full_data_roc_auc", auc],
        ["apparent_full_data_pr_auc", ap],
        ["expected_value_positive_class", expected_value],
        ["n_selected_heatmaps", len(selected)],
    ], columns=["item", "value"])

    print(f"[INFO] Writing Excel: {out_xlsx}")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        sheets = {
            "01_data_summary": data_summary,
            "02_predictions": df,
            "03_global_shap_bits": shap_global,
            "04_selected_drugs": selected,
            "05_atom_contributions": atom_df,
            "06_bit_contrib_selected": bit_selected_df,
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

    readme = f"""SHAP and atom-level heatmap interpretation outputs

Input file:
  {input_path}

Model:
  ECFP4 radius={args.radius}, nBits={args.n_bits}, useChirality={not args.no_chirality}
  RandomForestClassifier(n_estimators=500, min_samples_leaf=2)
  Trained on all model-ready compounds with label_weight as sample_weight.

Interpretation:
  SHAP values are computed for the positive class.
  Atom-level heatmaps approximate atom contributions by distributing active ECFP bit SHAP values over Morgan environments from RDKit bitInfo.
  Red indicates contribution toward the positive class; blue indicates contribution toward the negative class.
  These maps are exploratory and should not be interpreted as causal structural alerts.

Main outputs:
  {out_xlsx.name}
  {out_predictions.name}
  {out_global.name}
  {out_selected.name}
  {out_atom.name}
  {out_bit_selected.name}
  heatmaps_svg/*.svg
  heatmaps_png/*.png
"""
    (out_dir / f"README_SHAP_heatmaps_{ts}.txt").write_text(readme, encoding="utf-8")

    print("[INFO] Done.")
    for pth in [out_xlsx, out_predictions, out_global, out_selected, out_atom, out_bit_selected, out_dir / "figure_shap_global_top_bits.png", out_dir / "figure_shap_probability_distribution.png"]:
        print(pth)
    print(out_dir / "heatmaps_svg")
    print(out_dir / "heatmaps_png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
