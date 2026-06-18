#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04b_make_gnn_dataset_from_exported_faers_files_colab.py

目的:
    ローカルPostgreSQLからエクスポート済みのCSVを用いて、
    Google Colab上でGNN/ECFP用データセットを作成する。

入力:
    input/drug_structure_master.xlsx
    faers_minimal_export/latest_case_flags.csv
    faers_minimal_export/drug_clean.csv

実行例:
    python 04b_make_gnn_dataset_from_exported_faers_files_colab.py \
      --structure-master input/drug_structure_master.xlsx \
      --faers-export-dir faers_minimal_export \
      --output-dir output
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 0. ラベル条件
# ============================================================

HIGH_MIN_A = 5
HIGH_MIN_ROR = 2.0

MODERATE_MIN_A = 3
MODERATE_MIN_ROR = 2.0

NEG_HIGH_MIN_TOTAL_EXPOSED = 2000
NEG_HIGH_MIN_EXPECTED_SHOCK = 2.0
NEG_HIGH_MIN_EXPECTED_BROAD = 2.0

NEG_MOD_MIN_TOTAL_EXPOSED = 1000
NEG_MOD_MIN_EXPECTED_SHOCK = 1.0
NEG_MOD_MIN_EXPECTED_BROAD = 1.0

NEG_EXP_MIN_TOTAL_EXPOSED = 500
NEG_EXP_MIN_EXPECTED_BROAD = 2.0

WEIGHT_POS_HIGH = 1.0
WEIGHT_POS_MODERATE = 0.5
WEIGHT_NEG_HIGH = 1.0
WEIGHT_NEG_MODERATE = 0.8
WEIGHT_NEG_EXPANDED = 0.6


# ============================================================
# 1. 補助関数
# ============================================================

def norm_upper(x: Any) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().upper())


def parse_bool(x: Any) -> Optional[bool]:
    if pd.isna(x):
        return None
    s = norm_upper(x)
    if s in {"TRUE", "T", "YES", "Y", "1", "IN_SCOPE", "SCOPE"}:
        return True
    if s in {"FALSE", "F", "NO", "N", "0", "OUT_OF_SCOPE", "OUT"}:
        return False
    if isinstance(x, bool):
        return x
    return None


def first_nonempty(series: pd.Series) -> Any:
    vals = [v for v in series.tolist() if not pd.isna(v) and str(v).strip() != ""]
    return vals[0] if vals else pd.NA


def standardize_structure_master(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(path)
    raw.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in raw.columns]

    required_cols = [
        "drugname_upper",
        "ingredient",
        "pubchem_cid",
        "canonical_smiles",
        "inchikey",
        "in_scope",
        "exclude_reason",
    ]
    for col in required_cols:
        if col not in raw.columns:
            raw[col] = pd.NA

    raw["drugname_upper"] = raw["drugname_upper"].map(norm_upper)
    raw["ingredient"] = raw["ingredient"].map(norm_upper)
    raw.loc[raw["ingredient"].eq(""), "ingredient"] = raw.loc[raw["ingredient"].eq(""), "drugname_upper"]
    raw["in_scope_bool"] = raw["in_scope"].map(parse_bool)
    raw["inchikey"] = raw["inchikey"].map(norm_upper)

    raw = raw[raw["drugname_upper"].ne("")].copy()
    raw = raw.drop_duplicates(subset=["drugname_upper", "ingredient"], keep="first")

    map_df = raw[["drugname_upper", "ingredient"]].drop_duplicates()

    info = (
        raw.groupby("ingredient", dropna=False)
        .agg(
            pubchem_cid=("pubchem_cid", first_nonempty),
            canonical_smiles=("canonical_smiles", first_nonempty),
            inchikey=("inchikey", first_nonempty),
            in_scope=("in_scope_bool", lambda x: None if x.dropna().empty else bool(x.dropna().all())),
            exclude_reason=("exclude_reason", first_nonempty),
            n_raw_names_mapped=("drugname_upper", "nunique"),
            raw_names_mapped=("drugname_upper", lambda x: " | ".join(sorted(set(x)))),
        )
        .reset_index()
        .rename(columns={"ingredient": "analysis_drug_name"})
    )

    return map_df, info


def calc_ror(summary: pd.DataFrame, total_event: float, total_nonevent: float, suffix: str) -> pd.DataFrame:
    out = summary.copy()
    out["a"] = pd.to_numeric(out["a"], errors="coerce").fillna(0).astype(float)
    out["b"] = pd.to_numeric(out["b"], errors="coerce").fillna(0).astype(float)
    out["c"] = float(total_event) - out["a"]
    out["d"] = float(total_nonevent) - out["b"]

    zero = (out["a"] == 0) | (out["b"] == 0) | (out["c"] == 0) | (out["d"] == 0)
    aa = np.where(zero, out["a"] + 0.5, out["a"])
    bb = np.where(zero, out["b"] + 0.5, out["b"])
    cc = np.where(zero, out["c"] + 0.5, out["c"])
    dd = np.where(zero, out["d"] + 0.5, out["d"])

    out[f"ror_{suffix}"] = (aa * dd) / (bb * cc)
    out[f"se_logror_{suffix}"] = np.sqrt((1 / aa) + (1 / bb) + (1 / cc) + (1 / dd))
    out[f"lower_ci_{suffix}"] = np.exp(np.log(out[f"ror_{suffix}"]) - 1.96 * out[f"se_logror_{suffix}"])
    out[f"upper_ci_{suffix}"] = np.exp(np.log(out[f"ror_{suffix}"]) + 1.96 * out[f"se_logror_{suffix}"])
    out[f"total_exposed_{suffix}"] = out["a"] + out["b"]

    out = out.rename(columns={
        "a": f"a_{suffix}",
        "b": f"b_{suffix}",
        "c": f"c_{suffix}",
        "d": f"d_{suffix}",
    })

    return out[
        [
            "analysis_drug_name",
            f"a_{suffix}",
            f"b_{suffix}",
            f"c_{suffix}",
            f"d_{suffix}",
            f"total_exposed_{suffix}",
            f"ror_{suffix}",
            f"lower_ci_{suffix}",
            f"upper_ci_{suffix}",
            f"se_logror_{suffix}",
        ]
    ]


def summarize_exposure(exposure: pd.DataFrame, flags: pd.DataFrame, event_col: str) -> pd.DataFrame:
    x = exposure.merge(flags[["primaryid", event_col]], on="primaryid", how="left")
    x[event_col] = x[event_col].fillna(0).astype(int)
    g = (
        x.groupby("analysis_drug_name", dropna=False)
        .agg(total=("primaryid", "nunique"), a=(event_col, "sum"))
        .reset_index()
    )
    g["b"] = g["total"] - g["a"]
    return g[["analysis_drug_name", "a", "b"]]


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
# 2. メイン
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-master", default="input/drug_structure_master.xlsx")
    parser.add_argument("--faers-export-dir", default="faers_minimal_export")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    structure_master = Path(args.structure_master)
    export_dir = Path(args.faers_export_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flags_path = export_dir / "latest_case_flags.csv"
    drug_path = export_dir / "drug_clean.csv"

    for p in [structure_master, flags_path, drug_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print("[INFO] Loading structure master...")
    map_df, structure_info = standardize_structure_master(structure_master)
    print(f"[INFO] map rows: {len(map_df)}")
    print(f"[INFO] ingredient records: {len(structure_info)}")

    print("[INFO] Loading case flags...")
    flags = pd.read_csv(flags_path, dtype={"primaryid": str})
    flags["strict_shock"] = pd.to_numeric(flags["strict_shock"], errors="coerce").fillna(0).astype(int)
    flags["broad_anaphylaxis"] = pd.to_numeric(flags["broad_anaphylaxis"], errors="coerce").fillna(0).astype(int)

    n_latest = len(flags)
    n_strict = int(flags["strict_shock"].sum())
    n_broad = int(flags["broad_anaphylaxis"].sum())
    n_nonevent_strict = n_latest - n_strict
    strict_rate = n_strict / n_latest
    broad_rate = n_broad / n_latest

    print(f"[INFO] n_latest_cases: {n_latest}")
    print(f"[INFO] n_strict_shock: {n_strict}")
    print(f"[INFO] n_broad_anaphylaxis: {n_broad}")

    print("[INFO] Loading drug_clean.csv...")
    drug = pd.read_csv(drug_path, dtype={"primaryid": str})
    drug["drugname_upper"] = drug["drugname_upper"].map(norm_upper)
    drug["role_cod"] = drug["role_cod"].map(norm_upper)

    print(f"[INFO] drug rows: {len(drug)}")

    print("[INFO] Applying drug name mapping...")
    drug = drug.merge(map_df, on="drugname_upper", how="left")
    drug["analysis_drug_name"] = drug["ingredient"].fillna(drug["drugname_upper"])
    drug = drug.drop(columns=["ingredient"])
    drug = drug.drop_duplicates(subset=["primaryid", "analysis_drug_name", "role_cod"])

    exposure_ps = (
        drug[drug["role_cod"].eq("PS")][["primaryid", "analysis_drug_name"]]
        .drop_duplicates()
    )
    exposure_psss = (
        drug[drug["role_cod"].isin(["PS", "SS"])][["primaryid", "analysis_drug_name"]]
        .drop_duplicates()
    )

    print(f"[INFO] exposure PS rows: {len(exposure_ps)}")
    print(f"[INFO] exposure PS+SS rows: {len(exposure_psss)}")

    print("[INFO] Calculating ROR...")
    summary_ps_raw = summarize_exposure(exposure_ps, flags, "strict_shock")
    summary_psss_raw = summarize_exposure(exposure_psss, flags, "strict_shock")

    summary_ps = calc_ror(summary_ps_raw, n_strict, n_nonevent_strict, "ps")
    summary_psss = calc_ror(summary_psss_raw, n_strict, n_nonevent_strict, "psss")

    print("[INFO] Calculating broad counts...")
    broad_x = exposure_psss.merge(
        flags[["primaryid", "strict_shock", "broad_anaphylaxis"]],
        on="primaryid",
        how="left",
    )
    broad_x[["strict_shock", "broad_anaphylaxis"]] = broad_x[["strict_shock", "broad_anaphylaxis"]].fillna(0).astype(int)

    broad_counts = (
        broad_x.groupby("analysis_drug_name", dropna=False)
        .agg(
            total_exposed_psss_for_broad=("primaryid", "nunique"),
            shock_count_psss=("strict_shock", "sum"),
            broad_count_psss=("broad_anaphylaxis", "sum"),
        )
        .reset_index()
    )

    print("[INFO] Merging...")
    label = summary_psss.merge(summary_ps, on="analysis_drug_name", how="outer")
    label = label.merge(broad_counts, on="analysis_drug_name", how="left")
    label = label.merge(structure_info, on="analysis_drug_name", how="left")

    num_cols = [
        "a_psss", "b_psss", "c_psss", "d_psss", "total_exposed_psss",
        "a_ps", "b_ps", "c_ps", "d_ps", "total_exposed_ps",
        "total_exposed_psss_for_broad", "shock_count_psss", "broad_count_psss",
    ]
    for col in num_cols:
        if col in label.columns:
            label[col] = pd.to_numeric(label[col], errors="coerce").fillna(0)

    label["expected_shock"] = label["total_exposed_psss_for_broad"] * strict_rate
    label["expected_broad"] = label["total_exposed_psss_for_broad"] * broad_rate

    label["positive_high_flag"] = (
        (label["a_ps"] >= HIGH_MIN_A) &
        (label["ror_ps"] >= HIGH_MIN_ROR) &
        (label["lower_ci_ps"] > 1)
    )

    label["positive_moderate_flag"] = (
        (~label["positive_high_flag"]) &
        (label["a_psss"] >= MODERATE_MIN_A) &
        (label["ror_psss"] >= MODERATE_MIN_ROR) &
        (label["lower_ci_psss"] > 1)
    )

    label["negative_high_flag"] = (
        (~label["positive_high_flag"]) &
        (~label["positive_moderate_flag"]) &
        (label["total_exposed_psss_for_broad"] >= NEG_HIGH_MIN_TOTAL_EXPOSED) &
        (label["expected_shock"] >= NEG_HIGH_MIN_EXPECTED_SHOCK) &
        (label["expected_broad"] >= NEG_HIGH_MIN_EXPECTED_BROAD) &
        (label["shock_count_psss"] == 0) &
        (label["broad_count_psss"] == 0)
    )

    label["negative_moderate_flag"] = (
        (~label["positive_high_flag"]) &
        (~label["positive_moderate_flag"]) &
        (~label["negative_high_flag"]) &
        (label["total_exposed_psss_for_broad"] >= NEG_MOD_MIN_TOTAL_EXPOSED) &
        (label["expected_shock"] >= NEG_MOD_MIN_EXPECTED_SHOCK) &
        (label["expected_broad"] >= NEG_MOD_MIN_EXPECTED_BROAD) &
        (label["shock_count_psss"] == 0) &
        (label["broad_count_psss"] == 0)
    )

    label["negative_expanded_flag"] = (
        (~label["positive_high_flag"]) &
        (~label["positive_moderate_flag"]) &
        (~label["negative_high_flag"]) &
        (~label["negative_moderate_flag"]) &
        (label["total_exposed_psss_for_broad"] >= NEG_EXP_MIN_TOTAL_EXPOSED) &
        (label["expected_broad"] >= NEG_EXP_MIN_EXPECTED_BROAD) &
        (label["shock_count_psss"] == 0) &
        (label["broad_count_psss"] == 0)
    )

    label["faers_signal_class"] = np.select(
        [
            label["positive_high_flag"],
            label["positive_moderate_flag"],
            label["negative_high_flag"],
            label["negative_moderate_flag"],
            label["negative_expanded_flag"],
        ],
        [
            "positive_high",
            "positive_moderate",
            "negative_high",
            "negative_moderate",
            "negative_expanded",
        ],
        default="uncertain",
    )

    label["structure_available"] = label["canonical_smiles"].notna() & label["canonical_smiles"].astype(str).str.strip().ne("")
    label["in_scope_bool"] = label["in_scope"].map(lambda x: bool(x) if isinstance(x, bool) else parse_bool(x))
    label["usable_for_model"] = label["structure_available"] & (label["in_scope_bool"] == True)

    label["model_label"] = np.nan
    label.loc[
        label["faers_signal_class"].isin(["positive_high", "positive_moderate"]) &
        label["usable_for_model"],
        "model_label"
    ] = 1
    label.loc[
        label["faers_signal_class"].isin(["negative_high", "negative_moderate", "negative_expanded"]) &
        label["usable_for_model"],
        "model_label"
    ] = 0

    weight_map = {
        "positive_high": WEIGHT_POS_HIGH,
        "positive_moderate": WEIGHT_POS_MODERATE,
        "negative_high": WEIGHT_NEG_HIGH,
        "negative_moderate": WEIGHT_NEG_MODERATE,
        "negative_expanded": WEIGHT_NEG_EXPANDED,
    }
    label["label_weight"] = label["faers_signal_class"].map(weight_map)
    label.loc[~label["usable_for_model"], "label_weight"] = np.nan

    model_ready = label[label["model_label"].notna()].copy()
    model_ready = model_ready.sort_values(
        ["model_label", "label_weight", "analysis_drug_name"],
        ascending=[False, False, True]
    )

    label_counts = label["faers_signal_class"].value_counts(dropna=False).rename_axis("faers_signal_class").reset_index(name="n_drugs")

    model_counts = (
        model_ready.groupby(["model_label", "faers_signal_class"], dropna=False)
        .size()
        .reset_index(name="n_drugs")
    )

    scope_counts = (
        label.groupby(["faers_signal_class", "usable_for_model"], dropna=False)
        .size()
        .reset_index(name="n_drugs")
    )

    qa = pd.DataFrame([
        ["n_latest_cases", n_latest],
        ["n_strict_shock_cases", n_strict],
        ["n_broad_anaphylaxis_cases", n_broad],
        ["strict_shock_rate", strict_rate],
        ["broad_anaphylaxis_rate", broad_rate],
        ["n_drug_rows", len(drug)],
        ["n_all_label_rows", len(label)],
        ["n_model_ready_rows", len(model_ready)],
        ["n_model_positive", int((model_ready["model_label"] == 1).sum())],
        ["n_model_negative", int((model_ready["model_label"] == 0).sum())],
        ["n_model_missing_smiles", int(model_ready["canonical_smiles"].isna().sum())],
        ["n_model_not_in_scope", int((model_ready["in_scope_bool"] != True).sum())],
    ], columns=["item", "value"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = output_dir / f"gnn_dataset_anaphylaxis_colab_{ts}.xlsx"
    out_model_csv = output_dir / f"gnn_dataset_model_ready_colab_{ts}.csv"
    out_all_csv = output_dir / f"gnn_dataset_all_labels_colab_{ts}.csv"

    label.to_csv(out_all_csv, index=False, encoding="utf-8-sig")
    model_ready.to_csv(out_model_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] Writing Excel: {out_xlsx}")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        add_sheet(writer, model_ready, "01_model_ready")
        add_sheet(writer, label, "02_all_labels")
        add_sheet(writer, label[label["faers_signal_class"] == "positive_high"], "03_positive_high")
        add_sheet(writer, label[label["faers_signal_class"] == "positive_moderate"], "04_positive_moderate")
        add_sheet(writer, label[label["faers_signal_class"].str.startswith("negative")], "05_negative_candidates")
        add_sheet(writer, label_counts, "06_label_counts")
        add_sheet(writer, model_counts, "07_model_counts")
        add_sheet(writer, scope_counts, "08_scope_counts")
        add_sheet(writer, qa, "09_qa")
        add_sheet(writer, structure_info, "10_structure_info")

    print("[INFO] Done.")
    print(qa.to_string(index=False))
    print("[INFO] Output files:")
    print(f"  {out_xlsx}")
    print(f"  {out_model_csv}")
    print(f"  {out_all_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
