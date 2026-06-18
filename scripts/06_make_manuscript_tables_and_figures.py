#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
08_make_manuscript_tables_figures_colab.py

目的:
    これまでのFAERS × structure master × ECFP baseline解析結果から、
    学術論文用の表と高解像度Figureを自動生成する。

入力:
    output/ 以下の最新ファイルを自動検出する。
      - gnn_dataset_anaphylaxis_colab_*.xlsx
      - gnn_dataset_model_ready_colab_*.csv
      - ecfp_baseline_model_summary_*.csv
      - scaffold_split_ecfp_model_summary_*.csv
      - scaffold_split_scaffold_summary_*.csv
      - sensitivity_ecfp_rf_metric_summary_*.csv
      - sensitivity_ecfp_rf_drug_probability_summary_*.csv
      - sensitivity_ecfp_rf_bit_importance_*.csv
      - sensitivity_ecfp_rf_results_*.xlsx

出力:
    output/manuscript_tables_YYYYMMDD_HHMMSS.xlsx
    output/supplementary_tables_YYYYMMDD_HHMMSS.xlsx

    output/figure1_dataset_workflow.png/pdf/svg
    output/figure2_model_performance.png/pdf/svg
    output/figure3_sensitivity_analysis.png/pdf/svg
    output/figure4_drug_probability_distribution.png/pdf/svg

    output/supplementary_figure_s1_permutation_summary.png/pdf/svg
    output/supplementary_figure_s2_scaffold_distribution.png/pdf/svg
    output/supplementary_figure_s3_top_ecfp_bits.png/pdf/svg

特徴:
    - 600 dpi PNG
    - PDF/SVGベクター出力
    - 大きめフォント
    - 余白を広く確保
    - 色だけに依存しない表示（marker, hatch, label併用）
    - 論文用テーブルをxlsx出力
    - 入力ファイルが一部欠けても作成可能な範囲で続行

実行例:
    python 08_make_manuscript_tables_figures_colab.py \
      --output-dir output \
      --dpi 600

注意:
    図は原稿作成用の第一版であり、投稿先のFigure規定に合わせて最終調整する。
"""

from __future__ import annotations

import argparse
import math
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ============================================================
# 0. Global figure settings
# ============================================================

def set_manuscript_style(font_size: int = 12) -> None:
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


def save_figure(fig, base_path: Path, dpi: int = 600) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        out = base_path.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def latest_file(output_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(output_dir.glob(pattern))
    return files[-1] if files else None


def safe_read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_read_excel(path: Optional[Path], sheet_name: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def clean_label(x: str) -> str:
    mapping = {
        "main_all_labels": "Main analysis",
        "no_negative_expanded": "No expanded\nnegatives",
        "positive_high_vs_all_negative": "High-confidence\npositives only",
        "strict_high_confidence": "High-confidence\nlabels only",
        "no_low_weight_labels": "Weight ≥ 0.8\nlabels only",
        "random": "Random split",
        "scaffold": "Scaffold split",
        "logistic_l2": "Logistic\nregression",
        "logistic_l1": "Logistic L1",
        "random_forest": "Random\nForest",
        "lightgbm": "LightGBM",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "balanced_accuracy": "Balanced\naccuracy",
        "fixed_0.5": "Fixed 0.5",
        "train_youden": "Train-fold\nYouden",
        "positive_high": "Positive high",
        "positive_moderate": "Positive moderate",
        "negative_high": "Negative high",
        "negative_moderate": "Negative moderate",
        "negative_expanded": "Negative expanded",
    }
    return mapping.get(str(x), str(x))


def wrap_names(names: List[str], width: int = 18) -> List[str]:
    return ["\n".join(textwrap.wrap(str(n), width=width, break_long_words=False)) for n in names]


def add_value_labels(ax, xs, ys, fmt="{:.2f}", y_offset=0.015, rotation=0):
    for x, y in zip(xs, ys):
        if pd.isna(y):
            continue
        ax.text(x, y + y_offset, fmt.format(y), ha="center", va="bottom", rotation=rotation, fontsize=max(8, plt.rcParams["font.size"] - 2))


# ============================================================
# 1. Data collection
# ============================================================

def detect_inputs(output_dir: Path) -> Dict[str, Optional[Path]]:
    return {
        "dataset_xlsx": latest_file(output_dir, "gnn_dataset_anaphylaxis_colab_*.xlsx"),
        "model_ready_csv": latest_file(output_dir, "gnn_dataset_model_ready_colab_*.csv"),
        "all_labels_csv": latest_file(output_dir, "gnn_dataset_all_labels_colab_*.csv"),
        "ecfp_summary": latest_file(output_dir, "ecfp_baseline_model_summary_*.csv"),
        "ecfp_results_xlsx": latest_file(output_dir, "ecfp_baseline_results_*.xlsx"),
        "scaffold_summary": latest_file(output_dir, "scaffold_split_ecfp_model_summary_*.csv"),
        "scaffold_results_xlsx": latest_file(output_dir, "scaffold_split_ecfp_results_*.xlsx"),
        "scaffold_groups": latest_file(output_dir, "scaffold_split_scaffold_summary_*.csv"),
        "sensitivity_summary": latest_file(output_dir, "sensitivity_ecfp_rf_metric_summary_*.csv"),
        "sensitivity_results_xlsx": latest_file(output_dir, "sensitivity_ecfp_rf_results_*.xlsx"),
        "sensitivity_drug_prob": latest_file(output_dir, "sensitivity_ecfp_rf_drug_probability_summary_*.csv"),
        "sensitivity_bits": latest_file(output_dir, "sensitivity_ecfp_rf_bit_importance_*.csv"),
    }


def build_key_tables(paths: Dict[str, Optional[Path]]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    model_ready = safe_read_csv(paths["model_ready_csv"])
    all_labels = safe_read_csv(paths["all_labels_csv"])
    ecfp_summary = safe_read_csv(paths["ecfp_summary"])
    scaffold_summary = safe_read_csv(paths["scaffold_summary"])
    scaffold_groups = safe_read_csv(paths["scaffold_groups"])
    sensitivity_summary = safe_read_csv(paths["sensitivity_summary"])
    drug_prob = safe_read_csv(paths["sensitivity_drug_prob"])
    bits = safe_read_csv(paths["sensitivity_bits"])

    dataset_qa = safe_read_excel(paths["dataset_xlsx"], "09_qa")
    label_counts = safe_read_excel(paths["dataset_xlsx"], "06_label_counts")
    model_counts = safe_read_excel(paths["dataset_xlsx"], "07_model_counts")
    scope_counts = safe_read_excel(paths["dataset_xlsx"], "08_scope_counts")
    structure_info = safe_read_excel(paths["dataset_xlsx"], "10_structure_info")

    tables["model_ready"] = model_ready
    tables["all_labels"] = all_labels
    tables["ecfp_summary"] = ecfp_summary
    tables["scaffold_summary"] = scaffold_summary
    tables["scaffold_groups"] = scaffold_groups
    tables["sensitivity_summary"] = sensitivity_summary
    tables["drug_prob"] = drug_prob
    tables["bits"] = bits
    tables["dataset_qa"] = dataset_qa
    tables["label_counts"] = label_counts
    tables["model_counts"] = model_counts
    tables["scope_counts"] = scope_counts
    tables["structure_info"] = structure_info

    # Manuscript-ready dataset summary
    if not dataset_qa.empty:
        qa_map = dict(zip(dataset_qa["item"].astype(str), dataset_qa["value"]))
        dataset_summary = pd.DataFrame([
            ["Deduplicated FAERS reports", qa_map.get("n_latest_cases", np.nan)],
            ["Strict anaphylactic shock reports", qa_map.get("n_strict_shock_cases", np.nan)],
            ["Broad anaphylaxis reports", qa_map.get("n_broad_anaphylaxis_cases", np.nan)],
            ["All drug-level label rows", qa_map.get("n_all_label_rows", np.nan)],
            ["Model-ready compounds", qa_map.get("n_model_ready_rows", np.nan)],
            ["Model-ready positives", qa_map.get("n_model_positive", np.nan)],
            ["Model-ready negatives", qa_map.get("n_model_negative", np.nan)],
            ["Model-ready records with missing SMILES", qa_map.get("n_model_missing_smiles", np.nan)],
            ["Model-ready records outside scope", qa_map.get("n_model_not_in_scope", np.nan)],
        ], columns=["Item", "Count"])
    else:
        dataset_summary = pd.DataFrame()

    tables["table_dataset_summary"] = dataset_summary

    # Final model-ready composition
    if not model_ready.empty:
        comp = (
            model_ready.groupby(["model_label", "faers_signal_class", "label_weight"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["model_label", "faers_signal_class"], ascending=[False, True])
        )
        comp["model_label_name"] = comp["model_label"].map({1: "Positive", 0: "Negative", 1.0: "Positive", 0.0: "Negative"})
        comp["faers_signal_class_label"] = comp["faers_signal_class"].map(clean_label)
    else:
        comp = pd.DataFrame()
    tables["table_model_ready_composition"] = comp

    # Model performance table
    perf_rows = []
    if not ecfp_summary.empty:
        df = ecfp_summary.copy()
        # Random split summary from 05 has no threshold_type; ROC/PR are threshold-independent.
        for metric in ["roc_auc", "pr_auc", "balanced_accuracy"]:
            sub = df[df["metric"].eq(metric)].copy()
            for _, r in sub.iterrows():
                perf_rows.append({
                    "analysis": "Repeated random split",
                    "model": clean_label(r.get("model")),
                    "threshold": "Mixed / not applicable",
                    "metric": clean_label(metric),
                    "mean": r.get("mean"),
                    "sd": r.get("sd"),
                    "q025": r.get("q025"),
                    "q975": r.get("q975"),
                })
    if not scaffold_summary.empty:
        df = scaffold_summary.copy()
        for metric in ["roc_auc", "pr_auc", "balanced_accuracy"]:
            sub = df[(df["metric"].eq(metric)) & (df["threshold_type"].eq("fixed_0.5"))].copy()
            for _, r in sub.iterrows():
                perf_rows.append({
                    "analysis": "Scaffold split",
                    "model": clean_label(r.get("model")),
                    "threshold": clean_label(r.get("threshold_type")),
                    "metric": clean_label(metric),
                    "mean": r.get("mean"),
                    "sd": r.get("sd"),
                    "q025": r.get("q025"),
                    "q975": r.get("q975"),
                })
    perf_table = pd.DataFrame(perf_rows)
    tables["table_model_performance"] = perf_table

    # Sensitivity table
    if not sensitivity_summary.empty:
        sens = sensitivity_summary.copy()
        sens = sens[
            sens["metric"].isin(["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity"])
        ].copy()
        sens["scenario_label"] = sens["scenario"].map(clean_label)
        sens["split_label"] = sens["split_type"].map(clean_label)
        sens["metric_label"] = sens["metric"].map(clean_label)
        sens["threshold_label"] = sens["threshold_type"].map(clean_label)
    else:
        sens = pd.DataFrame()
    tables["table_sensitivity"] = sens

    return tables


# ============================================================
# 2. Figure generation
# ============================================================

def make_figure1_workflow(output_dir: Path, dpi: int = 600) -> None:
    """
    Dataset construction workflow.
    Text boxes with arrows. High readability.
    """
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.axis("off")

    boxes = [
        ("FAERS database\nPostgreSQL tables:\ndemo / drug / reac", 0.04, 0.62),
        ("Case-level\ndeduplication\n400,514 reports", 0.25, 0.62),
        ("Endpoint definition\nStrict: 543 reports\nBroad: 1,981 reports", 0.46, 0.62),
        ("Drug exposure\nPS and PS/SS\n1,434,511 drug records", 0.67, 0.62),
        ("Drug-level ROR\nPositive and conservative\nnegative signal labels", 0.04, 0.25),
        ("Structure master\n202 candidates\nPubChem + manual curation", 0.25, 0.25),
        ("Scope filtering\n137 in scope\n65 excluded", 0.46, 0.25),
        ("Model-ready dataset\n106 compounds\n59 positive / 47 negative", 0.67, 0.25),
    ]

    for text, x, y in boxes:
        ax.text(
            x, y, text,
            transform=ax.transAxes,
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.5", linewidth=1.2, facecolor="white"),
            fontsize=13,
            linespacing=1.25,
        )

    arrow_pairs = [
        ((0.20, 0.62), (0.245, 0.62)),
        ((0.41, 0.62), (0.455, 0.62)),
        ((0.62, 0.62), (0.665, 0.62)),
        ((0.75, 0.53), (0.12, 0.36)),
        ((0.20, 0.25), (0.245, 0.25)),
        ((0.41, 0.25), (0.455, 0.25)),
        ((0.62, 0.25), (0.665, 0.25)),
    ]

    for start, end in arrow_pairs:
        ax.annotate(
            "",
            xy=end, xycoords=ax.transAxes,
            xytext=start, textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=1.4),
        )

    ax.text(
        0.04, 0.93,
        "Figure 1. Workflow for constructing the FAERS-derived structure-linked model-ready dataset",
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=15,
        fontweight="bold",
    )

    save_figure(fig, output_dir / "figure1_dataset_workflow", dpi=dpi)


def make_figure2_model_performance(tables: Dict[str, pd.DataFrame], output_dir: Path, dpi: int = 600) -> None:
    """
    Compare model performance for random split and scaffold split.
    One panel per metric output as separate high-res files plus combined style not using subplots.
    Main output figure2_model_performance uses ROC-AUC.
    Additional Figure2_PR and Figure2_BA are also saved.
    """
    ecfp = tables["ecfp_summary"]
    scaff = tables["scaffold_summary"]

    if ecfp.empty or scaff.empty:
        print("[WARN] Cannot create Figure 2: missing ECFP or scaffold summary.")
        return

    def make_metric(metric: str, file_stem: str, ylabel: str):
        rows = []
        # random
        sub = ecfp[ecfp["metric"].eq(metric)].copy()
        for _, r in sub.iterrows():
            rows.append({
                "analysis": "Random split",
                "model": clean_label(r["model"]),
                "mean": r["mean"],
                "sd": r["sd"],
            })

        # scaffold fixed
        sub = scaff[(scaff["metric"].eq(metric)) & (scaff["threshold_type"].eq("fixed_0.5"))].copy()
        for _, r in sub.iterrows():
            rows.append({
                "analysis": "Scaffold split",
                "model": clean_label(r["model"]),
                "mean": r["mean"],
                "sd": r["sd"],
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return

        model_order = ["Logistic\nregression", "Random\nForest", "LightGBM"]
        df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
        df = df.dropna(subset=["model"]).sort_values(["model", "analysis"])

        analyses = ["Random split", "Scaffold split"]
        x = np.arange(len(model_order))
        width = 0.34

        fig, ax = plt.subplots(figsize=(9.5, 6.2))

        for i, analysis in enumerate(analyses):
            vals = []
            errs = []
            for model in model_order:
                s = df[(df["model"].eq(model)) & (df["analysis"].eq(analysis))]
                vals.append(float(s["mean"].iloc[0]) if len(s) else np.nan)
                errs.append(float(s["sd"].iloc[0]) if len(s) else 0.0)
            xpos = x + (i - 0.5) * width
            bars = ax.bar(
                xpos, vals, width,
                yerr=errs,
                capsize=4,
                label=analysis,
                edgecolor="black",
                linewidth=0.8,
                hatch="" if i == 0 else "///",
            )
            add_value_labels(ax, xpos, vals, fmt="{:.2f}", y_offset=0.015)

        ax.axhline(0.5, linestyle="--", linewidth=1.1)
        ax.set_ylim(0.35, 1.02)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(model_order)
        ax.set_title(f"{ylabel} across repeated random and scaffold splits")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="y", linewidth=0.5, alpha=0.4)
        save_figure(fig, output_dir / file_stem, dpi=dpi)

    make_metric("roc_auc", "figure2_model_performance", "ROC-AUC")
    make_metric("pr_auc", "figure2b_model_performance_pr_auc", "PR-AUC")
    make_metric("balanced_accuracy", "figure2c_model_performance_balanced_accuracy", "Balanced accuracy")


def make_figure3_sensitivity(tables: Dict[str, pd.DataFrame], output_dir: Path, dpi: int = 600) -> None:
    """
    Sensitivity analysis for RF: main figure uses ROC-AUC.
    """
    sens = tables["sensitivity_summary"]
    if sens.empty:
        print("[WARN] Cannot create Figure 3: missing sensitivity summary.")
        return

    scenario_order = [
        "main_all_labels",
        "no_negative_expanded",
        "positive_high_vs_all_negative",
        "strict_high_confidence",
        "no_low_weight_labels",
    ]

    def make_metric(metric: str, stem: str, ylabel: str):
        df = sens[
            (sens["metric"].eq(metric)) &
            (sens["threshold_type"].eq("fixed_0.5"))
        ].copy()

        if df.empty:
            return

        df["scenario"] = pd.Categorical(df["scenario"], categories=scenario_order, ordered=True)
        df = df.dropna(subset=["scenario"]).sort_values(["scenario", "split_type"])

        scenarios = scenario_order
        split_order = ["random", "scaffold"]

        x = np.arange(len(scenarios))
        width = 0.34

        fig, ax = plt.subplots(figsize=(12.5, 6.5))

        for i, split_type in enumerate(split_order):
            vals = []
            errs = []
            for scen in scenarios:
                s = df[(df["scenario"].eq(scen)) & (df["split_type"].eq(split_type))]
                vals.append(float(s["mean"].iloc[0]) if len(s) else np.nan)
                errs.append(float(s["sd"].iloc[0]) if len(s) else 0.0)

            xpos = x + (i - 0.5) * width
            ax.bar(
                xpos, vals, width,
                yerr=errs,
                capsize=4,
                label=clean_label(split_type),
                edgecolor="black",
                linewidth=0.8,
                hatch="" if i == 0 else "///",
            )
            add_value_labels(ax, xpos, vals, fmt="{:.2f}", y_offset=0.015)

        ax.axhline(0.5, linestyle="--", linewidth=1.1)
        ax.set_ylim(0.30, 1.03)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Sensitivity-analysis scenario")
        ax.set_xticks(x)
        ax.set_xticklabels([clean_label(s) for s in scenarios], rotation=0)
        ax.set_title(f"Random Forest sensitivity analysis: {ylabel}")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="y", linewidth=0.5, alpha=0.4)

        save_figure(fig, output_dir / stem, dpi=dpi)

    make_metric("roc_auc", "figure3_sensitivity_analysis", "ROC-AUC")
    make_metric("pr_auc", "figure3b_sensitivity_analysis_pr_auc", "PR-AUC")
    make_metric("balanced_accuracy", "figure3c_sensitivity_analysis_balanced_accuracy", "Balanced accuracy")


def make_figure4_drug_probability(tables: Dict[str, pd.DataFrame], output_dir: Path, dpi: int = 600) -> None:
    prob = tables["drug_prob"]
    if prob.empty:
        print("[WARN] Cannot create Figure 4: missing drug probability summary.")
        return

    df = prob[
        (prob["scenario"].eq("main_all_labels")) &
        (prob["split_type"].eq("scaffold"))
    ].copy()

    if df.empty:
        df = prob[prob["scenario"].eq("main_all_labels")].copy()

    if df.empty:
        print("[WARN] Cannot create Figure 4: no main_all_labels probabilities.")
        return

    df = df.sort_values("mean_prob_positive", ascending=True).reset_index(drop=True)
    df["drug_label"] = df["analysis_drug_name"].astype(str)

    fig_height = max(10, 0.16 * len(df))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    y_positions = np.arange(len(df))
    # Use marker shapes; matplotlib default colors are used.
    pos = df["model_label"].astype(int).eq(1)
    neg = df["model_label"].astype(int).eq(0)

    ax.scatter(
        df.loc[neg, "mean_prob_positive"],
        y_positions[neg],
        marker="o",
        s=40,
        label="Negative label",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.scatter(
        df.loc[pos, "mean_prob_positive"],
        y_positions[pos],
        marker="^",
        s=48,
        label="Positive label",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axvline(0.5, linestyle="--", linewidth=1.1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability for positive label")
    ax.set_ylabel("Drug")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(wrap_names(df["drug_label"].tolist(), width=24), fontsize=8)
    ax.set_title("Drug-level predicted probabilities in scaffold split\nRandom Forest, main analysis")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", linewidth=0.5, alpha=0.4)

    save_figure(fig, output_dir / "figure4_drug_probability_distribution", dpi=dpi)

    # Focused figure: top potential false positives/negatives.
    fp = df[(df["model_label"].eq(0)) & (df["mean_prob_positive"] >= 0.6)].sort_values("mean_prob_positive", ascending=False).head(12)
    fn = df[(df["model_label"].eq(1)) & (df["mean_prob_positive"] <= 0.4)].sort_values("mean_prob_positive", ascending=True).head(12)
    focus = pd.concat([fp, fn], ignore_index=True)
    if not focus.empty:
        focus = focus.sort_values("mean_prob_positive", ascending=True).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.45 * len(focus))))
        y = np.arange(len(focus))
        ax.barh(y, focus["mean_prob_positive"], edgecolor="black", linewidth=0.8)
        ax.axvline(0.5, linestyle="--", linewidth=1.1)
        ax.set_xlim(0, 1)
        ax.set_yticks(y)
        labels = [
            f"{drug}\n({clean_label(cls)})"
            for drug, cls in zip(focus["analysis_drug_name"], focus["faers_signal_class"])
        ]
        ax.set_yticklabels(wrap_names(labels, width=32), fontsize=9)
        ax.set_xlabel("Mean predicted probability for positive label")
        ax.set_title("Potential label-review candidates")
        ax.grid(axis="x", linewidth=0.5, alpha=0.4)
        save_figure(fig, output_dir / "figure4b_label_review_candidates", dpi=dpi)


def make_supplementary_figures(tables: Dict[str, pd.DataFrame], output_dir: Path, dpi: int = 600) -> None:
    # S1: permutation summary
    sens_results = safe_read_excel(
        latest_file(output_dir, "sensitivity_ecfp_rf_results_*.xlsx"),
        "04_permutation_summary",
    )
    if not sens_results.empty:
        df = sens_results.copy()
        df = df[df["split_type"].isin(["random", "scaffold"])].copy()
        if not df.empty:
            df["scenario_label"] = df["scenario"].map(clean_label)
            df = df.sort_values(["scenario", "split_type"])

            scenarios = list(dict.fromkeys(df["scenario"].tolist()))
            x = np.arange(len(scenarios))
            width = 0.34

            fig, ax = plt.subplots(figsize=(12.5, 6.2))
            for i, split_type in enumerate(["random", "scaffold"]):
                vals = []
                qs = []
                for scen in scenarios:
                    s = df[(df["scenario"].eq(scen)) & (df["split_type"].eq(split_type))]
                    vals.append(float(s["observed_roc_auc_mean"].iloc[0]) if len(s) else np.nan)
                    qs.append(float(s["perm_roc_auc_q95"].iloc[0]) if len(s) else np.nan)

                xpos = x + (i - 0.5) * width
                ax.bar(
                    xpos, vals, width,
                    label=f"{clean_label(split_type)} observed",
                    edgecolor="black",
                    linewidth=0.8,
                    hatch="" if i == 0 else "///",
                )
                ax.scatter(xpos, qs, marker="x", s=80, label=f"{clean_label(split_type)} permuted 95th" if i == 0 else None)

            ax.axhline(0.5, linestyle="--", linewidth=1.1)
            ax.set_ylim(0.3, 1.03)
            ax.set_ylabel("ROC-AUC")
            ax.set_xticks(x)
            ax.set_xticklabels([clean_label(s) for s in scenarios])
            ax.set_title("Permutation test summary")
            ax.legend(frameon=False, loc="lower right", ncol=1)
            ax.grid(axis="y", linewidth=0.5, alpha=0.4)
            save_figure(fig, output_dir / "supplementary_figure_s1_permutation_summary", dpi=dpi)

    # S2: scaffold distribution
    scaff = tables["scaffold_groups"]
    if not scaff.empty:
        df = scaff.copy()
        top = df.sort_values("n", ascending=False).head(25).copy()
        top = top.sort_values("n", ascending=True)
        fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.36 * len(top))))
        y = np.arange(len(top))
        ax.barh(y, top["n"], edgecolor="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(wrap_names(top["scaffold_key"].tolist(), width=28), fontsize=8)
        ax.set_xlabel("Number of drugs")
        ax.set_ylabel("Bemis–Murcko scaffold group")
        ax.set_title("Largest scaffold groups in the model-ready dataset")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="x", linewidth=0.5, alpha=0.4)
        save_figure(fig, output_dir / "supplementary_figure_s2_scaffold_distribution", dpi=dpi)

    # S3: ECFP bit importance
    bits = tables["bits"]
    if not bits.empty:
        df = bits[bits["scenario"].eq("main_all_labels")].copy()
        if df.empty:
            df = bits.copy()
        df = df.sort_values("importance", ascending=False).head(25)
        df = df.sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9.5, 7.5))
        y = np.arange(len(df))
        labels = [f"Bit {int(b)}" for b in df["bit"]]
        ax.barh(y, df["importance"], edgecolor="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Random Forest feature importance")
        ax.set_ylabel("ECFP bit")
        ax.set_title("Top ECFP bits in the Random Forest model\nMain analysis")
        ax.grid(axis="x", linewidth=0.5, alpha=0.4)
        save_figure(fig, output_dir / "supplementary_figure_s3_top_ecfp_bits", dpi=dpi)


# ============================================================
# 3. Tables
# ============================================================

def write_tables(tables: Dict[str, pd.DataFrame], output_dir: Path, ts: str) -> Tuple[Path, Path]:
    manuscript_path = output_dir / f"manuscript_tables_{ts}.xlsx"
    supplement_path = output_dir / f"supplementary_tables_{ts}.xlsx"

    with pd.ExcelWriter(manuscript_path, engine="openpyxl") as writer:
        sheets = {
            "Table1_Dataset_summary": tables.get("table_dataset_summary", pd.DataFrame()),
            "Table2_Model_ready": tables.get("table_model_ready_composition", pd.DataFrame()),
            "Table3_Model_performance": tables.get("table_model_performance", pd.DataFrame()),
            "Table4_Sensitivity": tables.get("table_sensitivity", pd.DataFrame()),
        }
        for name, df in sheets.items():
            if df is None or df.empty:
                df = pd.DataFrame({"note": ["No data available"]})
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 70)

    with pd.ExcelWriter(supplement_path, engine="openpyxl") as writer:
        supp = {
            "S1_Model_ready_dataset": tables.get("model_ready", pd.DataFrame()),
            "S2_All_labels": tables.get("all_labels", pd.DataFrame()),
            "S3_Scaffold_groups": tables.get("scaffold_groups", pd.DataFrame()),
            "S4_Drug_probabilities": tables.get("drug_prob", pd.DataFrame()),
            "S5_ECFP_bit_importance": tables.get("bits", pd.DataFrame()),
            "S6_ECFP_summary": tables.get("ecfp_summary", pd.DataFrame()),
            "S7_Scaffold_summary": tables.get("scaffold_summary", pd.DataFrame()),
            "S8_Sensitivity_summary": tables.get("sensitivity_summary", pd.DataFrame()),
            "S9_Structure_info": tables.get("structure_info", pd.DataFrame()),
        }
        for name, df in supp.items():
            if df is None or df.empty:
                df = pd.DataFrame({"note": ["No data available"]})
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 70)

    return manuscript_path, supplement_path


def write_input_manifest(paths: Dict[str, Optional[Path]], output_dir: Path, ts: str) -> Path:
    manifest = pd.DataFrame([
        {"key": k, "path": str(v) if v is not None else ""}
        for k, v in paths.items()
    ])
    out = output_dir / f"manuscript_generation_manifest_{ts}.csv"
    manifest.to_csv(out, index=False, encoding="utf-8-sig")
    return out


# ============================================================
# 4. Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font-size", type=int, default=12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_manuscript_style(font_size=args.font_size)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = detect_inputs(output_dir)
    print("===== Detected input files =====")
    for k, v in paths.items():
        print(f"{k}: {v}")

    tables = build_key_tables(paths)

    manifest_path = write_input_manifest(paths, output_dir, ts)
    manuscript_path, supplement_path = write_tables(tables, output_dir, ts)

    print("[INFO] Creating figures...")
    make_figure1_workflow(output_dir, dpi=args.dpi)
    make_figure2_model_performance(tables, output_dir, dpi=args.dpi)
    make_figure3_sensitivity(tables, output_dir, dpi=args.dpi)
    make_figure4_drug_probability(tables, output_dir, dpi=args.dpi)
    make_supplementary_figures(tables, output_dir, dpi=args.dpi)

    print("[INFO] Done.")
    print("===== Output files =====")
    print(manuscript_path)
    print(supplement_path)
    print(manifest_path)
    for p in sorted(output_dir.glob("figure*.png")):
        print(p)
    for p in sorted(output_dir.glob("supplementary_figure*.png")):
        print(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
