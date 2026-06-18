#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
08_make_manuscript_figures_v2_colab.py

目的:
    既存のoutputファイルから、判読性を改善した論文用Figureを作成する。
    v1で問題になりやすい以下を修正する。
      - Figure 1のボックス重なり・文字切れ
      - Figure 3のエラーバーが上限を超えて見にくい問題
      - Figure 4の全薬剤名表示による過密
      - 棒グラフ中心から、点推定＋95%区間風のpoint-range表示へ変更

出力:
    output_v2/figure1_dataset_workflow_v2.png/pdf/svg
    output_v2/figure2_model_performance_roc_auc_v2.png/pdf/svg
    output_v2/figure2_model_performance_pr_auc_v2.png/pdf/svg
    output_v2/figure3_sensitivity_roc_auc_v2.png/pdf/svg
    output_v2/figure3_sensitivity_pr_auc_v2.png/pdf/svg
    output_v2/figure4_probability_distribution_v2.png/pdf/svg
    output_v2/figure4_label_review_candidates_v2.png/pdf/svg
    output_v2/manuscript_figure_captions_v2.txt

実行例:
    python 08_make_manuscript_figures_v2_colab.py \
      --input-dir output \
      --output-dir output_v2 \
      --dpi 600 \
      --font-size 11
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================

def latest_file(input_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(input_dir.glob(pattern))
    return files[-1] if files else None


def read_csv_or_empty(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_excel_sheet_or_empty(path: Optional[Path], sheet: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return pd.DataFrame()


def clean_label(x) -> str:
    mp = {
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
        "balanced_accuracy": "Balanced accuracy",
        "positive_high": "Positive high",
        "positive_moderate": "Positive moderate",
        "negative_high": "Negative high",
        "negative_moderate": "Negative moderate",
        "negative_expanded": "Negative expanded",
    }
    return mp.get(str(x), str(x))


def wrap_text(s: str, width: int = 20) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


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


def save_fig(fig, out_base: Path, dpi: int = 600) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def detect_files(input_dir: Path) -> Dict[str, Optional[Path]]:
    return {
        "dataset_xlsx": latest_file(input_dir, "gnn_dataset_anaphylaxis_colab_*.xlsx"),
        "model_ready_csv": latest_file(input_dir, "gnn_dataset_model_ready_colab_*.csv"),
        "ecfp_summary": latest_file(input_dir, "ecfp_baseline_model_summary_*.csv"),
        "scaffold_summary": latest_file(input_dir, "scaffold_split_ecfp_model_summary_*.csv"),
        "sensitivity_summary": latest_file(input_dir, "sensitivity_ecfp_rf_metric_summary_*.csv"),
        "sensitivity_results": latest_file(input_dir, "sensitivity_ecfp_rf_results_*.xlsx"),
        "drug_prob": latest_file(input_dir, "sensitivity_ecfp_rf_drug_probability_summary_*.csv"),
        "bit_importance": latest_file(input_dir, "sensitivity_ecfp_rf_bit_importance_*.csv"),
    }


# ============================================================
# Figure 1
# ============================================================

def figure1_workflow(out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(17.5, 9.5))
    ax.set_axis_off()

    # Wider spacing. No overlapping boxes.
    boxes = [
        ("FAERS database\nPostgreSQL tables:\ndemo / drug / reac", 0.06, 0.75),
        ("Case-level\ndeduplication\n400,514 reports", 0.30, 0.75),
        ("Endpoint definition\nStrict: 543 reports\nBroad: 1,981 reports", 0.54, 0.75),
        ("Drug exposure\nPS and PS/SS\n1,434,511 drug records", 0.78, 0.75),
        ("Drug-level ROR\nPositive and conservative\nnegative labels", 0.06, 0.45),
        ("Structure master\n202 candidates\nPubChem + manual curation", 0.30, 0.45),
        ("Scope filtering\n137 in scope\n65 excluded", 0.54, 0.45),
        ("Label–structure\nintegration\n7,918 drug-level rows", 0.78, 0.45),
        ("Model-ready dataset\n106 compounds\n59 positive / 47 negative", 0.30, 0.17),
        ("ECFP baseline\nRandom Forest\nRandom + scaffold split", 0.54, 0.17),
    ]

    box_w = 0.18
    box_h = 0.135

    for text, x, y in boxes:
        ax.text(
            x, y, text,
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=12,
            linespacing=1.20,
            bbox=dict(
                boxstyle="round,pad=0.45",
                linewidth=1.2,
                facecolor="white",
            ),
            clip_on=False,
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2), xycoords=ax.transAxes,
            xytext=(x1, y1), textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=1.4, shrinkA=4, shrinkB=4),
        )

    # Row 1
    arrow(0.16, 0.75, 0.22, 0.75)
    arrow(0.40, 0.75, 0.46, 0.75)
    arrow(0.64, 0.75, 0.70, 0.75)
    # Transition down
    arrow(0.78, 0.66, 0.06, 0.54)
    # Row 2
    arrow(0.16, 0.45, 0.22, 0.45)
    arrow(0.40, 0.45, 0.46, 0.45)
    arrow(0.64, 0.45, 0.70, 0.45)
    # Row 2 to row 3
    arrow(0.78, 0.36, 0.30, 0.26)
    arrow(0.40, 0.17, 0.46, 0.17)

    ax.set_title(
        "Workflow for constructing the FAERS-derived structure-linked model-ready dataset",
        pad=20,
        fontweight="bold",
    )
    save_fig(fig, out_dir / "figure1_dataset_workflow_v2", dpi=dpi)


# ============================================================
# Figure 2
# ============================================================

def build_model_perf_df(ecfp: pd.DataFrame, scaff: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []

    if not ecfp.empty:
        sub = ecfp[ecfp["metric"].eq(metric)].copy()
        for _, r in sub.iterrows():
            if str(r.get("model")) not in ["logistic_l2", "random_forest", "lightgbm"]:
                continue
            rows.append({
                "split_type": "random",
                "model": str(r.get("model")),
                "mean": float(r.get("mean")),
                "low": float(r.get("q025")),
                "high": float(r.get("q975")),
            })

    if not scaff.empty:
        sub = scaff[
            (scaff["metric"].eq(metric)) &
            (scaff["threshold_type"].eq("fixed_0.5"))
        ].copy()
        for _, r in sub.iterrows():
            if str(r.get("model")) not in ["logistic_l2", "random_forest", "lightgbm"]:
                continue
            rows.append({
                "split_type": "scaffold",
                "model": str(r.get("model")),
                "mean": float(r.get("mean")),
                "low": float(r.get("q025")),
                "high": float(r.get("q975")),
            })

    return pd.DataFrame(rows)


def point_range_by_model(df: pd.DataFrame, metric_label: str, out_base: Path, dpi: int) -> None:
    if df.empty:
        return

    model_order = ["logistic_l2", "random_forest", "lightgbm"]
    model_labels = [clean_label(m) for m in model_order]
    split_order = ["random", "scaffold"]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    x = np.arange(len(model_order))
    offsets = {"random": -0.10, "scaffold": 0.10}
    markers = {"random": "o", "scaffold": "s"}
    linestyles = {"random": "-", "scaffold": "--"}

    for split in split_order:
        means, lows, highs = [], [], []
        for m in model_order:
            sub = df[(df["model"].eq(m)) & (df["split_type"].eq(split))]
            if len(sub):
                r = sub.iloc[0]
                means.append(r["mean"])
                lows.append(r["low"])
                highs.append(r["high"])
            else:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)

        xpos = x + offsets[split]
        yerr = np.array([
            np.array(means) - np.array(lows),
            np.array(highs) - np.array(means),
        ])
        ax.errorbar(
            xpos,
            means,
            yerr=yerr,
            fmt=markers[split],
            linestyle=linestyles[split],
            linewidth=1.3,
            capsize=4,
            markersize=7,
            label=clean_label(split),
        )

        for xp, val in zip(xpos, means):
            if np.isfinite(val):
                ax.text(xp, min(val + 0.035, 1.02), f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    ax.axhline(0.5, linestyle=":", linewidth=1.2)
    ax.set_ylim(0.30, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Model")
    ax.set_title(f"{metric_label} across repeated random and scaffold splits")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    save_fig(fig, out_base, dpi=dpi)


def figure2_model_performance(ecfp: pd.DataFrame, scaff: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    point_range_by_model(
        build_model_perf_df(ecfp, scaff, "roc_auc"),
        "ROC-AUC",
        out_dir / "figure2_model_performance_roc_auc_v2",
        dpi,
    )
    point_range_by_model(
        build_model_perf_df(ecfp, scaff, "pr_auc"),
        "PR-AUC",
        out_dir / "figure2_model_performance_pr_auc_v2",
        dpi,
    )
    point_range_by_model(
        build_model_perf_df(ecfp, scaff, "balanced_accuracy"),
        "Balanced accuracy",
        out_dir / "figure2_model_performance_balanced_accuracy_v2",
        dpi,
    )


# ============================================================
# Figure 3
# ============================================================

def figure3_sensitivity(sens: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if sens.empty:
        print("[WARN] sensitivity summary missing; Figure 3 skipped.")
        return

    scenario_order = [
        "main_all_labels",
        "no_negative_expanded",
        "positive_high_vs_all_negative",
        "strict_high_confidence",
        "no_low_weight_labels",
    ]

    def make(metric: str, ylabel: str, stem: str):
        df = sens[
            (sens["metric"].eq(metric)) &
            (sens["threshold_type"].eq("fixed_0.5"))
        ].copy()
        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        y = np.arange(len(scenario_order))
        offsets = {"random": 0.12, "scaffold": -0.12}
        markers = {"random": "o", "scaffold": "s"}

        for split in ["random", "scaffold"]:
            means, lows, highs = [], [], []
            for scen in scenario_order:
                sub = df[(df["scenario"].eq(scen)) & (df["split_type"].eq(split))]
                if len(sub):
                    r = sub.iloc[0]
                    means.append(float(r["mean"]))
                    lows.append(float(r["q025"]))
                    highs.append(float(r["q975"]))
                else:
                    means.append(np.nan)
                    lows.append(np.nan)
                    highs.append(np.nan)

            ypos = y + offsets[split]
            xerr = np.array([
                np.array(means) - np.array(lows),
                np.array(highs) - np.array(means),
            ])
            ax.errorbar(
                means,
                ypos,
                xerr=xerr,
                fmt=markers[split],
                linestyle="None",
                capsize=4,
                markersize=7,
                label=clean_label(split),
            )
            for xp, yp in zip(means, ypos):
                if np.isfinite(xp):
                    ax.text(min(xp + 0.025, 1.02), yp, f"{xp:.2f}", ha="left", va="center", fontsize=10)

        ax.axvline(0.5, linestyle=":", linewidth=1.2)
        ax.set_xlim(0.30, 1.05)
        ax.set_yticks(y)
        ax.set_yticklabels([clean_label(s) for s in scenario_order])
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_ylabel("Sensitivity-analysis scenario")
        ax.set_title(f"Random Forest sensitivity analysis: {ylabel}")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="x", linewidth=0.5, alpha=0.4)
        save_fig(fig, out_dir / stem, dpi=dpi)

    make("roc_auc", "ROC-AUC", "figure3_sensitivity_roc_auc_v2")
    make("pr_auc", "PR-AUC", "figure3_sensitivity_pr_auc_v2")
    make("balanced_accuracy", "Balanced accuracy", "figure3_sensitivity_balanced_accuracy_v2")


# ============================================================
# Figure 4
# ============================================================

def figure4_probabilities(prob: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if prob.empty:
        print("[WARN] probability summary missing; Figure 4 skipped.")
        return

    df = prob[
        (prob["scenario"].eq("main_all_labels")) &
        (prob["split_type"].eq("scaffold"))
    ].copy()

    if df.empty:
        df = prob[prob["scenario"].eq("main_all_labels")].copy()

    if df.empty:
        print("[WARN] no main_all_labels probability data; Figure 4 skipped.")
        return

    df["label_name"] = df["model_label"].map({1: "Positive label", 0: "Negative label", 1.0: "Positive label", 0.0: "Negative label"})

    # Figure 4A: distribution without drug names.
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    groups = ["Negative label", "Positive label"]
    data = [df[df["label_name"].eq(g)]["mean_prob_positive"].dropna().values for g in groups]

    ax.boxplot(
        data,
        labels=groups,
        showmeans=True,
        patch_artist=False,
        widths=0.45,
    )

    # Add jittered points, deterministic.
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=28, alpha=0.75, edgecolor="black", linewidth=0.4)

    ax.axhline(0.5, linestyle=":", linewidth=1.2)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean predicted probability for positive label")
    ax.set_xlabel("Observed weak label")
    ax.set_title("Distribution of drug-level predicted probabilities\nRandom Forest, scaffold split, main analysis")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    save_fig(fig, out_dir / "figure4_probability_distribution_v2", dpi=dpi)

    # Figure 4B: label review candidates, selected only.
    fp = (
        df[(df["model_label"].eq(0)) & (df["mean_prob_positive"] >= 0.60)]
        .sort_values("mean_prob_positive", ascending=False)
        .head(10)
    )
    fn = (
        df[(df["model_label"].eq(1)) & (df["mean_prob_positive"] <= 0.45)]
        .sort_values("mean_prob_positive", ascending=True)
        .head(10)
    )
    focus = pd.concat([fp, fn], axis=0, ignore_index=True)
    if focus.empty:
        return

    focus = focus.sort_values("mean_prob_positive", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.45 * len(focus))))
    y = np.arange(len(focus))
    ax.barh(y, focus["mean_prob_positive"], edgecolor="black", linewidth=0.8)
    ax.axvline(0.5, linestyle=":", linewidth=1.2)
    labels = [
        f"{drug} ({clean_label(cls)})"
        for drug, cls in zip(focus["analysis_drug_name"], focus["faers_signal_class"])
    ]
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_text(x, width=34) for x in labels], fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability for positive label")
    ax.set_ylabel("Drug")
    ax.set_title("Label-review candidates based on predicted probabilities")
    ax.grid(axis="x", linewidth=0.5, alpha=0.4)
    save_fig(fig, out_dir / "figure4_label_review_candidates_v2", dpi=dpi)


# ============================================================
# Captions
# ============================================================

def write_captions(out_dir: Path) -> None:
    captions = """Figure 1. Workflow for constructing the FAERS-derived structure-linked model-ready dataset.
FAERS reports were deduplicated at the case level, strict and broad anaphylaxis endpoints were defined using MedDRA preferred terms, and drug exposure records were extracted using FAERS role codes. Drug-level reporting signals were integrated with a curated molecular structure master table to generate a final model-ready dataset of 106 compounds.

Figure 2. Model performance across repeated random and scaffold splits.
Point estimates show mean performance across repeated cross-validation, and horizontal or vertical error bars indicate the empirical 2.5th to 97.5th percentile interval. Random Forest showed the highest ROC-AUC and PR-AUC and retained performance under scaffold-based splitting.

Figure 3. Sensitivity analysis of the ECFP-based Random Forest model.
Sensitivity analyses assessed dependence on expanded negative labels and lower-confidence labels. Performance was evaluated using both repeated random splitting and Bemis–Murcko scaffold-based splitting.

Figure 4. Drug-level predicted probabilities in the main scaffold-split analysis.
Panel A summarizes the distribution of mean predicted probabilities by weak label class. Panel B highlights drugs whose predicted probabilities were discordant with their weak labels and may warrant label review or biological interpretation.
"""
    (out_dir / "manuscript_figure_captions_v2.txt").write_text(captions, encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="output")
    parser.add_argument("--output-dir", default="output_v2")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font-size", type=int, default=11)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_style(args.font_size)

    paths = detect_files(input_dir)
    print("===== Detected files =====")
    for k, v in paths.items():
        print(f"{k}: {v}")

    ecfp = read_csv_or_empty(paths["ecfp_summary"])
    scaff = read_csv_or_empty(paths["scaffold_summary"])
    sens = read_csv_or_empty(paths["sensitivity_summary"])
    prob = read_csv_or_empty(paths["drug_prob"])

    figure1_workflow(out_dir, dpi=args.dpi)
    figure2_model_performance(ecfp, scaff, out_dir, dpi=args.dpi)
    figure3_sensitivity(sens, out_dir, dpi=args.dpi)
    figure4_probabilities(prob, out_dir, dpi=args.dpi)
    write_captions(out_dir)

    manifest = pd.DataFrame([{"key": k, "path": str(v) if v is not None else ""} for k, v in paths.items()])
    manifest.to_csv(out_dir / "figure_generation_manifest_v2.csv", index=False, encoding="utf-8-sig")

    print("===== Created files =====")
    for p in sorted(out_dir.glob("*")):
        print(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
