#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_autofill_structure_master_pubchem_v2.py

目的:
    drug_structure_master_curated_v1_template.xlsx を読み込み、
    PubChem PUG-REST を用いて PubChem CID / canonical SMILES / InChIKey などを半自動補完する。

入力:
    既定:
        drug_structure_master_curated_v1_template.xlsx
    または:
        --input path/to/file.xlsx

出力:
    既定:
        output/drug_structure_master_curated_v1_autofill_YYYYMMDD_HHMMSS.xlsx
        output/structure_curation_summary_YYYYMMDD_HHMMSS.xlsx
        output/pubchem_cache.json

重要:
    本スクリプトは「一次補完」です。
    ブランド名、配合剤、抗体、ペプチド、デバイス、金属錯体などは人手確認が必要です。

推奨実行例:
    python 03_autofill_structure_master_pubchem.py ^
      --input drug_structure_master_curated_v1_template.xlsx ^
      --output-dir output

必要パッケージ:
    pip install pandas openpyxl requests

任意:
    RDKitが入っている場合のみ、SMILES妥当性チェックを行う。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests


# ============================================================
# 0. 設定
# ============================================================

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
DEFAULT_INPUT = "drug_structure_master_curated_v1_template.xlsx"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CACHE = "output/pubchem_cache.json"

# PubChem PUG-RESTの利用負荷を抑える。
SLEEP_SEC = 0.25
TIMEOUT_SEC = 30
MAX_RETRIES = 3

# 分子量が大きすぎる場合は、通常の低分子GNNでは慎重扱いにする。
MW_EXCLUDE_THRESHOLD = 2000.0
MW_REVIEW_THRESHOLD = 1000.0


# ============================================================
# 1. ブランド名・製剤名の暫定正規化辞書
# ============================================================
# 必ずしも網羅的ではない。
# PubChem照合のための「候補名」として使う。
# 最終判断はExcel上で確認する。

BRAND_TO_INGREDIENT: Dict[str, str] = {
    # 陰性候補で出現したブランド/製剤名
    "CABOMETYX": "CABOZANTINIB",
    "ORGOVYX": "RELUGOLIX",
    "NUPLAZID": "PIMAVANSERIN",
    "IBRANCE": "PALBOCICLIB",
    "POMALYST": "POMALIDOMIDE",
    "XELJANZ XR": "TOFACITINIB",
    "OTEZLA": "APREMILAST",
    "KISQALI": "RIBOCICLIB",
    "SPRAVATO": "ESKETAMINE",
    "REMODULIN": "TREPROSTINIL",
    "LUPKYNIS": "VOCLOSPORIN",
    "LUMRYZ": "SODIUM OXYBATE",
    "CLOZARIL": "CLOZAPINE",
    "ELIGARD": "LEUPROLIDE",
    "LANTUS SOLOSTAR": "INSULIN GLARGINE",
    "TRULICITY": "DULAGLUTIDE",
    "PARAGARD T 380A": "COPPER",
    "TRELEGY ELLIPTA": "FLUTICASONE FUROATE\\UMECLIDINIUM\\VILANTEROL",

    # 陽性候補で出現しやすいブランド
    "XOLAIR": "OMALIZUMAB",
    "TECENTRIQ": "ATEZOLIZUMAB",
    "ULTRAVIST": "IOPROMIDE",
    "EPIPEN": "EPINEPHRINE",
    "EPIPEN JR": "EPINEPHRINE",

    # 代表的なブランド候補
    "KEYTRUDA": "PEMBROLIZUMAB",
    "OPDIVO": "NIVOLUMAB",
    "HUMIRA": "ADALIMUMAB",
    "REMICADE": "INFLIXIMAB",
    "RITUXAN": "RITUXIMAB",
}


# ============================================================
# 2. 除外候補判定ルール
# ============================================================

BIOLOGIC_KEYWORDS = [
    "MAB",
    "OMALIZUMAB",
    "RITUXIMAB",
    "INFLIXIMAB",
    "ADALIMUMAB",
    "ATEZOLIZUMAB",
    "PEMBROLIZUMAB",
    "NIVOLUMAB",
    "DUPILUMAB",
    "USTEKINUMAB",
    "TOCILIZUMAB",
    "BEVACIZUMAB",
    "TRASTUZUMAB",
    "DARATUMUMAB",
]

PEPTIDE_OR_PROTEIN_KEYWORDS = [
    "INSULIN",
    "DULAGLUTIDE",
    "SEMAGLUTIDE",
    "LIRAGLUTIDE",
    "EXENATIDE",
    "LEUPROLIDE",
    "GOSERELIN",
    "OCTREOTIDE",
    "TERIPARATIDE",
]

DEVICE_KEYWORDS = [
    "PARAGARD",
    "T 380A",
    "IUD",
    "INTRAUTERINE",
]

VACCINE_KEYWORDS = [
    "VACCINE",
    "VAX",
    "COMIRNATY",
    "SPIKEVAX",
]

MIXTURE_KEYWORDS = [
    "DIETARY SUPPLEMENT",
    "HERBAL",
    "HERBALS",
    "POLYETHYLENE GLYCOLS",
    "MULTIVITAMIN",
]

INORGANIC_KEYWORDS = [
    "ZINC",
    "IRON",
    "MAGNESIUM",
    "CALCIUM",
    "COPPER",
    "POTASSIUM",
    "SODIUM CHLORIDE",
]


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().upper())


def looks_like_combination(name: str) -> bool:
    name = norm_text(name)
    if "\\" in name:
        return True
    if " / " in name or "/" in name:
        return True
    # 明らかなAND結合。ただし単語内のANDは避ける。
    if re.search(r"\bAND\b", name):
        return True
    return False


def initial_exclusion_reason(raw_name: str, ingredient: str) -> Tuple[Optional[bool], Optional[str], str]:
    """
    PubChem照合前の暫定 in_scope / exclude_reason / note を返す。
    戻り値:
        in_scope_initial:
            True/False/None
        exclude_reason:
            str or None
        note:
            str
    """
    raw = norm_text(raw_name)
    ing = norm_text(ingredient)
    target = f"{raw} {ing}"

    if looks_like_combination(raw) or looks_like_combination(ing):
        return False, "combination_product", "Backslash/slash/AND suggests a combination product; manual review required."

    if any(k in target for k in BIOLOGIC_KEYWORDS):
        return False, "biologic_mAb", "Biologic or monoclonal antibody-like name; exclude from small-molecule GNN."

    if any(k in target for k in PEPTIDE_OR_PROTEIN_KEYWORDS):
        return False, "peptide_biologic", "Peptide/protein-like drug; exclude or review separately."

    if any(k in target for k in DEVICE_KEYWORDS):
        return False, "device", "Device or device-associated product; exclude from molecular GNN."

    if any(k in target for k in VACCINE_KEYWORDS):
        return False, "vaccine", "Vaccine-like product; exclude from small-molecule GNN."

    if any(k in target for k in MIXTURE_KEYWORDS):
        return False, "mixture", "Mixture/polymer/supplement-like product; exclude or review."

    # 無機物はPubChem取得できる場合があるが、通常の低分子薬GNNでは別扱い。
    if any(k == ing or k == raw for k in INORGANIC_KEYWORDS):
        return False, "inorganic", "Inorganic/simple element or salt; exclude or review separately."

    return None, None, ""


def apply_brand_mapping(raw_name: str, ingredient: str) -> Tuple[str, str]:
    """
    ingredient候補を返す。
    """
    raw = norm_text(raw_name)
    ing = norm_text(ingredient)

    if raw in BRAND_TO_INGREDIENT:
        return BRAND_TO_INGREDIENT[raw], f"Brand/drug product name mapped to ingredient candidate: {BRAND_TO_INGREDIENT[raw]}"

    # ingredientが空欄ならrawを使う。
    if ing == "":
        return raw, "Ingredient was blank; copied from drugname_upper."

    return ing, ""


# ============================================================
# 3. PubChem API
# ============================================================

def load_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def request_json(url: str, sleep_sec: float = SLEEP_SEC) -> Optional[Dict[str, Any]]:
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            time.sleep(sleep_sec)
            r = requests.get(url, timeout=TIMEOUT_SEC)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404, 400):
                return None
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(1 + i)
        except Exception as e:
            last_err = repr(e)
            time.sleep(1 + i)
    return {"_error": last_err}


def pubchem_cids_by_name(name: str, cache: Dict[str, Any]) -> Dict[str, Any]:
    key = f"cids::{name}"
    if key in cache:
        return cache[key]

    url = f"{PUBCHEM_BASE}/compound/name/{quote(name)}/cids/JSON"
    data = request_json(url)
    if data is None:
        out = {"ok": False, "cids": [], "error": "not_found"}
    elif "_error" in data:
        out = {"ok": False, "cids": [], "error": data["_error"]}
    else:
        cids = data.get("IdentifierList", {}).get("CID", [])
        out = {"ok": bool(cids), "cids": cids, "error": None if cids else "no_cid"}
    cache[key] = out
    return out


def pubchem_properties_by_cid(cid: int, cache: Dict[str, Any]) -> Dict[str, Any]:
    props = [
        "CanonicalSMILES",
        "IsomericSMILES",
        "ConnectivitySMILES",
        "SMILES",
        "InChI",
        "InChIKey",
        "MolecularFormula",
        "MolecularWeight",
        "IUPACName",
    ]
    key = f"props::{cid}"
    if key in cache:
        return cache[key]

    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{','.join(props)}/JSON"
    data = request_json(url)
    if data is None:
        out = {"ok": False, "error": "not_found"}
    elif "_error" in data:
        out = {"ok": False, "error": data["_error"]}
    else:
        items = data.get("PropertyTable", {}).get("Properties", [])
        if items:
            out = {"ok": True, "props": items[0], "error": None}
        else:
            out = {"ok": False, "error": "no_properties"}
    cache[key] = out
    return out


def pubchem_synonyms_by_cid(cid: int, cache: Dict[str, Any]) -> Dict[str, Any]:
    key = f"synonyms::{cid}"
    if key in cache:
        return cache[key]

    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
    data = request_json(url)
    if data is None:
        out = {"ok": False, "synonyms": [], "error": "not_found"}
    elif "_error" in data:
        out = {"ok": False, "synonyms": [], "error": data["_error"]}
    else:
        infos = data.get("InformationList", {}).get("Information", [])
        syn = infos[0].get("Synonym", []) if infos else []
        out = {"ok": bool(syn), "synonyms": syn, "error": None if syn else "no_synonyms"}
    cache[key] = out
    return out


def query_pubchem(name: str, cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    nameからPubChem CIDと構造情報を取得する。
    """
    name = norm_text(name)
    if not name:
        return {
            "pubchem_ok": False,
            "pubchem_query": name,
            "pubchem_error": "blank_query",
        }

    cids_res = pubchem_cids_by_name(name, cache)
    if not cids_res.get("ok"):
        return {
            "pubchem_ok": False,
            "pubchem_query": name,
            "pubchem_error": cids_res.get("error"),
        }

    cid = int(cids_res["cids"][0])
    props_res = pubchem_properties_by_cid(cid, cache)
    if not props_res.get("ok"):
        return {
            "pubchem_ok": False,
            "pubchem_query": name,
            "pubchem_cid": cid,
            "pubchem_error": props_res.get("error"),
        }

    p = props_res["props"]
    syn_res = pubchem_synonyms_by_cid(cid, cache)
    synonyms = syn_res.get("synonyms", []) if syn_res.get("ok") else []

    # PubChemのレスポンスでは、環境・API仕様によりSMILESのキー名が
    # CanonicalSMILESではなくConnectivitySMILES等になる場合がある。
    canonical_smiles = (
        p.get("CanonicalSMILES")
        or p.get("ConnectivitySMILES")
        or p.get("SMILES")
        or p.get("IsomericSMILES")
    )
    isomeric_smiles = (
        p.get("IsomericSMILES")
        or p.get("CanonicalSMILES")
        or p.get("ConnectivitySMILES")
        or p.get("SMILES")
    )

    return {
        "pubchem_ok": True,
        "pubchem_query": name,
        "pubchem_cid": p.get("CID", cid),
        "canonical_smiles": canonical_smiles,
        "isomeric_smiles": isomeric_smiles,
        "inchi": p.get("InChI"),
        "inchikey": p.get("InChIKey"),
        "molecular_formula": p.get("MolecularFormula"),
        "molecular_weight": p.get("MolecularWeight"),
        "iupac_name": p.get("IUPACName"),
        "first_synonym": synonyms[0] if synonyms else None,
        "pubchem_error": None,
    }


# ============================================================
# 4. 任意のRDKitチェック
# ============================================================

def get_rdkit_checker():
    try:
        from rdkit import Chem
        return Chem
    except Exception:
        return None


def rdkit_check(smiles: Optional[str], Chem) -> Tuple[Optional[bool], Optional[str]]:
    if not smiles or pd.isna(smiles):
        return None, "no_smiles"
    if Chem is None:
        return None, "rdkit_not_installed"
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return False, "rdkit_parse_failed"
        return True, "rdkit_parse_ok"
    except Exception as e:
        return False, f"rdkit_error: {e}"


# ============================================================
# 5. メイン処理
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input Excel file")
    parser.add_argument("--sheet", default="curation_master", help="Input sheet name")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="PubChem cache JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for test run")
    parser.add_argument("--no-pubchem", action="store_true", help="Do not query PubChem; only apply rule-based flags")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)

    print(f"[INFO] Reading: {input_path}")
    df = pd.read_excel(input_path, sheet_name=args.sheet)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    required = ["drugname_upper", "ingredient"]
    for col in required:
        if col not in df.columns:
            print(f"[ERROR] Required column missing: {col}", file=sys.stderr)
            return 1

    # 欠損列を補う。
    for col in [
        "pubchem_cid",
        "canonical_smiles",
        "inchikey",
        "in_scope",
        "exclude_reason",
        "mapping_note",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # 自動補完用の列
    auto_cols = [
        "ingredient_auto",
        "ingredient_mapping_note",
        "pubchem_ok",
        "pubchem_query",
        "pubchem_error",
        "isomeric_smiles",
        "inchi",
        "molecular_formula",
        "molecular_weight",
        "iupac_name",
        "first_synonym",
        "rdkit_parse_ok",
        "rdkit_note",
        "in_scope_auto",
        "exclude_reason_auto",
        "autofill_review_flag",
        "autofill_note",
    ]
    for col in auto_cols:
        if col not in df.columns:
            df[col] = pd.NA

    Chem = get_rdkit_checker()

    records = []
    for idx, row in df.iterrows():
        raw = norm_text(row.get("drugname_upper"))
        current_ing = norm_text(row.get("ingredient"))
        original_smiles = row.get("canonical_smiles")

        ingredient_auto, brand_note = apply_brand_mapping(raw, current_ing)
        in_scope_initial, excl_initial, excl_note = initial_exclusion_reason(raw, ingredient_auto)

        pubchem_data: Dict[str, Any] = {}
        if not args.no_pubchem and in_scope_initial is not False and not looks_like_combination(ingredient_auto):
            print(f"[INFO] PubChem query {idx + 1}/{len(df)}: {raw} -> {ingredient_auto}")
            pubchem_data = query_pubchem(ingredient_auto, cache)
            save_cache(cache_path, cache)
        else:
            pubchem_data = {
                "pubchem_ok": False,
                "pubchem_query": ingredient_auto,
                "pubchem_error": "skipped_by_rule" if not args.no_pubchem else "no_pubchem_mode",
            }

        # 既存値があれば尊重し、空欄のみ補完する。
        pubchem_cid = row.get("pubchem_cid")
        canonical_smiles = row.get("canonical_smiles")
        inchikey = row.get("inchikey")

        if (pd.isna(pubchem_cid) or str(pubchem_cid).strip() == "") and pubchem_data.get("pubchem_ok"):
            pubchem_cid = pubchem_data.get("pubchem_cid")

        if (pd.isna(canonical_smiles) or str(canonical_smiles).strip() == "") and pubchem_data.get("pubchem_ok"):
            canonical_smiles = pubchem_data.get("canonical_smiles")

        if (pd.isna(inchikey) or str(inchikey).strip() == "") and pubchem_data.get("pubchem_ok"):
            inchikey = pubchem_data.get("inchikey")

        # RDKit check
        rdkit_ok, rdkit_note = rdkit_check(canonical_smiles, Chem)

        # 分子量に基づく追加レビュー
        mw = pubchem_data.get("molecular_weight")
        try:
            mw_float = float(mw) if mw is not None and not pd.isna(mw) else None
        except Exception:
            mw_float = None

        in_scope_auto = in_scope_initial
        exclude_reason_auto = excl_initial
        review_notes = []

        if brand_note:
            review_notes.append(brand_note)
        if excl_note:
            review_notes.append(excl_note)

        if in_scope_initial is None:
            if pubchem_data.get("pubchem_ok") and canonical_smiles and rdkit_ok is not False:
                in_scope_auto = True
            else:
                in_scope_auto = pd.NA

        if mw_float is not None:
            if mw_float >= MW_EXCLUDE_THRESHOLD:
                in_scope_auto = False
                exclude_reason_auto = exclude_reason_auto or "large_molecule"
                review_notes.append(f"Molecular weight >= {MW_EXCLUDE_THRESHOLD}; exclude or review separately.")
            elif mw_float >= MW_REVIEW_THRESHOLD:
                review_notes.append(f"Molecular weight >= {MW_REVIEW_THRESHOLD}; manual review recommended.")

        if looks_like_combination(raw) or looks_like_combination(ingredient_auto):
            in_scope_auto = False
            exclude_reason_auto = exclude_reason_auto or "combination_product"

        # PubChem未取得の場合は要確認
        review_flag = False
        if in_scope_auto is pd.NA or pd.isna(in_scope_auto):
            review_flag = True
        if not pubchem_data.get("pubchem_ok") and in_scope_initial is not False:
            review_flag = True
        if rdkit_ok is False:
            review_flag = True
        if brand_note:
            review_flag = True

        record = row.to_dict()
        record.update({
            "ingredient_auto": ingredient_auto,
            "ingredient_mapping_note": brand_note,
            "pubchem_ok": pubchem_data.get("pubchem_ok"),
            "pubchem_query": pubchem_data.get("pubchem_query"),
            "pubchem_error": pubchem_data.get("pubchem_error"),
            "pubchem_cid": pubchem_cid,
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "isomeric_smiles": pubchem_data.get("isomeric_smiles"),
            "inchi": pubchem_data.get("inchi"),
            "molecular_formula": pubchem_data.get("molecular_formula"),
            "molecular_weight": mw_float,
            "iupac_name": pubchem_data.get("iupac_name"),
            "first_synonym": pubchem_data.get("first_synonym"),
            "rdkit_parse_ok": rdkit_ok,
            "rdkit_note": rdkit_note,
            "in_scope_auto": in_scope_auto,
            "exclude_reason_auto": exclude_reason_auto,
            "autofill_review_flag": review_flag,
            "autofill_note": " | ".join([n for n in review_notes if n]),
        })

        # 既存in_scope/exclude_reasonが空欄なら、自動候補を入れる。
        if pd.isna(record.get("in_scope")) or str(record.get("in_scope")).strip() == "":
            record["in_scope"] = in_scope_auto

        if pd.isna(record.get("exclude_reason")) or str(record.get("exclude_reason")).strip() == "":
            record["exclude_reason"] = exclude_reason_auto

        # ingredientは自動補正候補を反映。ただし元値との差は ingredient_auto にも残す。
        record["ingredient"] = ingredient_auto

        # mapping_noteに追記
        existing_note = "" if pd.isna(record.get("mapping_note")) else str(record.get("mapping_note"))
        auto_note = record.get("autofill_note") or ""
        if auto_note:
            record["mapping_note"] = (existing_note + " | " + auto_note).strip(" |")

        records.append(record)

    out_df = pd.DataFrame(records)

    # 列順を整理
    preferred_cols = [
        "drugname_upper",
        "ingredient",
        "ingredient_auto",
        "task_group",
        "label_confidence",
        "model_label",
        "label_weight",
        "pubchem_cid",
        "canonical_smiles",
        "isomeric_smiles",
        "inchi",
        "inchikey",
        "molecular_formula",
        "molecular_weight",
        "iupac_name",
        "first_synonym",
        "in_scope",
        "in_scope_auto",
        "exclude_reason",
        "exclude_reason_auto",
        "mapping_note",
        "autofill_review_flag",
        "autofill_note",
        "pubchem_ok",
        "pubchem_query",
        "pubchem_error",
        "rdkit_parse_ok",
        "rdkit_note",
    ]
    remaining_cols = [c for c in out_df.columns if c not in preferred_cols]
    out_df = out_df[[c for c in preferred_cols if c in out_df.columns] + remaining_cols]

    # サマリー
    def safe_count(series, value):
        try:
            return int((series == value).sum())
        except Exception:
            return 0

    summary = pd.DataFrame([
        ["n_total", len(out_df)],
        ["pubchem_ok_true", safe_count(out_df["pubchem_ok"], True)],
        ["canonical_smiles_nonmissing", int(out_df["canonical_smiles"].notna().sum())],
        ["inchikey_nonmissing", int(out_df["inchikey"].notna().sum())],
        ["in_scope_true", safe_count(out_df["in_scope"], True)],
        ["in_scope_false", safe_count(out_df["in_scope"], False)],
        ["review_flag_true", safe_count(out_df["autofill_review_flag"], True)],
        ["rdkit_installed", Chem is not None],
    ], columns=["item", "value"])

    by_group = (
        out_df
        .groupby(["task_group", "label_confidence"], dropna=False)
        .agg(
            n=("drugname_upper", "count"),
            pubchem_ok=("pubchem_ok", lambda x: int((x == True).sum())),
            smiles_nonmissing=("canonical_smiles", lambda x: int(x.notna().sum())),
            in_scope_true=("in_scope", lambda x: int((x == True).sum())),
            in_scope_false=("in_scope", lambda x: int((x == False).sum())),
            review_flag=("autofill_review_flag", lambda x: int((x == True).sum())),
        )
        .reset_index()
    )

    review_df = out_df[out_df["autofill_review_flag"] == True].copy()
    excluded_df = out_df[out_df["in_scope"] == False].copy()
    inscope_df = out_df[out_df["in_scope"] == True].copy()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = output_dir / f"drug_structure_master_curated_v1_autofill_v2_{ts}.xlsx"
    summary_xlsx = output_dir / f"structure_curation_summary_v2_{ts}.xlsx"

    print(f"[INFO] Writing: {out_xlsx}")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="autofill_master", index=False)
        inscope_df.to_excel(writer, sheet_name="in_scope_candidates", index=False)
        excluded_df.to_excel(writer, sheet_name="excluded_candidates", index=False)
        review_df.to_excel(writer, sheet_name="manual_review", index=False)
        by_group.to_excel(writer, sheet_name="summary_by_group", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

        # 幅調整
        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes = "A2"
            for col_cells in worksheet.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                worksheet.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 60)

    print(f"[INFO] Writing: {summary_xlsx}")
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        by_group.to_excel(writer, sheet_name="summary_by_group", index=False)
        review_df.to_excel(writer, sheet_name="manual_review", index=False)

    print("[INFO] Done.")
    print(summary.to_string(index=False))
    print("")
    print("[INFO] Output files:")
    print(f"  {out_xlsx}")
    print(f"  {summary_xlsx}")
    print(f"  {cache_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
