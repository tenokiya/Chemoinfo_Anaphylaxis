# Reproduction Guide

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Main model evaluation

```bash
python scripts/03_train_ecfp_baseline.py \
  --input data/processed/model_ready.csv \
  --output-dir results/reproduced/ecfp_baseline

python scripts/04_scaffold_split_ecfp.py \
  --input data/processed/model_ready.csv \
  --output-dir results/reproduced/scaffold_split
```

## 3. Sensitivity analyses

```bash
python scripts/05_sensitivity_analysis_ecfp_rf.py \
  --input data/processed/model_ready.csv \
  --output-dir results/reproduced/sensitivity

python scripts/10_treatment_confounding_sensitivity.py \
  --input data/processed/model_ready.csv \
  --rule-file data/processed/treatment_confounding_drug_rules.csv \
  --output-dir results/reproduced/treatment_sensitivity
```

## 4. Interpretation analyses

```bash
python scripts/08_interpret_final_rf_shap_heatmap.py \
  --input data/processed/model_ready.csv \
  --search-dir results/reproduced/scaffold_split \
  --output-dir results/reproduced/interpretation_ecfp

python scripts/09_interpretable_shap_descriptors.py \
  --input data/processed/model_ready.csv \
  --search-dir results/reproduced/scaffold_split \
  --output-dir results/reproduced/interpretation_descriptors
```

## 5. Full dataset construction

The full reconstruction from FAERS/AEMS minimal exports requires the raw/minimal files that are not included in this draft repository. Place them under `data/faers_minimal_export/` or update command-line arguments accordingly.
