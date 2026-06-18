# Chemoinfo Anaphylaxis

This repository contains code, curated data, and result files for a chemoinformatics analysis of anaphylactic shock-related pharmacovigilance labels using molecular structures.

## Repository status

Draft prepared before manuscript submission. Last curated file-selection check: 2026-06-18 11:36 JST.

## Scope

- Structure curation from drug names and PubChem-derived identifiers.
- FAERS/AEMS-derived anaphylactic shock and broad anaphylaxis labels.
- ECFP4-based baseline models using logistic regression, random forest, and LightGBM.
- Scaffold-aware evaluation.
- Sensitivity analyses for label definitions and treatment-drug confounding.
- SHAP-based post hoc interpretation.

## Recommended starting point

For reproducing the modeling results from the processed dataset, start with:

```bash
pip install -r requirements.txt
python scripts/03_train_ecfp_baseline.py --input data/processed/model_ready.csv --output-dir results/reproduced/ecfp_baseline
python scripts/04_scaffold_split_ecfp.py --input data/processed/model_ready.csv --output-dir results/reproduced/scaffold_split
python scripts/05_sensitivity_analysis_ecfp_rf.py --input data/processed/model_ready.csv --output-dir results/reproduced/sensitivity
python scripts/10_treatment_confounding_sensitivity.py --input data/processed/model_ready.csv --rule-file data/processed/treatment_confounding_drug_rules.csv --output-dir results/reproduced/treatment_sensitivity
```

## Data note

`data/processed/model_ready.csv` is the recommended model-ready input file. The original FAERS/AEMS minimal export files used during dataset construction are not included in this draft repository because redistribution conditions and file-size handling should be checked before public release.

## Important methodological note

The molecular structure key in the model-ready dataset uses canonical SMILES generated without explicit stereochemical separation in the current workflow. Stereoisomer-level distinctions should therefore not be over-interpreted unless the workflow is revised to use isomeric SMILES.

## License note

`LICENSE_CODE_MIT_DRAFT.txt` is a draft license for code only. Data redistribution terms should be finalized before making the repository public.
