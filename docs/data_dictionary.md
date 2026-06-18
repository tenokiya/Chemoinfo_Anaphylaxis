# Data Dictionary

Last checked: 2026-06-18 11:36 JST

## `data/processed/model_ready.csv`

Main model-ready dataset. One row per drug/structure candidate used for modeling.

- Rows: 106
- Columns: 42

| Column | Note |
|---|---|
| `analysis_drug_name` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `a_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `b_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `c_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `d_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `total_exposed_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `ror_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `lower_ci_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `upper_ci_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `se_logror_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `a_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `b_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `c_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `d_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `total_exposed_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `ror_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `lower_ci_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `upper_ci_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `se_logror_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `total_exposed_psss_for_broad` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `shock_count_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `broad_count_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `pubchem_cid` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `canonical_smiles` | Canonical SMILES used for molecular fingerprints; current workflow does not emphasize stereochemical separation. |
| `inchikey` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `in_scope` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `exclude_reason` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `n_raw_names_mapped` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `raw_names_mapped` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `expected_shock` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `expected_broad` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `positive_high_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `positive_moderate_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `negative_high_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `negative_moderate_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `negative_expanded_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `faers_signal_class` | Rule-based FAERS/AEMS signal class used to derive labels. |
| `structure_available` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `in_scope_bool` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `usable_for_model` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `model_label` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `label_weight` | Generated/curated variable; see manuscript methods and scripts for derivation. |

## `data/processed/all_labels.csv`

Drug-level label table derived from FAERS/AEMS minimal export.

- Rows: 7918
- Columns: 42

| Column | Note |
|---|---|
| `analysis_drug_name` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `a_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `b_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `c_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `d_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `total_exposed_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `ror_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `lower_ci_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `upper_ci_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `se_logror_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `a_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `b_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `c_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `d_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `total_exposed_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `ror_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `lower_ci_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `upper_ci_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `se_logror_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `total_exposed_psss_for_broad` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `shock_count_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `broad_count_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `pubchem_cid` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `canonical_smiles` | Canonical SMILES used for molecular fingerprints; current workflow does not emphasize stereochemical separation. |
| `inchikey` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `in_scope` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `exclude_reason` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `n_raw_names_mapped` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `raw_names_mapped` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `expected_shock` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `expected_broad` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `positive_high_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `positive_moderate_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `negative_high_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `negative_moderate_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `negative_expanded_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `faers_signal_class` | Rule-based FAERS/AEMS signal class used to derive labels. |
| `structure_available` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `in_scope_bool` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `usable_for_model` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `model_label` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `label_weight` | Generated/curated variable; see manuscript methods and scripts for derivation. |

## `data/processed/treatment_confounded_model_ready.csv`

Model-ready dataset with treatment-confounding flags.

- Rows: 106
- Columns: 53

| Column | Note |
|---|---|
| `analysis_drug_name` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `a_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `b_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `c_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `d_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `total_exposed_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `ror_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `lower_ci_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `upper_ci_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `se_logror_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `a_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `b_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `c_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `d_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `total_exposed_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `ror_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `lower_ci_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `upper_ci_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `se_logror_ps` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `total_exposed_psss_for_broad` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `shock_count_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `broad_count_psss` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `pubchem_cid` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `canonical_smiles` | Canonical SMILES used for molecular fingerprints; current workflow does not emphasize stereochemical separation. |
| `inchikey` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `in_scope` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `exclude_reason` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `n_raw_names_mapped` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `raw_names_mapped` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `expected_shock` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `expected_broad` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `positive_high_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `positive_moderate_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `negative_high_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `negative_moderate_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `negative_expanded_flag` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `faers_signal_class` | Rule-based FAERS/AEMS signal class used to derive labels. |
| `structure_available` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `in_scope_bool` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `usable_for_model` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `model_label` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `label_weight` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `analysis_drug_name_norm` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `treatment_confounded_candidate` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `treatment_confounded_category` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `treatment_review_level` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `matched_treatment_rule` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `treatment_review_comment` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `is_corticosteroid_candidate` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `is_antihistamine_candidate` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `is_core_treatment_candidate` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `is_broad_treatment_candidate` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `recommended_label_status` | Generated/curated variable; see manuscript methods and scripts for derivation. |

## `data/processed/treatment_confounding_drug_rules.csv`

Rule table for identifying treatment-related drug groups.

- Rows: 55
- Columns: 4

| Column | Note |
|---|---|
| `analysis_drug_name` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `confounding_category` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `review_level` | Generated/curated variable; see manuscript methods and scripts for derivation. |
| `comment` | Generated/curated variable; see manuscript methods and scripts for derivation. |
