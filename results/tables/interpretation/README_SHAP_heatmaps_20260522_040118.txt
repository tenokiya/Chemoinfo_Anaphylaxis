SHAP and atom-level heatmap interpretation outputs

Input file:
  output/gnn_dataset_model_ready_colab_20260521_085012.csv

Model:
  ECFP4 radius=2, nBits=2048, useChirality=True
  RandomForestClassifier(n_estimators=500, min_samples_leaf=2)
  Trained on all model-ready compounds with label_weight as sample_weight.

Interpretation:
  SHAP values are computed for the positive class.
  Atom-level heatmaps approximate atom contributions by distributing active ECFP bit SHAP values over Morgan environments from RDKit bitInfo.
  Red indicates contribution toward the positive class; blue indicates contribution toward the negative class.
  These maps are exploratory and should not be interpreted as causal structural alerts.

Main outputs:
  shap_interpretation_summary_20260522_040118.xlsx
  shap_predictions_20260522_040118.csv
  shap_global_bit_importance_20260522_040118.csv
  shap_selected_drugs_20260522_040118.csv
  shap_atom_contributions_20260522_040118.csv
  shap_bit_contributions_selected_20260522_040118.csv
  heatmaps_svg/*.svg
  heatmaps_png/*.png
