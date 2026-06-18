Interpretable SHAP analysis using RDKit descriptors

Input:
  output/gnn_dataset_model_ready_colab_20260521_085012.csv

Model:
  RandomForestClassifier(n_estimators=500, min_samples_leaf=2)
  Features: RDKit descriptors, feature_set=core
  Sample weights: label_weight

Important interpretation:
  This is an interpretation-oriented descriptor model.
  It is not the primary performance model.
  The main performance model remains the ECFP-based Random Forest evaluated by random and scaffold splits.
  Beeswarm features are directly interpretable RDKit descriptors or fragment-count descriptors.
  SHAP values describe associations learned by the final model, not causal effects.

Main outputs:
  interpretable_shap_summary_20260522_041158.xlsx
  interpretable_feature_matrix_20260522_041158.csv
  interpretable_shap_predictions_20260522_041158.csv
  interpretable_shap_feature_importance_20260522_041158.csv
  interpretable_shap_long_top_features_20260522_041158.csv
  figure_interpretable_shap_beeswarm.png/pdf/svg
  figure_interpretable_shap_bar.png/pdf/svg
  waterfall_plots/*.png/pdf/svg
