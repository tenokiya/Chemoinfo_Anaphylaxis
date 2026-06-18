# Repository File Policy

## Include in GitHub repository

- Source code required for reproducing analyses.
- Processed model-ready dataset.
- Curated structure master and curation audit file.
- Final model summaries, selected predictions, manuscript tables, and final figures.
- Documentation required for reuse.

## Exclude from regular Git tracking

- Raw or minimally processed FAERS/AEMS exports unless redistribution and file-size policy are finalized.
- Exploratory notebooks containing personal Colab metadata or execution history.
- Duplicate timestamped outputs superseded by newer results.
- Redundant PNG/PDF/SVG duplicates unless needed for manuscript or review.

## Consider for Zenodo or GitHub Releases

- Large raw/minimal FAERS/AEMS exports.
- Complete figure output archive.
- Complete intermediate work directory for audit purposes.
