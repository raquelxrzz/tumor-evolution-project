# tumor-evolution-project

# Tumour Evolution Project

## Project overview

This project aims to develop a bioinformatics workflow for studying tumour evolution and intratumour heterogeneity using somatic mutations and copy-number data.

The current focus is the preparation and integration of VCF and Sequenza outputs for DPClust, which will subsequently be used to identify mutation clusters and investigate tumour subclonal composition.

## Repository structure

- `data/` — Original VCF and Sequenza files (stored locally).
- `scripts/` — Python scripts for data preprocessing and quality control.
- `output/dpclust_inputs/` — Generated DPClust input files and mutation-level QC results.
- `output/reports/` — Sample-specific preprocessing reports.

## Preprocessing workflow

The preprocessing pipeline performs the following steps:

1. Load VCF and Sequenza files.
2. Filter somatic single nucleotide variants (SNVs).
3. Extract reference and alternative read counts.
4. Match mutations between VCF and Sequenza.
5. Assign segment-level copy-number estimates.
6. Calculate DPClust input parameters.
7. Validate and export the prepared data.
8. Perform quality control and consistency checks.

## How to run

Install the required Python packages:

`pip install pandas numpy`

Select the sample in `scripts/prepare_dpclust_input.py` and run:

`python scripts/prepare_dpclust_input.py`

The generated files and preprocessing report are automatically saved in the `output/` directory.

## Current status

- [x] DPClust input preprocessing pipeline.
- [x] Quality control and report generation.
- [x] Preprocessing of all six PCA035 tumour regions.
- [ ] Preprocessing of PCA036.
- [ ] Validation of tumour purity and mutation copy-number estimates.
- [ ] Single-sample DPClust analysis.
- [ ] Multi-region DPClust analysis.
- [ ] Tumour evolutionary reconstruction.

## Preliminary findings

Initial quality control showed consistent variant allele frequencies between VCF and Sequenza. However, discrepancies were identified between reported and implied tumour purity values.

These inconsistencies require further investigation before proceeding with DPClust analysis.
