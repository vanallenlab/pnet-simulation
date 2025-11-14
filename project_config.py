from pathlib import Path

# Path('PATH/TO/REPO')
PROJECT_DIR = Path("/mnt/disks/gmiller_data1/pnet-simu-private/")
DATA_DIR = PROJECT_DIR / "data"  # where to find data
RESULT_DIR = PROJECT_DIR / "results"  # where to save results (performance metrics, processed outputs, etc.)
FIGURE_DIR = PROJECT_DIR / "figures"  # where to save figures

GERMLINE_DATA_DIR = (
    DATA_DIR / "germline"
)  # where to find all non-finalized germline data (raw VCFs, intermediate files, metadata, etc.)
PROCESSED_GERMLINE_VCFS_DIR = (
    GERMLINE_DATA_DIR / "processed_germline_vcfs"
)  # where to save processed germline VCFs (universal filters applied, pathogenicity filters applied, etc.)
PNET_DATABASE_DIR = DATA_DIR / "pnet_database"  # path to pnet_germline database extracted from the P1000 manuscript
# RAW_DATA_DIR = DATA_DIR / "raw"  # where to find raw data (original datasets: Reactome, raw somatic/germline data, etc.)
PROCESSED_DATA_DIR = (
    DATA_DIR / "processed"
)  # where to find model-ready, processed data (harmonized data, simulated datasets, etc.)


# Joint distribution simulations (Figure 1)
FIGURE_1_DIR = FIGURE_DIR / "figure_1_joint_distributions"
RESULT_DIR_FIGURE_1 = RESULT_DIR / "figure_1_joint_distributions"

# Single-gene spike-in perturbations (Figure 2)
FIGURE_2_DIR = FIGURE_DIR / "figure_2_single_gene_spike_in"
RESULT_DIR_FIGURE_2 = RESULT_DIR / "figure_2_single_gene_spike_in"

# Empirical analysis with P1000 somatic and germline data (Figure 3)
FIGURE_3_DIR = FIGURE_DIR / "figure_3_empirical_analysis"
RESULT_DIR_FIGURE_3 = RESULT_DIR / "figure_3_empirical_analysis"

# Supplementary figures
SUPPLEMENTARY_FIGURE_DIR = FIGURE_DIR / "supplementary_figures"
SUPPLEMENTARY_TABLES_DIR = RESULT_DIR / "supplementary_tables"
