"""
Script: make_perturbed_genotype_datasets.py

This script generates a set of perturbed binary genotype datasets by simulating the effect
of gene-disease associations under varying allele frequencies and odds ratios.

The simulation process involves:
1. Loading an original binary genotype matrix (e.g., somatic mutations)
    Optional: restrict to smaller number of features.
2. Selecting a single gene (default: 'AR') to perturb.
3. Splitting the samples into two groups: controls (class 0) and cases (class 1), with a fixed fraction of samples assigned as cases.
4. Flipping a defined proportion of 1s into the selected gene column for both control and case groups, based on:
   - Control allele frequency
   - Odds ratio (used to calculate case frequency: case_freq = (OR * control_freq) / (1 - control_freq + OR * control_freq)
        N.B.: the Relative Risk = min(control_freq * OR, 1.0))
5. Saving the resulting genotype matrix and associated binary labels to disk.
6. Logging metadata for each dataset in a summary CSV and to Weights & Biases (wandb).

Each output dataset is uniquely named using:
- The perturbed gene
- The odds ratio used
- The exact frequencies applied to each group (formatted safely for filenames)

Example output filename:
    somatic_mut_gene-AR_OR-2_caseFreq-0p1_ctrlFreq-0p05.csv

Intended use:
This script is designed for benchmarking predictive models by evaluating their ability to detect
simulated signal under known effect sizes and allele frequencies.
"""

# === IMPORTS === #
import logging
import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import wandb

# === LOGGING SETUP === #
logging.basicConfig(
    # filename="make_perturbed_genotype_datasets.log",
    encoding="utf-8",
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# === WANDB INIT === #
wandb.login()
wandb.init(
    project="prostate_met_status",
    group="simulation_single_gene_perturbation_001",
    notes="Generate perturbed genotype datasets for testing model performance. Corrected formula for case_freq so calculating based on OR (not relative risk).",
)


# === HELPER FUNCTION === #
def create_target_dataframe(df_index, sample_subset):
    """
    Create a target DataFrame with an 'is_met' column indicating whether
    each sample is in the sample_subset.

    Args:
        df_index (pd.Index): The index of the original DataFrame (sample names).
        sample_subset (list): A list of samples (or indices) to mark as 'met'.

    Returns:
        pd.DataFrame: A DataFrame with the same index as the input and an 'is_met' column.
    """
    target = pd.DataFrame({"Tumor_Sample_Barcode": df_index}).set_index("Tumor_Sample_Barcode")
    target["is_met"] = target.index.map(lambda x: 1 if x in sample_subset else 0)
    return target


def safe_float_str(val, digits=4):
    """
    Convert a float to a string with fixed decimal precision and replace '.' with 'p'
    for filename safety.
    """
    return str(round(val, digits)).replace(".", "p")


def format_odds_ratio(oratio):
    """
    Format an odds ratio: if it's an integer, drop the decimal.
    Otherwise, keep 2 decimal digits.
    """
    return str(int(oratio)) if float(oratio).is_integer() else safe_float_str(oratio, digits=2)


def create_splits(samples, case_samples, save_split_dir=None):
    """
    Create stratified train/validation/test splits for the dataset in a 70/15/15 ratio.
    The function saves the splits to CSV files in the specified directory.
    If no directory is provided, the splits are not saved.
    Args:
        samples (list): List of all sample names.
        case_samples (list): List of case sample names.
        save_split_dir (str, optional): Directory to save the split files. Defaults to None.
    """
    # Create labels (1 for case, 0 for control)
    labels = pd.Series(0, index=samples)
    labels.loc[case_samples] = 1

    # Stratified split: 70% train, 30% temp (val + test)
    train_inds, temp_inds = train_test_split(labels.index, test_size=0.30, stratify=labels, random_state=SEED)

    # Stratified split: 15% val, 15% test from temp
    temp_labels = labels.loc[temp_inds]
    val_inds, test_inds = train_test_split(temp_inds, test_size=0.50, stratify=temp_labels, random_state=SEED)

    for name, inds in zip(["training", "validation", "test"], [train_inds, val_inds, test_inds]):
        n_cases = labels.loc[inds].sum()
        logger.info(f"{name.title()} set: {len(inds)} samples, {n_cases} cases ({n_cases / len(inds):.2%})")

    if save_split_dir:
        save_split("training", labels, train_inds, save_split_dir)
        save_split("validation", labels, val_inds, save_split_dir)
        save_split("test", labels, test_inds, save_split_dir)
        logger.info(f"Stratified training/validation/test index files saved to {save_split_dir}")


def save_split(who, labels, indices, save_dir):
    df_split = pd.DataFrame(index=indices)
    df_split["response"] = labels.loc[indices]
    df_split.index.name = "id"
    df_split.to_csv(os.path.join(save_dir, f"{who}_set.csv"))


def compute_case_freq_from_or(control_freq: float, odds_ratio: float) -> float:
    """
    Calculate case group mutation frequency from control group frequency and odds ratio.

    Formula:
        case_freq = (OR * control_freq) / (1 - control_freq + OR * control_freq)

    Where:
        - OR: odds ratio of mutation in case vs. control group
        - control_freq: mutation frequency in the control group
        - case_freq: resulting mutation frequency in the case group

    This formula derives from the definition of odds ratio:
        OR = (P_case / (1 - P_case)) / (P_control / (1 - P_control))
    and is solved for P_case.

    Args:
        control_freq (float): Mutation frequency in the control group (between 0 and 1).
        odds_ratio (float): Desired odds ratio for the case group relative to control.

    Returns:
        float: Computed mutation frequency in the case group.
    """
    if not (0 <= control_freq < 1):
        raise ValueError("control_freq must be in the interval [0, 1).")
    if odds_ratio <= 0:
        raise ValueError("odds_ratio must be positive.")

    numerator = odds_ratio * control_freq
    denominator = 1 - control_freq + numerator
    return numerator / denominator


# === CONFIGURATION === #
DATA_DIR = "/mnt/disks/gmiller_data1/pnet_germline/processed/wandb-group-data_prep_germline_tier12_and_somatic/converted-IDs-to-somatic_imputed-germline_True_imputed-somatic_False_paired-samples-True/wandb-run-id-u5yt90p1"
SAVE_DIR = f"/mnt/disks/gmiller_data1/pnet_germline/processed/perturbed_genotype_datasets/p1000_somatic_mut/wandb-run-id-{wandb.run.id}"
DATA_FILENAME = "somatic_mut.csv"  # this is the file used as the 'background' dataset to perturb
summary_csv_path = os.path.join(SAVE_DIR, f"perturbation_summary_{wandb.run.id}.csv")
SEED = 42
np.random.seed(SEED)

logger.info(f"Files will be saved to: {SAVE_DIR}")
logger.info(f"Summary CSV will be saved to: {summary_csv_path}")
logger.info(f"Original data directory: {DATA_DIR}")
logger.info(f"Original data filename: {DATA_FILENAME}")
logger.info(f"Setting random seed to {SEED}")

# === LOAD DATA === #
logger.info(f"Loading original dataset from: {DATA_DIR}")
logger.info(f"Data filename: {DATA_FILENAME}")
data_f = os.path.join(DATA_DIR, DATA_FILENAME)
df = pd.read_csv(data_f, index_col=0)
samples = df.index.tolist()
genes = df.columns.tolist()
n_samples = len(samples)
n_genes = len(genes)
logger.info(f"Loaded dataset with shape: {df.shape}")

# === PARAMETERS === #
odds_ratios = [1, 1.1, 2, 10, 30]
control_frequencies = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
fractions_samples = [0.5]
gene_to_perturb = "AR"
n_features_to_keep = [10, 100]  # add n_genes if want to run all

assert gene_to_perturb in df.columns, f"Gene '{gene_to_perturb}' not found in dataset."

# === WANDB CONFIG UPDATE === #
hparams = {
    "data_dir": DATA_DIR,
    "save_dir": SAVE_DIR,
    "data_filename": DATA_FILENAME,
    "seed": SEED,
    "fractions_samples": fractions_samples,
    "n_samples": n_samples,
    "n_genes": n_features_to_keep,
    "summary_csv_path": summary_csv_path,
    "train_ind_file": os.path.join(SAVE_DIR, "training_set.csv"),
    "val_ind_file": os.path.join(SAVE_DIR, "validation_set.csv"),
    "test_ind_file": os.path.join(SAVE_DIR, "test_set.csv"),
    "input_data_wandb_id": "u5yt90p1",
}

wandb.config.update(hparams)

# === OUTPUT DIR === #
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger.info(f"Output directory created or already exists: {SAVE_DIR}")
except Exception as e:
    logger.error(f"Failed to create output directory {SAVE_DIR}: {e}")
    raise

# === METADATA TRACKING === #
summary_records = []

# === MAIN LOOP === #
for n_features in n_features_to_keep:
    # Subsample feature columns
    remaining_genes = [g for g in df.columns if g != gene_to_perturb]
    selected_genes = np.random.choice(remaining_genes, size=n_features - 1, replace=False).tolist()
    selected_genes.append(gene_to_perturb)  # Make sure the perturbed gene is always included

    df_subset = df[selected_genes]
    logger.info(f"Using {n_features} features for this run, including the perturbed gene {gene_to_perturb}")

    for frac_sample in fractions_samples:
        n_case = int(n_samples * frac_sample)
        n_control = n_samples - n_case

        # Randomly split samples into "control" and "case"
        shuffled_samples = np.random.permutation(samples)
        case_samples = shuffled_samples[:n_case]
        control_samples = shuffled_samples[n_case:]

        logger.info(f"Making train/val/test split with {n_case} cases and {n_control} controls.")
        create_splits(samples, case_samples, save_split_dir=SAVE_DIR)

        for control_freq, odds_ratio in product(control_frequencies, odds_ratios):
            logger.info(
                f"Generating dataset with control_freq: {control_freq}, odds_ratio: {odds_ratio}, gene_to_perturb: {gene_to_perturb}"
            )
            # Determine case frequency from control_freq and odds_ratio
            # case_freq = min(control_freq * odds_ratio, 1.0)  # Cap at 1.0 to avoid invalid probabilities
            case_freq = compute_case_freq_from_or(control_freq, odds_ratio)

            df_copy = df_subset.copy()
            df_copy[gene_to_perturb] = 0  # Clear out the gene first

            # Flip a % of control samples to 1 for the gene
            n_control_flip = max(1, int(n_control * control_freq))
            control_flips = np.random.choice(control_samples, size=n_control_flip, replace=False)
            df_copy.loc[control_flips, gene_to_perturb] = 1

            # Flip a % of case samples to 1 for the gene
            n_case_flip = max(1, int(n_case * case_freq))
            case_flips = np.random.choice(case_samples, size=n_case_flip, replace=False)
            df_copy.loc[case_flips, gene_to_perturb] = 1

            # Create target dataframe
            target = create_target_dataframe(df_copy.index, case_samples)

            # File naming
            suffix = (
                f"gene-{gene_to_perturb}_"
                f"OR-{format_odds_ratio(odds_ratio)}_"
                f"ctrlFreq-{safe_float_str(control_freq)}_"
                f"caseFreq-{safe_float_str(case_freq)}_"
                f"nFeatures-{n_features}"
            )

            logger.info(f"Output suffix: {suffix}")
            out_path = os.path.join(SAVE_DIR, f"somatic_mut_{suffix}.csv")
            target_out_path = os.path.join(SAVE_DIR, f"y_{suffix}.csv")

            # Save
            assert df_copy.shape[1] == n_features, f"Expected {n_features} features, but got {df.shape[1]}"
            df_copy.to_csv(out_path)
            target.to_csv(target_out_path, index=True)
            logger.info(f"Saved dataset: {out_path}")

            # Record metadata
            summary_records.append(
                {
                    "suffix": suffix,
                    "out_data_file": out_path,
                    "out_target_file": target_out_path,
                    "gene": gene_to_perturb,
                    "control_freq": control_freq,
                    "case_freq": case_freq,
                    "odds_ratio": odds_ratio,
                    "n_case_flips": n_case_flip,
                    "n_control_flips": n_control_flip,
                    "n_case": n_case,
                    "n_control": n_control,
                    "n_features": n_features,
                    "wandb_run_id": wandb.run.id,
                }
            )

# === SAVE SUMMARY TO WANDB === #
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(summary_csv_path, index=False)
logger.info(f"Saved perturbation summary to {summary_csv_path}")
logger.info("Finished generating perturbed datasets.")
wandb.finish()
