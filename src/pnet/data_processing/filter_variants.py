"""
Additional filtering: germline-specific factors (e.g. MAF <0.01)

Here we use the function that the germline folks use to filter down to a reasonable subset of variants (aka those likely to be high impact and biologically relevant, not just popping out due to population structure).

Modified from this notebook: https://app.terra.bio/#workspaces/vanallen-firecloud-nih/Germline_pipeline_components/analysis/launch/Jan2023_PathogenicVariantFilteringNotebook.ipynb
"""

import numpy as np
import pandas as pd
from pnet.data_processing import utils  # for is_binarized, relocate, binarize_df, restrict_vcf_to_genotype_columns
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


######
# Defining inclusion-exclusion criteria for variants with conflicting interpretation
######
def conflicting_filter_criteria(row):
    """
    Defining inclusion-exclusion criteria for variants with conflicting interpretation
    """
    patho_consolidated = row["Pathogenic_consolidated"]
    benign_consolidated = row["Benign_consolidated"]
    uncertain_consolidated = row["Uncertain_significance"]
    if patho_consolidated < 2:
        return "exclude"
    elif patho_consolidated < benign_consolidated:
        return "exclude"
    elif patho_consolidated < uncertain_consolidated:
        return "exclude"
    else:
        return "include"


######
# Defining helper functions
######
# Subsets the vep df to variants in the predefined gene list
def subset_to_gene_list(vep_df, gene_list):
    df_subset = vep_df[vep_df["SYMBOL"].isin(gene_list)].copy()
    return df_subset


# Consolidate the consequence of each variant to a single consequence
def add_consolidated_consequence(vep_df):
    vep_copy = vep_df.copy()
    vep_copy["Consequence_consolidated"] = vep_copy["Consequence"].str.split(",").str[0]
    return vep_copy


# Subset to varaints that has severe consequences as defined by containing one of the keywords
def subset_to_severe_consequence(vep_df, severe_consequences=None):
    if severe_consequences is None:
        severe_consequences = "splice_donor_variant|frameshift_variant|stop_gained|splice_acceptor_variant|transcript_ablation|stop_lost|start_lost|transcript_amplification|feature_elongation|feature_truncation"
    vep_copy = add_consolidated_consequence(vep_df)
    vep_copy = vep_copy[vep_copy["Consequence_consolidated"].str.contains(severe_consequences)]
    return vep_copy


# Subset to varaints that has moderate consequences as defined by containing one of the keywords
def subset_to_moderate_consequence(vep_df, moderate_consequences=None):
    if moderate_consequences is None:
        moderate_consequences = "inframe_insertion|inframe_deletion|missense_variant|protein_altering_variant"
    vep_copy = add_consolidated_consequence(vep_df)
    vep_copy = vep_copy[vep_copy["Consequence_consolidated"].str.contains(moderate_consequences)]
    return vep_copy


# Remove variants with clinvar classification of one of the benigns
def subset_to_non_benign(vep_df, clinsig_col=None, clinvar_benign=None):
    if clinsig_col is None:
        clinsig_col = "ClinVar_updated_2022Aug_CLNSIG"
    if clinvar_benign is None:
        clinvar_benign = "Benign|Likely_benign|Benign/Likely_benign"
    vep_df_nob = vep_df[~(vep_df[clinsig_col].str.contains(clinvar_benign))].copy()
    return vep_df_nob


# Remove variants with MAF above the predefined frequency (e.g. to keep just rare variants)
def subset_to_low_frequency(vep_df, freq_col="gnomAD_AF", freq=0.01, verbose=False):
    # Convert missing to zero
    vep_df_copy = vep_df.copy()
    vep_df_copy.loc[vep_df["gnomAD_AF"] == "-", "gnomAD_AF"] = 0
    vep_df_copy.loc[vep_df["MAX_AF"] == "-", "MAX_AF"] = 0
    vep_df_copy["gnomAD_AF"] = vep_df_copy["gnomAD_AF"].astype(float)

    vep_df_copy_lf = vep_df_copy[vep_df_copy[freq_col] < freq].copy()

    if verbose:
        removed_df = vep_df_copy[vep_df_copy[freq_col] >= freq]
        logger.info(f"we removed {len(removed_df)} variants")
        logger.info(removed_df)
    return vep_df_copy_lf


# Remove variants with MAF below the predefined frequency (e.g. to keep just common variants)
def subset_to_high_frequency(vep_df, freq_col="gnomAD_AF", freq=0.01, verbose=False):
    # Convert missing to zero
    vep_df_copy = vep_df.copy()
    vep_df_copy.loc[vep_df["gnomAD_AF"] == "-", "gnomAD_AF"] = 0
    vep_df_copy.loc[vep_df["MAX_AF"] == "-", "MAX_AF"] = 0
    vep_df_copy["gnomAD_AF"] = vep_df_copy["gnomAD_AF"].astype(float)

    vep_df_copy_hf = vep_df_copy[vep_df_copy[freq_col] >= freq].copy()

    if verbose:
        removed_df = vep_df_copy[vep_df_copy[freq_col] < freq]
        logger.info(f"we removed {len(removed_df)} variants")
        logger.info(removed_df)
    return vep_df_copy_hf


# Subset to variants defined as pathogenic in clinvar
def subset_to_clinvar_pathogenic(vep_df, clinsig_col=None, clinvar_pathogenic=None, conflicting_col=None):
    if clinsig_col is None:
        clinsig_col = "ClinVar_updated_2022Aug_CLNSIG"
    if clinvar_pathogenic is None:
        clinvar_pathogenic = "Pathogenic|Likely_pathogenic|_risk_factor|risk_factor|Pathogenic/Likely_pathogenic"
    vep_df_patho = vep_df[vep_df[clinsig_col].str.contains(clinvar_pathogenic)]
    vep_df_patho_no_conflict = subset_to_clinvar_conflicting(vep_df_patho, clinsig_col, conflicting_col, invert=True)

    return vep_df_patho_no_conflict


# Subset to variants with conflicitng clinvar classifications
def subset_to_clinvar_conflicting(vep_df, clinsig_col=None, conflicting_col=None, invert=False):
    if clinsig_col is None:
        clinsig_col = "ClinVar_updated_2022Aug_CLNSIG"
    if conflicting_col is None:
        conflicting_col = "Conflicting_interpretations_of_pathogenicity"
    if not invert:
        vep_conflicting = vep_df[vep_df[clinsig_col].str.contains(conflicting_col)]
    else:
        vep_conflicting = vep_df[~vep_df[clinsig_col].str.contains(conflicting_col)]
    return vep_conflicting


# For variants with conflicting clinvar evidences, create a table where each column
# has the number of submissions relating to that classification
def make_confliciting_evidence_table(
    vep_df,
    id_col="Uploaded_variation",
    clinsig_col=None,
    conflicting_col=None,
    clin_conflict=None,
    conseq_col=None,
):
    if clinsig_col is None:
        clinsig_col = "ClinVar_updated_2022Aug_CLNSIG"
    if clin_conflict is None:
        clin_conflict = "ClinVar_updated_2022Aug_CLNSIGCONF"
    if conflicting_col is None:
        conflicting_col = "Conflicting_interpretations_of_pathogenicity"
    if conseq_col is None:
        conseq_col = "Consequence_consolidated"

    vep_df_no_dup = vep_df.drop_duplicates(subset=[id_col])

    ## get conflicting ones
    conflicting = subset_to_clinvar_conflicting(vep_df_no_dup, clinsig_col=clinsig_col, conflicting_col=conflicting_col)
    conflicting = conflicting[[id_col, clin_conflict]]
    conflicting = conflicting.set_index(id_col)

    ## Parse the lines of evidence for pathogenicity
    ## This line may change between VEP version depending on delimiter style
    conflicting_expanded = conflicting[clin_conflict].str.split("|", expand=True)

    ## get long format
    conflicting_expanded = conflicting_expanded.stack().reset_index()
    conflicting_expanded = conflicting_expanded.drop(["level_1"], axis=1)
    conflicting_expanded = conflicting_expanded.rename(columns={0: "clinvar_annotation"})

    ## correct classifications names that start with a "_"
    conflicting_expanded["clinvar_annotation"] = conflicting_expanded["clinvar_annotation"].apply(
        lambda x: x[1:] if x[0] == "_" else x
    )

    conflicting_expanded["count"] = conflicting_expanded["clinvar_annotation"].str.split("(").str[1]
    conflicting_expanded["count"] = conflicting_expanded["count"].str.split(")").str[0]
    conflicting_expanded["clinvar_annotation"] = conflicting_expanded["clinvar_annotation"].str.split("(").str[0]

    ## pivot table to get clinvar terms as col names
    conflicting_expanded_transformed = conflicting_expanded.pivot(columns="clinvar_annotation", values="count")
    conflicting_expanded_transformed[id_col] = conflicting_expanded[id_col]
    conflicting_expanded_transformed = conflicting_expanded_transformed.set_index(id_col)
    conflicting_expanded_transformed = conflicting_expanded_transformed.rename_axis(None, axis=1).reset_index()

    ## combine rows and their values for each position
    clinvar_classifications = [
        "Benign",
        "Likely_benign",
        "Likely_pathogenic",
        "Pathogenic",
        "Uncertain_significance",
    ]
    clinvar_count_operation = {var_type: "sum" for var_type in clinvar_classifications}

    for var_type in clinvar_classifications:
        if var_type in conflicting_expanded_transformed.columns:
            conflicting_expanded_transformed[var_type] = (
                conflicting_expanded_transformed[var_type].fillna(0).astype(int)
            )
        else:
            conflicting_expanded_transformed[var_type] = 0

    conflicting_expanded_transformed = conflicting_expanded_transformed.groupby([id_col], as_index=False).agg(
        clinvar_count_operation
    )

    ## get consolidated cols
    conflicting_expanded_transformed["Benign_consolidated"] = (
        conflicting_expanded_transformed["Benign"] + conflicting_expanded_transformed["Likely_benign"]
    )
    conflicting_expanded_transformed["Pathogenic_consolidated"] = (
        conflicting_expanded_transformed["Pathogenic"] + conflicting_expanded_transformed["Likely_pathogenic"]
    )

    ## Add variant consequences
    vep_df_no_dup = add_consolidated_consequence(vep_df_no_dup)
    conseq = vep_df_no_dup[[id_col, conseq_col]]
    conflicting_expanded_transformed_conseq = pd.merge(conflicting_expanded_transformed, conseq, on=id_col, how="inner")
    return conflicting_expanded_transformed_conseq


# A function that decides on whether to include the conflicitng variant
# This function is very project dependent. Make your own variant filter criteria as it fits for your project
# The requirement for filter_criteria: takes in a dataframe row and returns "exclude" or "include"
def variant_filter_criteria(row):
    patho_consolidated = row["Pathogenic_consolidated"]
    benign_consolidated = row["Benign_consolidated"]
    uncertain_consolidated = row["Uncertain_significance"]

    if patho_consolidated == 0:
        return "exclude"
    elif patho_consolidated < benign_consolidated:
        return "exclude"
    elif patho_consolidated < uncertain_consolidated:
        return "exclude"
    else:
        return "include"


# Make a decision whether to include or exclude the conflicting variants based on the lines of evidences
# The filter_critera is a user-defined function that decides on whether the variant should be included or excluded
def make_exlcusion_decision(evidence_table, filter_criteria=variant_filter_criteria):
    evidence_table_copy = evidence_table.copy()
    evidence_table_copy["decision"] = evidence_table_copy.apply(filter_criteria, axis=1)
    return evidence_table_copy


def remove_variants_too_common_in_dataset(genotype_vcf, proportion_threshold=0.05):
    """
    Inputs:
        - genotype_vcf: DF
            - rows = variants, columns = samples
            - expected to be binarized already
            - expected to have just the genotype columns
        - proportion_threshold: this is a proportion threshold.
            Told by Seunghun, Saud, and Hoyin that they typically use 5% as the threshold, so this is the default.

    """
    assert utils.is_binarized(genotype_vcf), "Expected binarized VCF, but got something with >2 values."
    logger.info(
        f"Removing variants that occur in higher proportion than {proportion_threshold} of our dataset's samples - these are likely artifacts."
    )
    logger.debug("working with a copy of the input DF so we don't change the original")
    vcf = genotype_vcf.copy()
    N = vcf.shape[1]
    N_vars = vcf.shape[0]
    n_samples_per_variant = vcf.sum(axis=1)
    vcf["n_samples_with_variant"] = n_samples_per_variant
    vcf["proportion_of_samples_with_variant"] = n_samples_per_variant / N
    vcf = utils.relocate(vcf, ["n_samples_with_variant", "proportion_of_samples_with_variant"])

    remove_df = (
        vcf[vcf["proportion_of_samples_with_variant"] >= proportion_threshold]
        .sort_values(by=["n_samples_with_variant"], ascending=False)
        .copy()
    )
    logger.debug(f"here are the {len(remove_df)} variants that we filter out")
    logger.debug(remove_df)

    logger.debug("only keeping the variants under the threshold")
    vcf = (
        vcf[vcf["proportion_of_samples_with_variant"] < proportion_threshold]
        .sort_values(by=["n_samples_with_variant"], ascending=False)
        .copy()
    )

    logger.info(
        f"returning the filtered DF (kept {vcf.shape[0]}/{N_vars}) and the DF of variants we removed (removed {remove_df.shape[0]}/{N_vars})"
    )
    return vcf, remove_df


def remove_vars_too_common_in_dataset_from_annotated_vcf(
    annot_vcf, proportion_threshold=0.05, id_col="Uploaded_variation"
):
    annot_vcf.set_index(id_col, inplace=True)
    genotype_only_vcf = utils.restrict_vcf_to_genotype_columns(annot_vcf)
    logger.debug(f"genotype_only_vcf.shape: {genotype_only_vcf.shape}")
    binarized_genotype_only_vcf = utils.binarize_df(genotype_only_vcf)
    filtered_binarized_genotype_only_vcf, _ = remove_variants_too_common_in_dataset(
        binarized_genotype_only_vcf, proportion_threshold=proportion_threshold
    )
    annot_vcf = annot_vcf.loc[filtered_binarized_genotype_only_vcf.index.tolist(), :]
    annot_vcf.reset_index(inplace=True)
    return annot_vcf


# Custom function to merge rows while prioritizing non-null values
def merge_near_duplicate_rows_in_vcf_helper(group):
    result = group.iloc[0].copy()
    if group.shape[0] == 1:  # nothing to check because only 1 row
        return result
    for col in group.columns[1:]:
        # Check if any non-NaN, non-"./." value exists in the group
        non_nan_values = group[col][~group[col].isin([np.nan, "./."])]
        if not non_nan_values.empty:
            # If yes, take the first non-NaN, non-"./." value
            result[col] = non_nan_values.iloc[0]
    return result


# Group by the ID column and apply the custom merging function
def merge_near_duplicate_rows_in_vcf(df, id_col):
    df_merged = df.groupby(id_col).apply(merge_near_duplicate_rows_in_vcf_helper).reset_index(drop=True)
    return df_merged


def variant_quality_filter(vcf_df, min_dp=10, min_gq=20, min_vaf=0.25, failed_qc_fill="./."):
    """
    TODO: for later processing steps to work, do I have to also change all the other sample information columns to NaN when I update the GT column?

    # Code example to test this function. We make a sample VCF DataFrame with genotype quality, variant allele frequency, and read depth columns for two samples
    data = {
        'POS': [100, 200, 300],
        'Sample1.GT': ['0/1', '1/1', '0/0'],
        'Sample1.GQ': [30, 10, 40],
        'Sample1.VAF': [0.3, 0.2, 0.5],
        'Sample1.DP': [15, 20, 8],
        'Sample2.GT': ['0/0', '0/1', '1/1'],
        'Sample2.GQ': [25, 35, 15],
        'Sample2.VAF': [0.2, 0.3, 0.6],
        'Sample2.DP': [12, 18, 25]
    }

    vcf_df = pd.DataFrame(data)
    display(vcf_df)
    variant_quality_filter(vcf_df)
    """

    logger.info(f"Performing variant quality filtration on VCF of shape {vcf_df.shape}")
    logger.debug("Get all sample names")
    sample_names = utils.get_sample_names_from_VCF(vcf_df)

    logger.debug("Filter samples based on thresholds; if any one fails, that sample/variant pair fail QC")
    for sample in sample_names:
        filter_condition = (
            (vcf_df[f"{sample}.GQ"] < min_gq) | (vcf_df[f"{sample}.VAF"] < min_vaf) | (vcf_df[f"{sample}.DP"] < min_dp)
        )
        vcf_df.loc[filter_condition, f"{sample}.GT"] = failed_qc_fill

    logger.debug(
        f"Returning vcf_df (shape {vcf_df.shape}), which now contains updated genotype values (fill value = {str(failed_qc_fill)}) for samples that don't pass the filtering conditions"
    )
    return vcf_df


######
# Defining the variant filtering workflow
######


def variant_selection_workflow(
    vep_df,
    genes_to_subset,
    id_col="Uploaded_variation",
    filter_criteria=variant_filter_criteria,
    clinsig_col="ClinVar_updated_2021Jun_CLNSIG",
    clin_conflict="ClinVar_updated_2021Jun_CLNSIGCONF",
    proportion_threshold=0.05,
):
    # Subset to variants in predefined gene list
    vep_gene_subset = subset_to_gene_list(vep_df, gene_list=genes_to_subset)

    # Set 1 - Variants with severe consequence
    # subset to variants with severe consequence
    vep_truncating = subset_to_severe_consequence(vep_gene_subset)
    # Remove benign high-impact variants
    vep_truncating_nob = subset_to_non_benign(vep_truncating, clinsig_col)
    # Remove high-impact variants with high-frequency
    vep_truncating_nob_lof = subset_to_low_frequency(vep_truncating_nob)
    logger.debug(f"vep_truncating_nob_lof: {vep_truncating_nob_lof.shape}\n{vep_truncating_nob_lof.head()}")

    # Set 2
    # Subset to variants with pathogenic clinvar annotation
    vep_patho = subset_to_clinvar_pathogenic(vep_gene_subset, clinsig_col=clinsig_col, conflicting_col=clin_conflict)
    logger.debug(f"vep_patho: {vep_patho.shape}\n{vep_patho.head()}")

    # Set 3
    # Subset to variants with conflicting clinvar annotation
    vep_conflicting = subset_to_clinvar_conflicting(
        vep_gene_subset, clinsig_col=clinsig_col, conflicting_col=clin_conflict
    )
    logger.debug(f"vep_conflicting: {vep_conflicting}")
    if len(vep_conflicting) > 0:
        # Create a table representing lines of evidence
        vep_conflicting_table = make_confliciting_evidence_table(
            vep_conflicting,
            id_col="Uploaded_variation",
            clinsig_col=clinsig_col,
            clin_conflict=clin_conflict,
        )
        # Makes inclusion-exclusion decision based on lines of evidence
        vep_conflicting_table = make_exlcusion_decision(vep_conflicting_table, filter_criteria=filter_criteria)

        # Keep only conflicting variants meeting inclusion criteria
        vep_conflicting_table_include = vep_conflicting_table[vep_conflicting_table["decision"] == "include"]
        vep_conflicting_table_include_merged = vep_conflicting_table_include[[id_col]].merge(vep_gene_subset, on=id_col)

        # Combine all variants subsetted so far and remove duplicate entries
        vep_all_patho_variants = pd.concat(
            [vep_truncating_nob_lof, vep_patho, vep_conflicting_table_include_merged],
            ignore_index=True,
        )

    else:
        # Combine all variants subsetted so far and remove duplicate entries
        vep_all_patho_variants = pd.concat([vep_truncating_nob_lof, vep_patho], ignore_index=True)

    vep_all_patho_variants = vep_all_patho_variants.drop_duplicates(subset=[id_col])
    vep_all_patho_variants = add_consolidated_consequence(vep_all_patho_variants)

    # Remove variants that appear in too large a proportion of samples (likely artifacts)
    vep_all_patho_variants = remove_vars_too_common_in_dataset_from_annotated_vcf(
        vep_all_patho_variants, proportion_threshold=proportion_threshold, id_col=id_col
    )

    return vep_all_patho_variants
