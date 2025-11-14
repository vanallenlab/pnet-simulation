import logging  # noqa: I001
import os

import pandas as pd

from pnet.data_processing import utils, data_manipulation  # for manipulating the germline VCFs, filtering, etc


logger = logging.getLogger(__name__)


##############################
# Prostate project data loading
##############################
def get_somatic_mutation(somatic_mut_f):
    logger.info("Getting somatic mutation data")
    somatic_mut = load_somatic_mut(somatic_mut_f)
    somatic_mut = format_mutation_data(somatic_mut)
    return somatic_mut


def get_germline_mutation(germline_vars_f):
    logger.info("Getting germline mutation data")
    germline_var_df = load_germline_mut(germline_vars_f)
    germline_mut = format_germline_mutation_data(germline_var_df)
    return germline_mut


def get_somatic_amp_and_del(somatic_cnv_f):
    logger.info("Getting somatic amplification and deletion data")
    cnv = load_somatic_cnv(somatic_cnv_f)
    somatic_amp = format_cnv_data(
        cnv,
        data_type="cnv_amp",
        cnv_levels=3,
        cnv_filter_single_event=True,
        mut_binary=False,
        selected_genes=None,
    )
    somatic_del = format_cnv_data(
        cnv,
        data_type="cnv_del",
        cnv_levels=3,
        cnv_filter_single_event=True,
        mut_binary=False,
        selected_genes=None,
    )
    return somatic_amp, somatic_del


def get_additional_data(
    additional_f,
    id_map_f,  # TODO: ideally, wouldn't need two file paths, just one.... but we need a way to harmonize the IDs.
    cols_to_include=[
        "PCA1",
        "PCA2",
        "PCA3",
        "PCA4",
        "PCA5",
        "PCA6",
        "PCA7",
        "PCA8",
        "PCA9",
        "PCA10",
    ],
):
    logger.info("Getting additional data")
    additional_df = load_additional_data(additional_f, id_map_f, cols_to_include=cols_to_include)
    return additional_df


def get_target(id_map_f, sample_metadata_f, id_to_use="Tumor_Sample_Barcode", target_col="is_met"):
    """
    Get a DF of the target variable indexed by a sample ID.
    """
    logger.info("Getting prediction target")  # TODO: alter to work when we aren't using paired samples
    sample_metadata = load_sample_metadata_and_target(id_map_f, sample_metadata_f)
    target = extract_target(sample_metadata, id_to_use=id_to_use, target_col="is_met")
    logger.info(f"Target column value_counts: {target[target_col].value_counts()}")
    return target


def get_indices_from_file(indices_f):
    pass


def load_sample_metadata_and_target(id_map_f, sample_metadata_f):
    logger.info("Loading the sample metadata DF that has all the IDs and also our target, metastatic status ('is_met')")
    sample_metadata = load_sample_metadata_with_all_germline_ids(sample_metadata_f, id_map_f)
    logger.debug(sample_metadata.head())
    logger.debug(sample_metadata.shape)
    return sample_metadata


def extract_target(df, id_to_use="Tumor_Sample_Barcode", target_col="is_met"):
    assert id_to_use in df.columns.tolist(), (
        "The ID you wanted to use isn't in the DF columns"
    )  # e.g. "Tumor_Sample_Barcode", "vcf_germline_ids"
    logger.info(f"Generating the target DF (target column '{target_col}' indexed by '{id_to_use}')")
    target = df.set_index(id_to_use).loc[:, [target_col]]
    logger.debug(target.head())
    logger.debug(len(target))
    logger.debug(f"target value_counts: {target[target_col].value_counts()}")
    return target


def load_somatic_mut(somatic_mut_f):
    logger.info(f"Load somatic mutation data from {somatic_mut_f}")
    somatic_mut = data_manipulation.load_df_verbose(somatic_mut_f)
    somatic_mut.set_index("Tumor_Sample_Barcode", inplace=True)
    return somatic_mut


def load_somatic_cnv(somatic_cnv_f):
    logger.info(f"Load somatic CNV data from {somatic_cnv_f}")
    cnv = data_manipulation.load_df_verbose(somatic_cnv_f)
    cnv.rename(columns={"Unnamed: 0": "Tumor_Sample_Barcode"}, inplace=True)
    cnv.set_index("Tumor_Sample_Barcode", inplace=True)
    return cnv


def load_germline_mut(germline_vars_f):
    logger.info(f"Loading germline mutation data from {germline_vars_f}")
    germline_var_df = pd.read_csv(germline_vars_f, low_memory=False, sep="\t")
    germline_var_df = germline_var_df.set_index("Uploaded_variation")
    logger.info(f"Shape of raw germline mutation data: {germline_var_df.shape}")
    return germline_var_df


def load_additional_data(
    additional_f,
    id_map_f,
    id_to_use="Tumor_Sample_Barcode",  # TODO: ideally, wouldn't need two file paths, just one.... but we need a way to harmonize the IDs.
    cols_to_include=[
        "PCA1",
        "PCA2",
        "PCA3",
        "PCA4",
        "PCA5",
        "PCA6",
        "PCA7",
        "PCA8",
        "PCA9",
        "PCA10",
    ],
):
    logger.info(f"Load additional data from {additional_f}")
    sample_metadata = load_sample_metadata_with_all_germline_ids(additional_f, id_map_f)
    assert id_to_use in sample_metadata.columns.tolist(), (
        f"The ID you wanted to use as index ({id_to_use}) isn't in the DF columns, which are \n{sample_metadata.columns.tolist()}"
    )  # e.g. "Tumor_Sample_Barcode", "vcf_germline_ids"
    additional_data = sample_metadata.set_index(id_to_use).loc[:, cols_to_include]
    return additional_data


def load_germline_metadata(
    metadata_f="../data/prostate/pathogenic_variants_with_clinical_annotation_1341_aug2021_correlation.csv",
):
    logger.info(f"Loading the germline metadata file at {metadata_f}")
    # the clinical metadata file can be indexed by ther germline ID ('sample_original')
    met_col = "Disease_status_saud_edited"
    id_col = "BamID_modified"
    is_met_col = "is_met"
    logger.debug(f"metadata_f: {metadata_f}")
    metadata = pd.read_csv(metadata_f)
    # create binary "is_met_col" column
    metadata[is_met_col] = metadata[met_col].map({"Metastatic": 1, "Primary": 0})
    metadata = metadata.rename(columns={id_col: "germline_id", met_col: "disease_status"})
    logger.debug("Head of the metadata DF:")
    logger.debug(metadata.head())
    return metadata


def load_sample_metadata_with_all_germline_ids(
    sample_metadata_f="../data/prostate/pathogenic_variants_with_clinical_annotation_1341_aug2021_correlation.csv",
    germline_somatic_id_map_f="../data/prostate/germline_somatic_id_map_outer_join.csv",
    sample_metadata_germline_id_col="germline_id",
    germline_id_map_col="sample_metadata_germline_id",
):
    """
    Args:
    - sample_metadata_f: filepath to the sample metadata file
    - germline_somatic_id_map_f: filepath to the germline ID mapping DF
    - sample_metadata_germline_id_col: name of the germline ID column in the sample metadata DF
    - germline_id_map_col: name of the column in the germline ID mapping DF that corresponds to (aka is the same as) the column data from the sample metadata DF
    """
    # Load in the DFs
    germline_somatic_id_map = pd.read_csv(germline_somatic_id_map_f)
    logger.debug(f"sample_metadata_f: {sample_metadata_f}")
    sample_metadata = load_germline_metadata(sample_metadata_f)

    # add more metadata columns to the sample_metadata DF
    sample_metadata = pd.merge(
        germline_somatic_id_map,
        sample_metadata,
        left_on=germline_id_map_col,
        right_on=sample_metadata_germline_id_col,
    )
    # drop the now redundant column
    sample_metadata.drop(sample_metadata_germline_id_col, axis=1, inplace=True)
    sample_metadata.set_index(germline_id_map_col, inplace=True, drop=True)
    return sample_metadata


##############################
# Prostate project data formatting/munging
##############################
def get_genes_in_common(
    *dataframes,
    tcga_gene_list_f="../../../pnet_germline/data/pnet_database/genes/tcga_prostate_expressed_genes_and_cancer_genes.csv",
):  # TODO: test function
    logger.info("Finding overlapping genes across the DFs")
    overlapping_genes = data_manipulation.find_overlapping_columns(*dataframes)

    logger.info(
        "Find the overlap between this and the pre-specified list of TCGA cancer genes and those expressed in the prostate"
    )
    # TODO: looks like there was an excel misshap! MAR1 has become 1-Mar, 10-Sept, etc. But I don't think these genes are covered by our datasets anyway...
    genes = pd.read_csv(tcga_gene_list_f)
    logger.debug(genes.head())
    overlapping_genes = data_manipulation.find_overlapping_elements(set(genes["genes"]), overlapping_genes)

    return overlapping_genes


def restrict_to_genes_in_common(*datasets):
    """
    Filter the columns of the dataframe(s).
    Args:
    - *datasets (Pandas DF): arbitrary number of DFs with format samples x genes/features
    """
    genes_in_common = get_genes_in_common(*datasets)
    restricted_dataframes = data_manipulation.filter_to_specified_columns(genes_in_common, *datasets)
    return restricted_dataframes


def format_mutation_data(mut_df, mut_binary=True):
    """
    Args:
    - mut_binary: True means that we should binarize the DF if it isn't already binarizied; we want a binary mutation DF.
    If binarized already, then should only have <=2 unique values in each column (0/1). If we have more, then we're probably working with a burden matrix, but this should be investigated.
    """
    if mut_binary and not data_manipulation.is_binarized(mut_df):
        # if mut_binary and max(somatic_mut.nunique()) >= 2:
        logger.info(f"Matrix was not binary. There were {max(mut_df.nunique())} unique values; binarizing now")
        mut_df[mut_df > 1.0] = 1.0
    return mut_df


def format_germline_mutation_data(df):
    """
    Args:
    - df: variant x [variant metadata and sample-level calls]  (quasi VCF file format)
    """
    logger.info("Starting process of formatting the germline mutation data.")
    logger.info("Extracting the variant metadata DF")
    variant_metadata = utils.get_variant_metadata_from_VCF(df)

    logger.info("Make the binary variant-level genotypes matrix (variants x samples)")
    binary_genotypes = utils.make_binary_genotype_mat_from_VCF(df)

    logger.info("Make the binary gene-level genotypes matrix (genes x samples)")
    gene_level_genotype_matrix = utils.convert_binary_var_mat_to_gene_level_mat(
        binary_genotypes, variant_metadata, binary_output=True
    )

    logger.info("Transposing to get samples x genes")
    germline_mut = gene_level_genotype_matrix.T
    return germline_mut


def harmonize_prostate_ids(
    datasets_w_germline_ids, datasets_w_somatic_ids, convert_ids_to="somatic"
):  # TODO: how should I handle IDs that can't be converted? Right now, I just replace with NAs and warn. But I think I should drop NA rows.
    """
    Args:
    - germline_datasets: any DFs that use the vcf_germline_ids ID as their index
    - somatic_datasets: any DFs that use the Tumor_Sample_Barcode ID as their index
    - convert_ids_to: if "somatic", then convert the germline_datasets IDs to match the somatic_datasets and analogously for "germline". If None, then don't alter the inputs.

    Returns two lists of DataFrames:
        one with modified DataFrames (either germline or somatic, depending on the value of `convert_ids_to`),
        and one with the original DataFrames (either somatic or germline, respectively).
    """
    if convert_ids_to not in ["somatic", "germline", None]:
        raise ValueError(
            f"The convert_ids_to parameter must be one of 'somatic', 'germline', or None but was input as {str(convert_ids_to)}."
        )

    if convert_ids_to == "somatic":
        logger.info("Converting germline IDs (vcf_germline_id) to somatic IDs (Tumor_Sample_Barcode)")
        altered_germline_datasets = []
        for df in datasets_w_germline_ids:
            df.index = convert_germline_id_to_somatic_id(df.index.tolist())
            df = data_manipulation.drop_na_index_rows(df)
            altered_germline_datasets.append(df)
        return altered_germline_datasets, datasets_w_somatic_ids

    elif convert_ids_to == "germline":
        logger.info("Converting somatic IDs (Tumor_Sample_Barcode) to germline IDs (vcf_germline_id)")
        altered_somatic_datasets = []
        for df in datasets_w_somatic_ids:
            df.index = convert_somatic_id_to_germline_id(df.index.tolist())
            df = data_manipulation.drop_na_index_rows(df)
            altered_somatic_datasets.append(df)
        return datasets_w_germline_ids, altered_somatic_datasets

    elif convert_ids_to is None:
        logger.info("Returning without converting any sample IDs")
        return datasets_w_germline_ids, datasets_w_somatic_ids


def convert_germline_id_to_somatic_id(germlineIDs, GERMLINE_DATADIR="../../../pnet_germline/data/"):
    """
    Convert list of germline IDs (vcf_germline_id) to somatic IDs (Tumor_Sample_Barcode).
    Warn if the converted list has any NAs: this means that a match wasn't found.
    """
    logger.debug("Loading the germline ID and somatic-germline ID mapping DF")
    germline_somatic_id_map_f = os.path.join(GERMLINE_DATADIR, "prostate/germline_somatic_id_map_outer_join.csv")
    germline_somatic_id_map = data_manipulation.load_df_verbose(germline_somatic_id_map_f)
    logger.debug(
        f"Converting list of {len(germlineIDs)} germline IDs (vcf_germline_id) to somatic IDs (Tumor_Sample_Barcode)."
    )
    somaticIDs = data_manipulation.convert_values(
        input_value=germlineIDs,
        source=germline_somatic_id_map.vcf_germline_id.tolist(),
        target=germline_somatic_id_map.Tumor_Sample_Barcode.tolist(),
    )
    logger.debug(f"first 5 germline IDs: {germlineIDs[:5]}")
    logger.debug(f"first 5 somatic IDs: {somaticIDs[:5]}")
    return somaticIDs


def convert_somatic_id_to_germline_id(somaticIDs, GERMLINE_DATADIR="../../../pnet_germline/data/"):
    """
    Convert list of somatic IDs (Tumor_Sample_Barcode) to germline IDs (vcf_germline_id).
    Warn if the converted list has any NAs: this means that a match wasn't found.
    """
    logger.debug("Loading the germline ID and somatic-germline ID mapping DF")
    germline_somatic_id_map_f = os.path.join(GERMLINE_DATADIR, "prostate/germline_somatic_id_map_outer_join.csv")
    germline_somatic_id_map = data_manipulation.load_df_verbose(germline_somatic_id_map_f)
    logger.info("Converting list of somatic IDs (Tumor_Sample_Barcode) to germline IDs (vcf_germline_id).")
    germlineIDs = data_manipulation.convert_values(
        input_value=somaticIDs,
        source=germline_somatic_id_map.Tumor_Sample_Barcode.tolist(),
        target=germline_somatic_id_map.vcf_germline_id.tolist(),
    )
    return germlineIDs


def zero_impute_somatic_datasets(germline_datasets, somatic_datasets, zero_impute_somatic=False):
    if zero_impute_somatic is True:
        logger.info(f"Starting process of zero-imputing the {len(somatic_datasets)} somatic dataset(s)")
        germline_features = set(data_manipulation.find_overlapping_columns(*germline_datasets))
        imputed_dataframes = []
        for df in somatic_datasets:
            features_only_in_germline = germline_features - set(df.columns)
            imputed_df = data_manipulation.impute_cols_with_a_constant(df, features_only_in_germline, fill=0)
            imputed_dataframes.append(imputed_df)
        return imputed_dataframes
    return somatic_datasets


def zero_impute_germline_datasets(
    germline_datasets, somatic_datasets, zero_impute_germline=True
):  # TODO: this is basically the same as the somatic version of the function; could make a more general single function
    """
    Return: returns zero-imputed germline datasets if the zero_imput_germline flag is True. Otherwise, returns the unaltered datasets.

    Example of germline zero-imputation:
    For genes that are in the somatic dataset but not germline, add colum of zeros to the germline dataset.
    This way, we will keep all of the somatic data when we run P-NET
    (not subset down to whatever germline gene subset we're using).
    """
    if zero_impute_germline is True:
        logger.info(f"Starting process of zero-imputing the {len(germline_datasets)} germline datasets")
        somatic_features = set(data_manipulation.find_overlapping_columns(*somatic_datasets))
        imputed_dataframes = []
        for df in germline_datasets:
            features_only_in_somatic = somatic_features - set(df.columns)
            imputed_df = data_manipulation.impute_cols_with_a_constant(df, features_only_in_somatic, fill=0)
            imputed_dataframes.append(imputed_df)
        return imputed_dataframes
    return germline_datasets


# pulled and modified from the P-NET paper code, data_reader.py script > load_data_type function.
def format_cnv_data(
    x,
    data_type="cnv",
    cnv_levels=5,
    cnv_filter_single_event=True,
    mut_binary=False,
    selected_genes=None,
):
    logger.info(f"formatting {data_type}")
    x = x.copy()

    if data_type == "cnv":
        if cnv_levels == 3:
            logger.info("cnv_levels = 3")
            # remove single amplification and single delteion, they are usually noisey
            if cnv_levels == 3:
                if cnv_filter_single_event:
                    x[x == -1.0] = 0.0
                    x[x == -2.0] = 1.0
                    x[x == 1.0] = 0.0
                    x[x == 2.0] = 1.0
                else:
                    x[x < 0.0] = -1.0
                    x[x > 0.0] = 1.0

    if data_type == "cnv_del":
        x[x >= 0.0] = 0.0
        if cnv_levels == 3:
            if cnv_filter_single_event:
                x[x == -1.0] = 0.0
                x[x == -2.0] = 1.0
            else:
                x[x < 0.0] = 1.0
        else:  # cnv == 5 , use everything
            x[x == -1.0] = 0.5
            x[x == -2.0] = 1.0

    if data_type == "cnv_amp":
        x[x <= 0.0] = 0.0
        if cnv_levels == 3:
            if cnv_filter_single_event:
                x[x == 1.0] = 0.0
                x[x == 2.0] = 1.0
            else:
                x[x > 0.0] = 1.0
        else:  # cnv == 5 , use everything
            x[x == 1.0] = 0.5
            x[x == 2.0] = 1.0

    if data_type == "cnv_single_del":
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == -1.0] = 1.0
        x[x != -1.0] = 0.0
    if data_type == "cnv_single_amp":
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == 1.0] = 1.0
        x[x != 1.0] = 0.0
    if data_type == "cnv_high_amp":
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == 2.0] = 1.0
        x[x != 2.0] = 0.0
    if data_type == "cnv_deep_del":
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == -2.0] = 1.0
        x[x != -2.0] = 0.0
    return x
