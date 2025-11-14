"""
Utility functions related to the processing and analysis of germline data.
Specifically, working with germline VCFs for the prostate cancer dataset.

Author: Gwen Miller <gwen_miller@g.harvard.edu>
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Logging setup
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


##############################
# General data loading and munging
##############################


def find_mapping(list_a, list_b, reverse_dict: bool = False):
    """
    To find the mapping between two lists where each element in list A is a strict substring of one of the elements in list B, you can use this function.
    # Example usage
    list_a = ['apple', 'banana', 'cat']
    list_b = ['apple pie', 'banana bread', 'black cat']

    result = find_mapping(list_a, list_b)
    print(result)

                 Mapped Value
    apple        apple pie
    banana       banana bread
    cat          black cat

    """
    mapping = {}
    for a in list_a:
        for b in list_b:
            if a in b:
                mapping[a] = b
                break
    logging.debug(f"Found matches for a total of {len(mapping)} out of {len(list_a)} items.")
    if reverse_dict is True:
        logging.info("Reversing the dict so the superstrings are the keys and the substrings are the values")
        logging.debug(
            f"len(set(mapping.values())) == len(mapping.values()): {len(set(mapping.values())) == len(mapping.values())}"
        )
        reversed_dict = {value: key for key, value in mapping.items()}
        mapping = reversed_dict
    return mapping


def filename(f: str) -> str:
    """Return the basename without extension for a filepath."""
    return os.path.splitext(os.path.basename(f))[0]


def savefig(save_path: str, png: bool = True, svg: bool = True) -> None:
    """Save the current matplotlib figure to disk, creating directories if needed."""
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    logging.info(f"saving plot to {save_path}")
    if png:
        plt.savefig(save_path, bbox_inches="tight")
    if svg:
        plt.savefig(f"{save_path}.svg", format="svg", bbox_inches="tight")


def relocate(df: pd.DataFrame, cols):
    """Move the specified columns to the front (preserving order)."""
    new_var_order = cols + df.columns.drop(cols).tolist()
    return df[new_var_order]


##############################
# Loading prostate datasets
##############################


def read_gene_list_from_csv(list_f: str):
    """Read a CSV whose header row contains gene names and return a stripped list of those names."""
    return pd.read_csv(list_f, comment="#").columns.str.strip().tolist()


def load_germline_metadata(
    metadata_f: str = "../data/prostate/pathogenic_variants_with_clinical_annotation_1341_aug2021_correlation.csv",
):
    """
    Load germline clinical metadata and add a binary 'is_met' column (Metastatic=1, Primary=0).
    Note that the clinical metadata file can be indexed by ther germline ID ('sample_original')
    Renames: BamID_modified -> germline_id, Disease_status_saud_edited -> disease_status
    """
    met_col = "Disease_status_saud_edited"
    id_col = "BamID_modified"
    terra_control_id = "sample_id"
    is_met_col = "is_met"

    logging.debug(f"metadata_f: {metadata_f}")
    metadata = pd.read_csv(metadata_f)
    # metadata = pd.read_csv(metadata_f, usecols = [id_col, met_col, terra_control_id])
    # create binary "is_met_col" column
    metadata[is_met_col] = metadata[met_col].map({"Metastatic": 1, "Primary": 0})
    metadata = metadata.rename(columns={id_col: "germline_id", met_col: "disease_status"})
    logging.debug("Head of the metadata DF:")
    logging.debug(metadata.head())
    return metadata


##############################
## working with VCFs
##############################


def is_binarized(df: pd.DataFrame) -> bool:
    """Return True if all values are 0.0/1.0 (or 0/1 when cast to float)."""
    return np.all((df.values == 0.0) | (df.values == 1.0))


def binarize(value, set_as_zero: str = "./.") -> int:
    """
    Map VCF-like genotype entries to 0/1.
    Any value equal to `set_as_zero` becomes 0, everything else becomes 1.
    """
    return 0 if value == set_as_zero else 1


def binarize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply `binarize` element-wise across a DataFrame."""
    return df.applymap(binarize)


def get_sample_names_from_VCF(df: pd.DataFrame):
    """
    Extract sample IDs from VCF columns that end with '.GT'.
    Example: 'SAMPLE123.GT' -> 'SAMPLE123'
    """
    logging.info("Extracting the sample IDs from the column names")
    sample_ids = [col.split(".GT")[0] for col in df.columns if col.endswith(".GT")]
    logging.debug("We found {} sample_ids".format(len(sample_ids)))
    return sample_ids


def get_sample_cols_from_VCF(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict a VCF-like DataFrame to genotype columns ('.GT' suffix) and drop the suffix from column names.
    Rows are preserved.
    """
    logging.debug("restricting to just the genotype columns (the ones that end in .GT)...")
    df = df.filter(regex=r"\.GT$")
    df.columns = df.columns.map(lambda x: x[:-3] if x.endswith(".GT") else x)
    logging.info(f"vcf shape: {df.shape}")
    return df


def restrict_vcf_to_genotype_columns(vcf: pd.DataFrame) -> pd.DataFrame:
    """Like `get_sample_cols_from_VCF`, but leaves index/rows untouched and only trims column names."""
    logging.debug("restricting to just the genotype columns (the ones that end in .GT)")
    vcf = vcf.filter(regex=r"\.GT$")
    logging.debug("renaming columns by dropping the .GT suffix")
    vcf.columns = vcf.columns.map(lambda x: x[:-3] if x.endswith(".GT") else x)
    return vcf


def make_binary_genotype_mat_from_VCF(df: pd.DataFrame) -> pd.DataFrame:
    df = restrict_vcf_to_genotype_columns(df)
    df = df.applymap(binarize)
    logging.debug("Head of the binary genotype matrix:")
    logging.debug(df.head())
    return df


def convert_binary_var_mat_to_gene_level_mat(binary_genotypes, variant_metadata, binary_output=False):
    """
    Description of process to go from a variant-level genotypes matrix to a (binary) gene-level genotypes matrix.
    1. Use list of vars included in the variant level matrix to filter the variant metadata df.
    2. Group the filtered variant metadata df by gene.
    - For each gene, get associated variants.
    - Filter the variant level matrix by these rows.
    - Sum across the rows to get a per-sample count of the number of variants in this gene.
    - Set this as a row in the `gene_burden_matrix`
    3. Create the final output matrix by concatenating each row of gene information (gene_burden_mat): n_genes by n_samples.
    4. If binary_output is True, then binarize the output.
    """
    logger.info("1. Use list of vars included in the variant level matrix to filter the variant metadata df.")
    vars_to_use = binary_genotypes.index.tolist()
    logger.debug(len(vars_to_use))
    variant_metadata = variant_metadata.loc[vars_to_use, :]
    logger.debug(f"filtered variant_metdata.shape: {variant_metadata.shape}")

    logger.info("2. Group the filtered variant metadata df by gene, get gene-level counts")
    logger.info("creating one row per gene with the variant counts for each sample")

    gene_burden_rows = []
    genes = []
    for gene in set(variant_metadata.SYMBOL.tolist()):
        curr_var_data = variant_metadata[gene == variant_metadata.SYMBOL]
        curr_vars = curr_var_data.index.tolist()
        logger.debug(f"for gene {gene} we have {len(curr_vars)} variants: {curr_vars}")
        logger.debug(f"binary_genotypes.loc[curr_vars,:]: {binary_genotypes.loc[curr_vars, :].shape}")
        curr_gene_level_info = binary_genotypes.loc[curr_vars, :].sum(axis=0)
        gene_burden_rows.append(curr_gene_level_info)
        genes.append(gene)

    logger.info(
        "3. Create the final output matrix by concatenating each row of gene information (gene_burden_mat): n_genes by n_samples."
    )
    gene_burden_mat = pd.DataFrame(gene_burden_rows, index=genes)

    if binary_output:
        logger.info("4. Binarizing the gene burden matrix (anything !=0. gets set to 1)")
        gene_burden_mat = gene_burden_mat.applymap(binarize_burden_mat)
        assert not (gene_burden_mat > 1).any().any(), (
            "At least one value in the supposedly binarized gene_burden_mat is greater than 1"
        )

    return gene_burden_mat


def get_variant_metadata_from_VCF(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of non-sample (annotation/metadata) columns from an annotated VCF.
    We identify sample columns by common VCF per-sample suffixes and drop those.
    """
    logging.info("grabbing the columns with variant metadata and information...")
    sample_suffixes = (".AD", ".DP", ".GQ", ".GT", ".VAF", ".PL")
    variant_annotation_cols = [c for c in df.columns if not c.endswith(sample_suffixes)]
    variant_metadata = df.loc[:, variant_annotation_cols]
    logging.debug("Head of the variant metadata DF:")
    logging.debug(variant_metadata.head())
    return variant_metadata


def filter_VCF_chunk(df, gene_list):
    assert "SYMBOL" in df.columns.tolist(), f"SYMBOL isn't in the columns, which are \n{df.columns.tolist()}"
    logging.debug("Determine which rows contain variants in genes of interest")
    filtered_df = df[df["SYMBOL"].isin(gene_list)]
    if len(filtered_df) > 0:
        logging.debug(
            f"df.SYMBOL.value_counts().index.isin(gene_list): {df.SYMBOL.value_counts().index.isin(gene_list)}"
        )
        # assert sum(df.SYMBOL.value_counts().index.isin(gene_list)) < 2, "we were only getting filtered DFs from 1 gene, but at this stage we have more than 1 of our target genes" # TODO: uncomment
    return filtered_df


def filter_annotated_vcf_by_gene_list_chunking(
    annot_vcf_f: str,
    gene_list,
    save_filtered_df_path,
    chunksize: int = 20000,
):
    """
    Filter a (possibly large) annotated VCF file to keep (1) only rows (variants) whose SYMBOL is in `gene_list` and (2) only the genotypes information for each sample (the .GT column)..
    This reads in row-chunks for efficiency and returns the concatenated filtered DataFrame.
    If `save_filtered_df_path` is provided, the result is also written to disk (tab-separated).
    """
    assert annot_vcf_f.endswith(".txt") or annot_vcf_f.endswith(".txt.gz"), (
        "Require file that ends with .txt or .txt.gz"
    )

    logging.debug(f"gene_list: {gene_list}")
    list_of_dfs = []
    with pd.read_csv(annot_vcf_f, chunksize=chunksize, sep="\t", low_memory=False) as reader:
        for i, chunk in tqdm(enumerate(reader), desc="Filtering VCF by gene list"):
            logging.debug(f"working on filtering chunk {i} with shape {chunk.shape}:")
            filtered_df = filter_VCF_chunk(chunk, gene_list)
            if not filtered_df.empty:
                logging.debug(f"In this chunk, filtered down to {len(filtered_df)} rows.")
                list_of_dfs.append(filtered_df)

    if len(list_of_dfs) == 0:
        msg = f"none of the genes were found in the VCF DF: {gene_list}"
        logging.warning(msg)
        return msg

    df = pd.concat(list_of_dfs, ignore_index=True)
    logging.debug(f"Restricted to the genes of interest, we have shape {df.shape}")

    if save_filtered_df_path:
        logging.info(f"Saving the filtered DF to {save_filtered_df_path}")
        out_dir = os.path.dirname(save_filtered_df_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(save_filtered_df_path, index=False, sep="\t")

    return df


def n_variants_per_sample_from_vcf(
    vcf: pd.DataFrame,
    savefig_f=False,
    plot_title: str = "# variants per sample",
    plot_id=None,
    plot_xlabel: str = "# variants/sample",
    ax=None,
):
    """
    Plot a histogram of the number of variants per sample (expects variants x samples matrix with 0/1 entries).
    Returns the matplotlib Axes (or Figure if no Axes were provided).
    """
    logging.info(f"working with a VCF with {vcf.shape[1]} samples and {vcf.shape[0]} rows (variants, genes, etc)")
    n_variants_per_sample = vcf.sum(axis=0)

    # Figure/Axes handling
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(n_variants_per_sample.tolist())
    if plot_id is not None and fig is not None:
        fig.suptitle(plot_id)
    ax.set_title(plot_title)
    ax.set_xlabel(plot_xlabel)

    if savefig_f and fig is not None:
        savefig(savefig_f)

    return ax if ax is not None else fig


def n_samples_per_variant_from_vcf(
    vcf: pd.DataFrame,
    savefig_f=False,
    plot_title: str = "# samples per variant (dataset MAF=0.01 in red)",
    plot_id=None,
    plot_xlabel: str = "log2(# samples/variant)",
    ax=None,
    logscale: bool = True,
):
    """
    Plot a histogram of the number of samples per variant (expects variants x samples matrix with 0/1 entries).
    Drops zero-count variants (with a warning). By default, plots log2 counts and draws a line at MAF=0.01.
    Returns the matplotlib Axes (or Figure if no Axes were provided).
    """
    logging.info(f"working with a VCF with {vcf.shape[1]} samples and {vcf.shape[0]} rows (variants, genes, etc)")
    n_samples_per_variant = vcf.sum(axis=1)

    if (n_samples_per_variant == 0).any():
        logging.warning(
            "Some variants have zero sample counts; dropping them. "
            "This could reflect an issue with VCF filtering or processing."
        )
        n_samples_per_variant = n_samples_per_variant[n_samples_per_variant > 0]

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    maf_1percent = vcf.shape[1] * 0.01
    if logscale:
        ax.hist(np.log2(n_samples_per_variant.tolist()))
        ax.axvline(x=np.log2(maf_1percent), color="red")
    else:
        ax.hist(n_samples_per_variant.tolist())
        ax.axvline(x=maf_1percent, color="red")
        plot_xlabel = "# samples/variant"

    if plot_id is not None and fig is not None:
        fig.suptitle(plot_id)

    ax.set_title(plot_title)
    ax.set_xlabel(plot_xlabel)

    if savefig_f and fig is not None:
        savefig(savefig_f)

    return ax if ax is not None else fig
