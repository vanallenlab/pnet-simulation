import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)
##################################
# Extracted from original `get_P1000_results.ipynb` for reuse
##################################


def load_feature_importances(importances_path):
    """
    Load feature importance data from a CSV file.

    Args:
        importances_path (str): Path to the feature importance file.

    Returns:
        pd.DataFrame: DataFrame of feature importances.
    """
    if not os.path.exists(importances_path):
        raise FileNotFoundError(f"File not found: {importances_path}")
    logger.debug(f"Loading feature importances from {importances_path}")
    return pd.read_csv(importances_path).set_index("Unnamed: 0")


def load_response_variable(
    response_path="../../pnet_germline/data/pnet_database/prostate/processed/response_paper.csv",
):
    # Load the response variable DataFrame
    response_df = pd.read_csv(response_path)
    response_df.rename(columns={"id": "Tumor_Sample_Barcode"}, inplace=True)
    response_df.set_index("Tumor_Sample_Barcode", inplace=True)
    return response_df


def process_importances(imps, response_df):
    """
    Process feature importances by joining with response data and calculating differences between sample classes.
    This function computes the mean feature importances for each response class and then calculates the difference between them.
    The result is a Series of feature importance differences.

    Args:
        imps (pd.DataFrame): Feature importance DataFrame.
        response_df (pd.DataFrame): Response variable DataFrame.

    Returns:
        pd.Series: Processed feature importance differences.
    """
    logger.debug(
        f"head of imps.join(response_df).groupby('response').mean(): {imps.join(response_df).groupby('response').mean().head()}"
    )
    logger.debug(
        f"shape of imps.join(response_df).groupby('response').mean(): {imps.join(response_df).groupby('response').mean().shape}"
    )
    return imps.join(response_df).groupby("response").mean().diff(axis=0).iloc[1]


def process_feature_importances(
    df_feature_importance_paths,
    response_df,
    importance_path_column="feature_importances_path",
    group_identifier_column="group_identifier",
    replicate_col="random_seed",
):
    """
    Process feature importance data for multiple runs and group by key
    (e.g., dataset), preserving replicate pairing via replicate_col.

    This function requires replicate_col to be present. Per-run feature importances
    and ranks are indexed by replicate_col to enable paired statistical testing.

    Args
    ----
    df_feature_importance_paths : pd.DataFrame
        DataFrame containing paths to feature importance files along with
        dataset and run metadata (including replicate_col).
    response_df : pd.DataFrame
        Response variable DataFrame indexed by sample identifier.
    importance_path_column : str, optional
        Column name in df_feature_importance_paths containing paths to
        feature importance files.
    group_identifier_column : str, optional
        Column name used to group runs (e.g., dataset identifier).
    replicate_col : str, optional
        Column name identifying replicate runs (e.g., random seed).
        Must be present to preserve pairing across runs.

    Returns
    -------
    df_imps_by_key : dict
        Mapping key -> DataFrame of per-run feature importances
        (index = replicate_col, columns = genes/features).
    df_ranks_by_key : dict
        Mapping key -> DataFrame of per-run feature ranks
        (index = replicate_col, columns = genes/features).
        Ranks are computed independently within each run
        (rank 1 = most important).
    """
    logger.info(
        f"Processing feature importances for {df_feature_importance_paths.shape[0]} runs..."
    )

    # Fail fast: pairing column must exist
    if replicate_col not in df_feature_importance_paths.columns:
        raise ValueError(
            f"replicate_col='{replicate_col}' must be present in df_feature_importance_paths "
            "to preserve pairing across runs."
        )

    imps_by_key = {}
    ranks_by_key = {}

    for _, row in df_feature_importance_paths.iterrows():
        try:
            imps = load_feature_importances(row[importance_path_column])
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        processed_imps = process_importances(imps, response_df)
        ranks = processed_imps.abs().rank(ascending=False)

        replicate = row[replicate_col]
        processed_imps.name = replicate
        ranks.name = replicate

        key = row[group_identifier_column]

        imps_by_key.setdefault(key, []).append(processed_imps)
        ranks_by_key.setdefault(key, []).append(ranks)

    # Build DataFrames (runs × genes)
    df_imps_by_key = {key: pd.DataFrame(imps) for key, imps in imps_by_key.items()}
    df_ranks_by_key = {key: pd.DataFrame(ranks) for key, ranks in ranks_by_key.items()}

    # Validate pairing: no duplicate replicates within a key
    for key, df in df_imps_by_key.items():
        if df.index.duplicated().any():
            raise ValueError(f"Duplicate replicates detected for key={key}.")
        df_imps_by_key[key] = df.sort_index()

    for key, df in df_ranks_by_key.items():
        if df.index.duplicated().any():
            raise ValueError(f"Duplicate replicates detected for key={key}.")
        df_ranks_by_key[key] = df.sort_index()

    return df_imps_by_key, df_ranks_by_key


def extract_top_features_from_df(
    df_per_dataset, top_n=10, keep_smallest_n=True, index_label=None
):
    """
    Extract the top N features by rank for each dataset.

    Args:
        df_per_dataset (dict): Dictionary containing feature-level pd DataFrames for each dataset.
        top_n (int): Number of top features to extract.
        keep_smallest_n (bool): Whether to sort in ascending order (lower rank is better).

    Returns:
        pd.DataFrame: DataFrame containing the top N features for each dataset.
    """
    top_features_df = pd.DataFrame()

    # Iterate over the dictionary to calculate top features
    for dataset, df in df_per_dataset.items():
        # Calculate the mean rank for each feature and select the top N
        top_features = df.mean(axis=0).sort_values(ascending=keep_smallest_n)[:top_n]
        # Add the top features as a column to the DataFrame
        top_features_df[dataset] = top_features.index

    # Set the index of the DataFrame to be 1 through top_n
    top_features_df.index = range(1, top_n + 1)

    if index_label is not None:
        top_features_df.index.name = index_label

    return top_features_df


def build_single_gene_rank_runs_long_df(
    df_ranks_by_key,
    gene_name="BRCA2",
    dataset_col="datasets",
    replicate_col="random_seed",
):
    """
    Input:
    ------
    df_ranks_by_key: dict[key -> DataFrame]
        Each DataFrame is (runs x genes) of per-run ranks
        index: random_seed
        columns: gene names
    gene_name: str
        Gene of interest to extract ranks for.
    dataset_col: str
        Name of the column to use for dataset grouping.
    replicate_col: str
        Name of the column to use for replicates (e.g., random seed / run ID). The basis of the downstream paired tests.
    Returns:
    -------
    pd.DataFrame with columns:
        replicate_col
        dataset_col
        model_col
        gene
        gene_name_group
        rank
    """
    records = []

    for dataset, df_ranks in df_ranks_by_key.items():
        # find columns that start with the gene name (handles grouped names if needed)
        matches = [c for c in df_ranks.columns if str(c).startswith(gene_name)]
        if not matches:
            continue

        for replicate, row in df_ranks[matches].iterrows():
            for col in matches:
                records.append(
                    {
                        dataset_col: dataset,
                        replicate_col: replicate,
                        "gene": gene_name,
                        "gene_name_group": col,
                        "rank": row[col],
                    }
                )

    return pd.DataFrame(records)


def calculate_proportions(somatic_mut, y, gene_list):
    logger.info("Group by class and compute mean proportions")
    # Align y to somatic_mut by index
    y_aligned = y.loc[somatic_mut.index]

    # Subset to genes of interest
    gene_df = somatic_mut[gene_list]

    # Combine with response variable
    combined = gene_df.copy()
    combined["class"] = y_aligned

    # Group by class and compute proportions (mean of binary values)
    proportions = combined.groupby("class").mean().T

    return proportions


def calc_OR_from_frequencies(mu1, mu0):
    """
    Calculate odds ratio (OR) from event frequencies in two groups.

    Args:
        mu1 (float): Event frequency in group 1 (e.g., treatment or case group).
        mu0 (float): Event frequency in group 0 (e.g., control group).

    Returns:
        float: Odds ratio (OR) = (mu1 / (1 - mu1)) / (mu0 / (1 - mu0))
    """
    # Convert to Series if needed
    mu1 = pd.Series(mu1)
    mu0 = pd.Series(mu0)
    # Initialize output with NaN
    OR = pd.Series(index=mu1.index, dtype=float)

    odds1 = mu1 / (1 - mu1)
    odds0 = mu0 / (1 - mu0)
    OR = odds1 / odds0
    return OR


def build_modality_stats(
    modality_dfs: dict[str, pd.DataFrame], y, response_col=None
) -> dict[str, pd.DataFrame]:
    """
    For every modality DataFrame, compute:
        • control (class 0) frequency μ0
        • case    (class 1) frequency μ1
        • odds ratio OR = (μ1 / (1-μ1)) ÷ (μ0 / (1-μ0))

    Returns
    -------
    stats_by_modality : dict
        key   = modality name (e.g. 'somatic_mut')
        value = DataFrame with columns
                ['gene', 'mu0', 'mu1', 'odds_ratio']
    """
    # ── 1. Ensure `y_series` is a simple Series ────────────────────────────
    if isinstance(y, pd.DataFrame):
        if response_col is not None:
            y_series = y[response_col]
        elif y.shape[1] == 1:
            y_series = y.iloc[:, 0]
        else:
            raise ValueError("y has multiple columns; please specify `response_col`.")
    else:
        y_series = y

    stats_by_modality = {}

    for modality, df_mod in modality_dfs.items():
        # all genes in this modality
        genes = df_mod.columns.tolist()

        # get frequencies by class
        props = calculate_proportions(
            df_mod, y_series, genes
        )  # index = gene, cols = {0,1}
        mu0 = props[0]
        mu1 = props[1]

        # Compute OR gene-wise
        OR = calc_OR_from_frequencies(mu1, mu0)

        # Pack into a tidy DataFrame
        stats_by_modality[modality] = pd.DataFrame(
            {
                "gene": props.index,
                "mu0": mu0.values,
                "mu1": mu1.values,
                "odds_ratio": OR.values,
            }
        ).set_index("gene")  # easier look-ups later

    return stats_by_modality


def annotate_top_features(
    top_10_features_by_rank: pd.DataFrame,
    dsets_to_visualize: list[str],
    modality_stats: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    For each dataset group column requested, return a DataFrame
    (rank 1–10) augmented with μ0, μ1, and OR looked up from the
    modality-specific summaries built in step 1.

    Returns
    -------
    dict
        key   = dataset group name
        value = DataFrame with columns
                ['feature', 'gene', 'modality', 'mu0', 'mu1', 'odds_ratio']
                indexed by rank (1–10)
    """
    annotated = {}

    for dset in dsets_to_visualize:
        feats = top_10_features_by_rank[dset].dropna()
        out_rows = []

        for rank, modality in feats.items():
            # Expect everything before first _ is the gene. Everything after the first _ is the modality name.
            try:
                gene, modality = modality.split("_", 1)
            except ValueError:
                raise ValueError(
                    f"modality name '{modality}' not in gene_modality form"
                )

            # Look up stats
            try:
                stats_row = modality_stats[modality].loc[gene]
            except KeyError:
                raise KeyError(f"Stats not found for {gene} in modality {modality}")

            out_rows.append(
                {
                    "modality": modality,
                    "gene": gene,
                    "modality": modality,
                    "mu0": stats_row.mu0,
                    "mu1": stats_row.mu1,
                    "odds_ratio": stats_row.odds_ratio,
                }
            )

        annotated[dset] = pd.DataFrame(out_rows, index=feats.index)

    return annotated


# For a single gene, extract importances and ranks from the processed dicts for each dataset combination
def get_single_gene_info(
    df_imps_by_key,
    df_ranks_by_key,
    gene_name,
    group_identifier_col="group_identifier",
):
    """
    Summarize importance and rank for a single gene across replicate runs.

    For each group key (e.g., dataset x model combination), this function:
      1) computes the mean feature importance across runs,
      2) computes the mean per-run rank across runs, and
      3) re-ranks genes based on their mean per-run rank (lower = better).

    This implements a "rank → average → re-rank" strategy, which rewards genes
    that are consistently ranked highly across replicates rather than genes that
    are strong in only a subset of runs.

    The returned table contains one row per group key (and gene match) and is
    intended for summary reporting and visualization. It is
    NOT suitable for paired statistical testing, as replicate-level information
    is aggregated.

    Parameters
    ----------
    df_imps_by_key : dict
        Mapping from group key to a DataFrame of per-run feature importances
        (index = replicate identifier, columns = gene or feature names).
    df_ranks_by_key : dict
        Mapping from group key to a DataFrame of per-run feature ranks
        (index = replicate identifier, columns = gene or feature names).
        Ranks are assumed to be computed independently within each run
        (rank 1 = most important).
    gene_name : str
        Gene name prefix to extract (e.g., "BRCA2").
    group_identifier_col : str, optional
        Name of the column identifying the group key in the output table.

    Returns
    -------
    pandas.DataFrame
        Summary table with one row per group key and gene match, including:
        - mean importance across runs,
        - absolute mean importance,
        - rank derived from the mean of per-run ranks.
    """

    records = []

    for key, df_imp in df_imps_by_key.items():
        mean_imp = df_imp.mean()
        mean_rank = df_ranks_by_key[key].mean()
        recomputed_mean = mean_rank.rank(ascending=True)  # 1 = best, 2 = next…

        matches = mean_imp[mean_imp.index.str.startswith(gene_name)]

        for match_name in matches.index:
            records.append(
                {
                    group_identifier_col: key,
                    "gene_name": gene_name,
                    "gene_name_group": match_name,
                    "importance": mean_imp[match_name],
                    "absolute_importance": abs(mean_imp[match_name]),
                    "rank": recomputed_mean[match_name],
                }
            )

    return pd.DataFrame(records)
