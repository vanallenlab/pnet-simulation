import logging
import os
import pickle

import numpy as np
import pandas as pd

import wandb

logger = logging.getLogger(__name__)


# Set your W&B API key or configure it in your environment
wandb.login()


def get_summary_metric_from_wandb(entity, project_name, metric, run_group=None):
    """
    Example:
    entity = "millergw"
    project_name = "prostate_met_status"
    run_group = "bdt_stability_experiment_003"
    metric = "test_roc_auc_score"
    get_summary_metric_from_wandb(entity, project_name, metric, run_group=run_group)
    """
    # Initialize the W&B API
    api = wandb.Api()
    # Retrieve runs from the specified project and run group
    runs = api.runs(entity + "/" + project_name, filters={"group": run_group})

    # Print summary data for each run in the group
    metrics = []
    for run in runs:
        summary = run.summary._json_dict
        metrics.append(summary[metric])
    return metrics


def get_sklearn_feature_imps(dir):
    # TODO: read in from W&B?
    # Read gene_imps from a Pickle file (format: len 20 list --> pandas DFs, samples x genes?)
    with open(os.path.join(dir, "gene_imps.pkl"), "rb") as file:
        feature_imps = pickle.load(file)
    return feature_imps


def get_pnet_gene_imps(dir):
    # Read gene_imps from a Pickle file (format: len 20 list --> pandas DFs, samples x genes?)
    with open(os.path.join(dir, "gene_imps.pkl"), "rb") as file:
        gene_imps = pickle.load(file)

    # # Read layerwise_imps from a Pickle file (format: len 20 list --> len 5 list --> pandas DF, samples x features)
    # with open(os.path.join(dir, 'layerwise_imps.pkl'), 'rb') as file:
    #     layerwise_imps = pickle.load(file)

    # print(gene_imps[0].shape)
    # for i in range(5):
    #     print(layerwise_imps[0][i].shape)
    return gene_imps


def make_pnet_gene_ranking_df(gene_imps, index):
    """
    For a given sample (aka index / patient), pull the gene rankings from each of the runs (dfs) in gene_imps.

    Inputs:
    - gene_imps: a list of DFs.
        len(gene_imps) = number of model runs.
        gene_imps.shape = patients x genes.
    """
    logger.debug("Formatting data")
    # Extract the first row from each DataFrame
    first_rows = [df.iloc[index] for df in gene_imps]

    # Create a DataFrame with the first rows
    first_rows_df = pd.DataFrame(first_rows)

    # Create a DataFrame with rankings for each gene in each row
    rankings_df = first_rows_df.apply(lambda row: row.abs().rank(ascending=False), axis=1)
    return rankings_df


def make_patient_gene_imp_df(gene_imps):
    """
    Input:
    - gene_imps: a list of DFs.
        len(gene_imps) = number of model runs.
        gene_imps.shape = patients x genes.
    Output:
    - rankings_df: a list of DFs.
        len(rankings_df) = number of patients.
        rankings_df.shape = number of runs x genes.
        values: feature importances

    # Example
    # Creating a list of DataFrames (representing patients x genes for each run)
    gene_imps = [
        pd.DataFrame(np.random.rand(5, 3), columns=['Gene1', 'Gene2', 'Gene3']),
        pd.DataFrame(np.random.rand(5, 3), columns=['Gene1', 'Gene2', 'Gene3']),
        pd.DataFrame(np.random.rand(5, 3), columns=['Gene1', 'Gene2', 'Gene3'])
    ]

    # Display the original and resulting DataFrames
    print("Original DataFrames:")
    for idx, df in enumerate(gene_imps):
        print(f"Run {idx + 1}:\n{df}\n")

    print("\nConcatenated DataFrame:")
    print(concatenated_df)

    print("\nResulting DataFrames:")
    for idx, df in enumerate(make_gene_imp_df(gene_imps)):
        print(f"Row {idx + 1}:\n{df}\n")
    """
    # Concatenate the DataFrames along axis=1
    concatenated_df = pd.concat(gene_imps, axis=1)

    # Get the number of genes and number of runs
    num_genes = len(gene_imps[0].columns)
    num_runs = len(gene_imps)

    # Reshape each row into a DataFrame with shape (n_runs, genes)
    result_list = [
        pd.DataFrame(np.array(row).reshape(num_runs, num_genes), columns=gene_imps[0].columns)
        for _, row in concatenated_df.iterrows()
    ]
    return result_list


def make_perpatient_rankings_dfs(gene_imps):
    """
    Input:
    - gene_imps: a list of DFs.
        len(gene_imps) = number of model runs.
        gene_imps.shape = patients x genes.
    Output:
    - result_list: a list of DFs.
        len(result_list) = number of patients.
        result_list.shape = number of runs x genes.
        values: feature rank
    """
    dfs = make_patient_gene_imp_df(gene_imps)

    # Create a DataFrame with rankings for each gene in each row
    rankings_dfs = [df.apply(lambda row: row.abs().rank(ascending=False), axis=1) for df in dfs]
    return rankings_dfs


def calc_perpatient_stability_metric(gene_imps, n_top_genes=50):
    # logger.warn("TODO: check the functionality of this function. Is it equivalent to old method? Yes, it is.")
    rankings_dfs = make_perpatient_rankings_dfs(gene_imps)
    logger.info("Calculating patient-level stability metric")
    row_averages = [df.mean(axis=0) for df in rankings_dfs]
    row_stdevs = [df.std(axis=0) for df in rankings_dfs]
    summary_dfs = [
        pd.DataFrame({"Average Across Rows": i, "Std Dev Across Rows": j}) for (i, j) in zip(row_averages, row_stdevs)
    ]
    summary_dfs = [df.sort_values(by="Average Across Rows", ascending=True) for df in summary_dfs]

    filtered_to_imp = [df[df["Average Across Rows"] < n_top_genes] for df in summary_dfs]
    stability_metrics = [df["Std Dev Across Rows"].median() for df in filtered_to_imp]
    return stability_metrics


def calc_perpatient_stability_metric_old(gene_imps, n_top_genes=50):
    logger.info("Calculating patient-level stability metric")
    stabs = []
    for patient_i in range(gene_imps[0].shape[0]):
        rankings_df = make_pnet_gene_ranking_df(gene_imps, patient_i)
        stability_metric = calc_stability_metric_on_runs_by_generank_df(rankings_df, n_top_genes)
        stabs.append(stability_metric)
    return stabs


def calc_stability_metric_on_runs_by_generank_df(rankings_df, n_top_genes=50):
    """
    Input:
    - rankings_df: runs x genes. Values are gene ranks within a given run.
    """
    logger.info("Compute the average and standard deviation across rows (aka runs)")
    average_across_rows = rankings_df.mean(axis=0)
    std_across_rows = rankings_df.std(axis=0)

    # Create a new DataFrame with averages and standard deviations
    summary_df = pd.DataFrame({"Average Across Rows": average_across_rows, "Std Dev Across Rows": std_across_rows})
    summary_df = summary_df.sort_values(by="Average Across Rows", ascending=True)

    filtered_to_imp = summary_df[summary_df["Average Across Rows"] < n_top_genes]
    stability_metric = filtered_to_imp["Std Dev Across Rows"].median()
    return stability_metric


def calc_model_stability(imp_lists, n_top_genes=50):
    """
    Input:
    - imp_lists: a list of pd.Series.
        len(imp_lists) = number of model runs.
        imp_lists[0] = feature importance data for the first model run
            pd.Series of length number features (e.g. 3x num_genes if using mut, amp, and del data)
            values are feature importance
    """
    logger.warn("TODO: check functionality. Is the rankings DF correct?")
    logger.info("Convert to rank and sort")
    rankings_df = pd.DataFrame(imp_lists).apply(lambda row: row.abs().rank(ascending=False), axis=1)
    stability = calc_stability_metric_on_runs_by_generank_df(rankings_df, n_top_genes)
    return stability
