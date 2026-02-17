import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# -------------------------
# Core stats utilities
# -------------------------
def fdr_bh(pvals):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : array-like
        1D p-values. NaNs are preserved.

    Returns
    -------
    qvals : np.ndarray
        BH-FDR adjusted p-values (q-values), same shape as input.
        NaNs remain NaN.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)

    mask = np.isfinite(p)
    if mask.any():
        q[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return q


def wilcoxon_signed_rank_test(diffs, alternative="two-sided"):
    """
    Perform a Wilcoxon signed-rank test on paired differences.

    Parameters
    ----------
    diffs : array-like
        1D array of paired differences (d_i), one per technical replicate
        (e.g., per random seed). NaNs are ignored.

    alternative : {"two-sided", "greater", "less"}, optional (default="two-sided")
        Defines the alternative hypothesis. "two-sided" tests whether the
        median difference is non-zero.

    Returns
    -------
    stat : float
        Wilcoxon test statistic.

    p_value : float
        P-value for the specified alternative.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]

    if len(diffs) == 0:
        return np.nan, np.nan

    # Wilcoxon cannot handle all-zero differences
    if np.allclose(diffs, 0):
        return 0.0, 1.0

    stat, p_value = wilcoxon(
        diffs,
        alternative=alternative,
        zero_method="wilcox",  # drop zero differences
        correction=False,  # exact for small n
        method="auto",
    )
    return float(stat), float(p_value)


def paired_signflip_permutation_test(diffs, alternative="greater"):
    """
    Exact paired sign-flip permutation test on the mean of paired differences.

    diffs: array-like of d_i = metric(P-NET, seed_i) - metric(baseline, seed_i)
    alternative: "greater" (1-sided), "less", or "two-sided"

    Returns
    -------
    dict with:
      - n
      - mean_delta (t_obs)
      - p_value
      - null_stats (np.array, length 2^n)
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return dict(n=0, mean_delta=np.nan, p_value=np.nan, null_stats=np.array([]))

    t_obs = diffs.mean()

    # enumerate all 2^n sign flips
    signs = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n)  # (2^n, n)
    null_stats = (signs * diffs).mean(axis=1)

    if alternative == "greater":
        p = (null_stats >= t_obs).mean()
    elif alternative == "less":
        p = (null_stats <= t_obs).mean()
    elif alternative == "two-sided":
        p = (np.abs(null_stats) >= np.abs(t_obs)).mean()
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    return dict(n=n, mean_delta=float(t_obs), p_value=float(p), null_stats=null_stats)


def paired_bootstrap_mean_ci(diffs, alpha=0.05, n_boot=10000, random_state=0):
    """
    Compute a percentile bootstrap confidence interval for the mean of
    paired differences.

    Parameters
    ----------
    diffs : array-like
        1D array of paired differences (d_i), one per technical replicate
        (e.g., per random seed). NaNs are ignored.

    alpha : float, optional (default=0.05)
        Significance level for the confidence interval. Returns a
        (1 - alpha) confidence interval.

    n_boot : int, optional (default=10000)
        Number of bootstrap resamples.

    random_state : int, optional (default=0)
        Random seed for reproducible bootstrap sampling.

    Returns
    -------
    ci_low : float
        Lower bound of the bootstrap confidence interval.

    ci_high : float
        Upper bound of the bootstrap confidence interval.

    Examples
    --------
    >>> diffs = [0.04, 0.03, 0.05, 0.02, 0.01]
    >>> paired_bootstrap_mean_ci(diffs, alpha=0.05, n_boot=1000, random_state=1)
    (0.01..., 0.04...)
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[b] = sample.mean()

    lo = np.quantile(boot_means, alpha / 2)
    hi = np.quantile(boot_means, 1 - alpha / 2)
    return float(lo), float(hi)


def all_unordered_pairs(items):
    items = list(items)
    return [
        (items[i], items[j])
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]


def summarize_paired_diffs(
    diffs_vec,
    a_vec,
    b_vec,
    alpha=0.05,
    n_boot=10000,
    random_state=0,
):
    """Helper function for dataset-dataset comparisons."""
    diffs_vec = np.asarray(diffs_vec, dtype=float)

    stat, p_val = wilcoxon_signed_rank_test(diffs_vec, alternative="two-sided")
    ci_low, ci_high = paired_bootstrap_mean_ci(
        diffs_vec, alpha=alpha, n_boot=n_boot, random_state=random_state
    )

    n = int(len(diffs_vec))
    wins = int(np.sum(diffs_vec > 0))

    return dict(
        n=n,
        wins=wins,
        wilcoxon_stat=stat,
        p_wilcoxon=p_val,
        ci_low=ci_low,
        ci_high=ci_high,
        mean_a=float(np.mean(a_vec)) if n else np.nan,
        mean_b=float(np.mean(b_vec)) if n else np.nan,
        mean_delta=float(np.mean(diffs_vec)) if n else np.nan,
    )


def add_fdr_bh_qvals(df, p_col="p_wilcoxon", q_col="q_fdr_bh", groupby_cols=None):
    """
    Add BH-FDR q-values to a DataFrame.

    If groupby_cols is provided, applies BH correction separately within each
    group; otherwise applies BH correction across the full DataFrame.
    """
    out = df.copy()
    if groupby_cols:
        out[q_col] = out.groupby(list(groupby_cols), dropna=False)[p_col].transform(
            fdr_bh
        )
    else:
        out[q_col] = fdr_bh(out[p_col].to_numpy())
    return out


# -------------------------
# Tidy results builder
# -------------------------
def make_dataset_pairs_baseline_vs_many(
    datasets,
    germline_baseline="germline_rare_lof",
    somatic_baseline="somatic_amp somatic_del somatic_mut",
):
    """
    Given a list of datasets, return pairs of (dataset, baseline)
    comparing all germline datasets to germline_baseline and all somatic-containing datasets to somatic_baseline.
    """
    ds = sorted(set(datasets))
    pairs = []

    for d in ds:
        if d in {germline_baseline, somatic_baseline}:
            continue

        if "somatic" in d:
            pairs.append((d, somatic_baseline))
        else:
            pairs.append((d, germline_baseline))

    return pairs


def compute_paired_wilcoxon_results(
    df_runs: pd.DataFrame,
    metric: str,
    seed_col: str = "random_seed",
    group_cols=("n_features", "odds_ratio", "control_frequency"),
    model_col: str = "model_type",
    pnet_label: str = "pnet",
    baseline_labels=("rf", "logistic_regression"),
    alpha: float = 0.05,
    n_boot: int = 10000,
    random_state: int = 0,
    fdr_correction: bool = False,
    fdr_groupby_cols=None,  # e.g. None (global), ["baseline"], ["baseline","n_features"]
    q_col: str = "q_fdr_bh",
):
    """
    NOTE: DEPRECATING
    Compute paired model comparison results using a two-sided Wilcoxon
    signed-rank test, with mean Δ and bootstrap CI.

    Returns one row per (group cell x baseline) with
          mean_delta, ci_low, ci_high, wins, n, p_perm_1s, plus group identifiers.

    Assumes df_runs contains one row per run and includes columns:
      - model_col
      - group_cols
      - seed_col
      - metric
    """

    # Filter rows with valid metric (mask-based to avoid pandas overload issues)
    keep_cols = list(group_cols) + [seed_col, model_col, metric]
    mask = df_runs[metric].notna()
    d = df_runs.loc[mask, keep_cols].copy()

    # Wide format: index = group + seed, columns = model
    wide = d.pivot_table(
        index=list(group_cols) + [seed_col],
        columns=model_col,
        values=metric,
        aggfunc="mean",
    )

    if pnet_label not in wide.columns:
        raise ValueError(f"{pnet_label} not found in model columns")

    rows = []

    for baseline in baseline_labels:
        if baseline not in wide.columns:
            continue

        pnet_vals = wide[pnet_label]
        base_vals = wide[baseline]

        diffs = (pnet_vals - base_vals).dropna()

        # align raw values for means
        pnet_df = pnet_vals.loc[diffs.index]
        base_df = base_vals.loc[diffs.index]

        merged = (
            diffs.rename("diff")
            .to_frame()
            .assign(pnet_val=pnet_df, base_val=base_df)
            .reset_index()
        )

        for keys, sub in merged.groupby(list(group_cols), sort=False):
            diffs_vec = sub["diff"].to_numpy()
            pnet_vec = sub["pnet_val"].to_numpy()
            base_vec = sub["base_val"].to_numpy()

            n = len(diffs_vec)
            wins = int(np.sum(diffs_vec > 0))

            # --- Wilcoxon signed-rank test (two-sided), via helper ---
            stat, p_val = wilcoxon_signed_rank_test(diffs_vec, alternative="two-sided")

            # --- bootstrap CI for mean Δ ---
            ci_low, ci_high = paired_bootstrap_mean_ci(
                diffs_vec,
                alpha=alpha,
                n_boot=n_boot,
                random_state=random_state,
            )

            row = dict(zip(group_cols, keys))
            row.update(
                baseline=baseline,
                **{
                    f"mean_pnet_{metric}": float(np.mean(pnet_vec))
                    if n > 0
                    else np.nan,
                    f"mean_baseline_{metric}": float(np.mean(base_vec))
                    if n > 0
                    else np.nan,
                    f"mean_delta_{metric}": float(np.mean(diffs_vec))
                    if n > 0
                    else np.nan,
                },
                ci_low=ci_low,
                ci_high=ci_high,
                wins=wins,
                n=n,
                wilcoxon_stat=stat,
                p_wilcoxon=p_val,
            )
            rows.append(row)

    df_out = pd.DataFrame(rows)
    if fdr_correction:
        df_out = add_fdr_bh_qvals(
            df_out,
            p_col="p_wilcoxon",
            q_col=q_col,
            groupby_cols=fdr_groupby_cols,
        )

    return df_out


def compute_paired_model_wilcoxon_results(
    df_runs: pd.DataFrame,
    metric: str,
    seed_col: str = "random_seed",
    group_cols=["datasets"],
    model_col="model_type",
    model_pairs=None,
    alpha: float = 0.05,
    n_boot: int = 10000,
    random_state: int = 0,
    fdr_correction: bool = False,
    fdr_groupby_cols=None,
    q_col: str = "q_fdr_bh",
):
    """Wrapper around the compute_paired_dataset_wilcoxon results to make it work for the model-model comparisons as well"""
    out = compute_paired_dataset_wilcoxon_results(
        df_runs=df_runs,
        metric=metric,
        seed_col=seed_col,
        group_cols=group_cols,
        dataset_col=model_col,  # this is the key difference
        dataset_pairs=model_pairs,
        alpha=alpha,
        n_boot=n_boot,
        random_state=random_state,
        fdr_correction=fdr_correction,
        fdr_groupby_cols=fdr_groupby_cols,
        q_col=q_col,
        require_matching_seeds=True,
    )
    out.rename(columns={"dataset_a": "model_a", "dataset_b": "model_b"}, inplace=True)

    return out


def compute_paired_dataset_wilcoxon_results(
    df_runs: pd.DataFrame,
    metric: str,
    seed_col: str = "random_seed",
    group_cols=(),
    dataset_col: str = "datasets",
    dataset_pairs=None,
    alpha: float = 0.05,
    n_boot: int = 10000,
    random_state: int = 0,
    fdr_correction: bool = False,
    fdr_groupby_cols=None,
    q_col: str = "q_fdr_bh",
    require_matching_seeds: bool = True,
):
    """
    Paired dataset-dataset comparisons using the Wilcoxon signed-rank test.

    This function compares a metric across dataset pairs using paired differences
    within replicate runs (paired by seed_col). Optional grouping variables
    (e.g., model_type) can be provided via group_cols to run separate paired tests
    within each group.

    Notes
    -----
    - There is no `model_col`. Treat model_type (or any other categorical splitter)
      as part of `group_cols`.
    - If require_matching_seeds=True, the function enforces that for each group and
      each dataset pair (a, b), the set of seeds present in dataset a equals the set
      present in dataset b. This prevents accidental unpaired comparisons caused by
      missing runs in one dataset.

    Parameters
    ----------
    df_runs : pd.DataFrame
        Long-format table with columns:
          - group_cols (optional),
          - seed_col,
          - dataset_col,
          - metric.
    metric : str
        Column name of the metric to compare (e.g., "test_avg_precision", "rank").
    seed_col : str
        Pairing identifier column (e.g., "random_seed").
    group_cols : sequence
        Columns defining groups within which to run paired tests (e.g., ("model_type",)).
    dataset_col : str
        Column that identifies the dataset used (forms the columns in the wide pivot).
    dataset_pairs : iterable or None
        Iterable of (dataset_a, dataset_b) pairs. If None, compares all unordered pairs.
    alpha, n_boot, random_state : float, int, int
        Passed through to summarize_paired_diffs for CI estimation.
    fdr_correction : bool
        If True, adds BH-FDR q-values.
    fdr_groupby_cols : list[str] or None
        Columns to group by when performing BH-FDR correction. If None, applies globally.
    q_col : str
        Name of q-value column to add.
    require_matching_seeds : bool
        If True, enforce identical seed sets between dataset_a and dataset_b within each group.
    logger : logging.Logger or None
        Optional logger for debug/info messages.

    Returns
    -------
    pd.DataFrame
        Paired comparison statistics for each dataset pair within each group.
    """
    group_cols = list(group_cols or [])

    # keep only rows with a defined metric
    mask = df_runs[metric].notna()
    keep_cols = group_cols + [seed_col, dataset_col, metric]
    d = df_runs.loc[mask, keep_cols].copy()

    # Build a unified group key column so we never have to special-case group_cols=()
    d = d.reset_index(drop=True)
    if group_cols:
        d["_group_key"] = list(zip(*(d[col] for col in group_cols)))
    else:
        d["_group_key"] = [()] * len(d)

    # Pivot to wide so each row is (group_key, seed) and columns are datasets.
    wide = d.pivot_table(
        index=["_group_key", seed_col],
        columns=dataset_col,
        values=metric,
        aggfunc="mean",
    )

    # Choose dataset pairs
    if dataset_pairs is None:
        dataset_pairs = all_unordered_pairs(wide.columns)

    # Precompute seed sets for strict checking
    seed_sets = None
    if require_matching_seeds:
        seed_sets = {}
        for (gkey, dset), g in d.groupby(["_group_key", dataset_col], sort=False):
            seed_sets[(gkey, dset)] = set(g[seed_col].unique())

    rows = []
    for a, b in dataset_pairs:
        if a not in wide.columns or b not in wide.columns:
            logger.debug(f"Skipping pair ({a}, {b}) because one dataset is missing.")
            continue

        sub = (
            wide[[a, b]].dropna(how="any").reset_index()
        )  # cols: _group_key, seed_col, a, b

        for gkey, g in sub.groupby("_group_key", sort=False):
            # Enforce seed set equality within this group, if requested
            if require_matching_seeds:
                set_a = seed_sets.get((gkey, a), set())
                set_b = seed_sets.get((gkey, b), set())
                if set_a != set_b:
                    missing_in_b = sorted(set_a - set_b)
                    missing_in_a = sorted(set_b - set_a)
                    # unpack group key for a more readable message if possible
                    group_desc = (
                        dict(zip(group_cols, gkey)) if group_cols else "(no groups)"
                    )
                    raise ValueError(
                        f"Seed mismatch for datasets ({a} vs {b}) within group {group_desc}: "
                        f"seeds in {a} but not {b}: {missing_in_b}; "
                        f"seeds in {b} but not {a}: {missing_in_a}"
                    )

            diffs = g[a].to_numpy() - g[b].to_numpy()
            stats = summarize_paired_diffs(
                diffs_vec=diffs,
                a_vec=g[a].to_numpy(),
                b_vec=g[b].to_numpy(),
                alpha=alpha,
                n_boot=n_boot,
                random_state=random_state,
            )

            row = {}
            if group_cols:
                for col_name, val in zip(group_cols, gkey):
                    row[col_name] = val

            row.update(
                dataset_a=a,
                dataset_b=b,
                **{
                    f"mean_dataset_a_{metric}": stats["mean_a"],
                    f"mean_dataset_b_{metric}": stats["mean_b"],
                    f"mean_delta_{metric}": stats["mean_delta"],
                },
                ci_low=stats["ci_low"],
                ci_high=stats["ci_high"],
                wins=stats["wins"],
                n=stats["n"],
                wilcoxon_stat=stats["wilcoxon_stat"],
                p_wilcoxon=stats["p_wilcoxon"],
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    if fdr_correction and len(out):
        out = add_fdr_bh_qvals(
            out, p_col="p_wilcoxon", q_col=q_col, groupby_cols=fdr_groupby_cols
        )
    return out


def compute_paired_perm_results(
    df_runs: pd.DataFrame,
    metric: str,
    seed_col: str = "random_seed",
    group_cols=("n_features", "odds_ratio", "control_frequency"),
    model_col: str = "model_type",
    pnet_label: str = "pnet",
    baseline_labels=("rf", "logistic_regression"),
    alternative: str = "greater",  # 1-sided: P-NET > baseline
    ci_alpha: float = 0.05,
    ci_n_boot: int = 10000,
    ci_random_state: int = 0,
) -> pd.DataFrame:
    """
    Returns one row per (group cell, baseline) with:
      mean_delta, ci_low, ci_high, wins, n, p_perm_1s, plus group identifiers.

    Assumes df_runs contains one row per run and includes:
      - model_col
      - group_cols
      - seed_col
      - metric
    """
    keep_cols = list(group_cols) + [seed_col, model_col, metric]

    mask = df_runs[metric].notna()
    d = df_runs.loc[mask, keep_cols].copy()

    # Wide: index=(group_cols + seed), columns=model, values=metric
    wide = d.pivot_table(
        index=list(group_cols) + [seed_col],
        columns=model_col,
        values=metric,
        aggfunc="mean",
    )

    if pnet_label not in wide.columns:
        raise ValueError(
            f"'{pnet_label}' not found in {model_col} after mapping. "
            f"Available: {list(wide.columns)}"
        )

    rows = []
    for baseline in baseline_labels:
        if baseline not in wide.columns:
            continue

            # Keep paired values for means
        pnet_vals = wide[pnet_label]
        base_vals = wide[baseline]

        diffs = (pnet_vals - base_vals).dropna()

        diffs_df = diffs.reset_index().rename(columns={diffs.name or 0: "diff"})
        if "diff" not in diffs_df.columns:
            diffs_df = diffs_df.rename(columns={diffs_df.columns[-1]: "diff"})

        # Also keep the paired raw values
        pnet_df = pnet_vals.reset_index().rename(columns={pnet_label: "pnet_val"})
        base_df = base_vals.reset_index().rename(columns={baseline: "base_val"})

        merged = diffs_df.merge(
            pnet_df, on=list(group_cols) + [seed_col], how="left"
        ).merge(base_df, on=list(group_cols) + [seed_col], how="left")

        for keys, sub in merged.groupby(list(group_cols), sort=False):
            diffs_vec = sub["diff"].to_numpy()
            pnet_vec = sub["pnet_val"].to_numpy()
            base_vec = sub["base_val"].to_numpy()

            n = int(len(diffs_vec))
            wins = int(np.sum(diffs_vec > 0))

            perm = paired_signflip_permutation_test(diffs_vec, alternative=alternative)
            ci_low, ci_high = paired_bootstrap_mean_ci(
                diffs_vec,
                alpha=ci_alpha,
                n_boot=ci_n_boot,
                random_state=ci_random_state,
            )

            row = dict(zip(group_cols, keys))
            row.update(
                baseline=baseline,
                mean_pnet=float(np.mean(pnet_vec)) if n > 0 else np.nan,
                mean_baseline=float(np.mean(base_vec)) if n > 0 else np.nan,
                mean_delta=float(np.mean(diffs_vec)) if n > 0 else np.nan,
                ci_low=ci_low,
                ci_high=ci_high,
                wins=wins,
                n=n,
                p_perm_1s=float(perm["p_value"]),
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    return out


# # Example usage of compute_paired_perm_results with fake data
# # One simulation cell
# group_cols = {
#     "n_features": 10,
#     "odds_ratio": 2.0,
#     "control_frequency": 0.1,
# }
# seeds = [0, 1, 2, 3, 4]

# rows = []

# # Construct values so that:
# # - P-NET vs RF: 4/5 wins (RF slightly higher on seed 4)
# # - P-NET vs LR: 3/5 wins (LR higher on seeds 1 and 3)

# pnet_vals = [0.80, 0.82, 0.81, 0.83, 0.79]

# rf_vals   = [0.75, 0.78, 0.77, 0.80, 0.80]  # seed 4: RF (0.80) > P-NET (0.79)

# lr_vals   = [0.79, 0.83, 0.80, 0.84, 0.78]  # seeds 1 & 3: LR > P-NET

# for seed, p, r, l in zip(seeds, pnet_vals, rf_vals, lr_vals):
#     rows.append({**group_cols, "random_seed": seed, "model_type": "P-NET", "train_avg_precision": p})
#     rows.append({**group_cols, "random_seed": seed, "model_type": "Random Forest", "train_avg_precision": r})
#     rows.append({**group_cols, "random_seed": seed, "model_type": "Logistic Regression", "train_avg_precision": l})

# df_fake2 = pd.DataFrame(rows)
# display(df_fake2)


# df_perm_test2 = compute_paired_perm_results(
#     df_runs=df_fake2,
#     metric="train_avg_precision",
#     seed_col="random_seed",
#     group_cols=("n_features", "odds_ratio", "control_frequency"),
#     model_col="model_type",
#     pnet_label="P-NET",
#     baseline_labels=("Random Forest", "Logistic Regression"),
#     alternative="greater",  # 1-sided: P-NET > baseline
#     ci_alpha=0.05,
#     ci_n_boot=5000,
#     ci_random_state=42,
# )

# df_perm_test2
