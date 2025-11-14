import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split

import wandb
from pnet import Pnet, pnet_loader, report_and_eval, util
from pnet.utils import modeling_utils

logging.basicConfig(
    encoding="utf-8",
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    # force=True,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_mu1_from_or(mu0, OR):
    "Compute the alternative allele frequency mu1 given the reference allele frequency mu0 and odds ratio OR."
    logger.debug(f"Computing mu1 from mu0 with OR={OR}.")
    return (mu0 * OR) / (1 + mu0 * (OR - 1))


def make_mu_vectors(num_genes, high_or_genes=10, OR=1.0, mu0_range=(0.001, 0.1)):
    """
    Generate two vectors of allele frequencies mu0 and mu1 for a set of genes.
    mu0 is uniformly sampled from a specified range, and mu1 is computed based on mu0 and an odds ratio OR.
    A specified number of genes (high_or_genes) will have a delta of 1, indicating they are perturbed.
    Args:
        num_genes (int): Total number of genes.
        high_or_genes (int): Number of genes to perturb (set delta to 1).
        OR (float): Odds ratio to compute mu1 from mu0.
        mu0_range (tuple): Range from which to sample mu0 values.
    """
    logger.info(f"Generating mu0 and mu1 vectors for {num_genes} genes with OR={OR}, high_or_genes={high_or_genes}.")
    logger.info(f"mu0 will be sampled uniformly from {mu0_range}.")
    mu0 = np.random.uniform(*mu0_range, size=num_genes)
    delta = np.zeros(num_genes)
    idx = np.random.choice(num_genes, high_or_genes, replace=False)
    delta[idx] = 1
    logger.info(f"mu1 identical to mu0 except for the {high_or_genes} perturbed genes, where the OR defines mu1 value.")
    mu1 = compute_mu1_from_or(mu0, OR) * delta + mu0 * (1 - delta)
    return mu0, mu1, idx


def make_block_correlation_matrix(num_genes, module_genes, sigma, noise_std=0.01):
    """
    Use make_block_correlation_matrix to set up a submatrix (module) with a constant correlation sigma.
    Remaining entries filled with low-magnitude noise (suggest Normal(0, 0.01) or Beta(1, 100) for correlation-compatible values)

    Create a block correlation matrix for a set of genes, where a specified subset of genes (module_genes)
    have a high correlation (sigma) with each other, while the rest of the genes have a small random noise correlation.
    The matrix is symmetric and has a diagonal of 1s.
    Args:
        num_genes (int): Total number of genes.
        module_genes (array-like): Indices of genes that form a module with high correlation.
        sigma (float): Correlation value for the module genes.
        noise_std (float): Standard deviation of the noise added to the correlation matrix.
    Returns:
        np.ndarray: A symmetric correlation matrix of shape (num_genes, num_genes).
    """
    logger.info(
        f"Creating block correlation matrix for {num_genes} genes with {len(module_genes)} module genes and within module correlation strength of sigma={sigma}."
    )
    R = np.random.normal(0, noise_std, size=(num_genes, num_genes))
    R = (R + R.T) / 2  # make symmetric
    np.fill_diagonal(R, 1.0)

    for i in module_genes:
        for j in module_genes:
            if i != j:
                R[i, j] = sigma
                R[j, i] = sigma
    return R


def correlation_to_covariance(R, mu, mode="binary", gene_std=None):
    """
    Construct a covariance matrix from a correlation matrix R.

    Parameters:
        R (np.ndarray): Correlation matrix of shape (G, G)
        mu (np.ndarray): Mean vector of shape (G,)
        mode (str): "binary" or "continuous"
        gene_std (np.ndarray or None): Std dev per gene (only used if mode="continuous")

    Returns:
        Sigma (np.ndarray): Covariance matrix of shape (G, G)
    """
    if mode == "binary":
        Sigma = R  # Use latent correlation structure directly
    elif mode == "continuous":
        if gene_std is None:
            gene_std = np.ones_like(mu)  # Default to unit variance if not specified
        Sigma = R * np.outer(gene_std, gene_std)
    else:
        raise ValueError("mode must be either 'binary' or 'continuous'")
    assert np.all(np.diag(Sigma) > 1e-6), "Covariance matrix has near-zero variance."
    return Sigma


def sample_continuous_genotypes(mu, Sigma, n_samples):
    """
    Sample continuous genotypes from a multivariate normal distribution with mean mu and covariance Sigma.
    Args:
        mu (np.ndarray): Mean vector for the multivariate normal distribution.
        Sigma (np.ndarray): Covariance matrix for the multivariate normal distribution.
        n_samples (int): Number of samples to generate.
    Returns:
        np.ndarray: A matrix of shape (n_samples, len(mu)) containing the sampled genotypes.
    """
    logger.info(f"Sampling continuous genotypes for {n_samples} samples from a multivariate normal distribution.")
    return np.random.multivariate_normal(mean=mu, cov=Sigma, size=n_samples)


def sample_binary_genotypes(mu, Sigma, n_samples):
    """
    Sample binary genotypes, where the binary genotypes are modeled as thresholded Gaussian variables.
    First, sample from a multivariate normal distribution with mean 0 and covariance Sigma.
    Then, apply a threshold based on the allele frequencies mu to convert to binary genotypes.
    Returns:
        np.ndarray: A matrix of shape (n_samples, len(mu)) containing binary genotypes (0 or 1).
    """
    logger.info(
        f"Sampling binary genotypes for {n_samples} samples where binary genotypes are modeled as thresholded Gaussian (multivariate normal) variables."
    )
    z = np.random.multivariate_normal(mean=np.zeros(len(mu)), cov=Sigma, size=n_samples)
    thresholds = norm.ppf(1 - mu)

    # DEBUGGING PRINTS
    logger.debug(f"Sigma diagonal (first 5): {np.round(np.diag(Sigma)[:5], 4)}")
    logger.debug(f"Sampled z (first 2 samples): {np.round(z[:2], 3)}")
    logger.debug(f"Thresholds (first 5): {np.round(thresholds[:5], 3)}")

    return (z > thresholds).astype(int)


def get_module_genes(mu, N=30):
    """
    Ensures a diverse range of baseline mutation probabilities in the correlation module by selecting genes that:
    - Are among the top 10 highest deltaMu
    - Are among the bottom 10 lowest probabilities
    - Are evenly spaced in the middle of the distribution

    Args:
        mu (np.ndarray): Vector of event frequency (allele frequencies) per gene. Best to use either mu1 or mu1-mu0 (deltaMu).
        N (int): Number of genes to include in the module.
    """
    logger.info(f"Selecting {N} module genes based on mutation probabilities.")
    argsorted = np.argsort(mu)
    top = argsorted[-10:]
    bottom = argsorted[:10]
    step = len(mu) // 10
    middle = argsorted[step // 2 :: step][:10]
    return np.unique(np.concatenate([top, bottom, middle]))


def get_module_genes_basic(num_genes, perturbed_genes, frac=0.5):
    """
    Select a subset of genes to form a module based on the perturbed genes.
    The module will contain a fraction of the perturbed genes and some random genes (of equal size to the portion you keep from the perturbed genes).

    Args:
        num_genes (int): Total number of genes.
        perturbed_genes (array-like): Indices of perturbed genes.
        frac (float): Fraction of perturbed genes to include in the module.

    Returns:
        np.ndarray: Indices of the selected module genes.
    """
    logger.info(
        f"Selecting module genes such that half come from the perturbed genes, and an equal number come from unperturbed genes. We include {frac * 100}% of all the perturbed genes."
    )
    num_to_keep = int(len(perturbed_genes) * frac)
    unperturbed_genes = np.setdiff1d(np.arange(num_genes), perturbed_genes)

    perturbed_genes_to_keep = np.random.choice(perturbed_genes, num_to_keep, replace=False)
    unperturbed_genes_to_keep = np.random.choice(unperturbed_genes, num_to_keep, replace=False)
    module_genes = np.concatenate([perturbed_genes_to_keep, unperturbed_genes_to_keep])
    assert len(module_genes) == 2 * num_to_keep, "Module genes should be twice the number of perturbed genes kept."
    return module_genes


# Example: visualize heatmap matrix for specific genes
def visualize_matrix(matrix, title="Heatmap", gene_indices=None):
    if gene_indices is not None:
        matrix_subset = matrix[np.ix_(gene_indices, gene_indices)]
    else:
        matrix_subset = matrix

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_subset, cmap="coolwarm", center=0, annot=False)

    plt.title(title)
    plt.xlabel("Gene index")
    plt.ylabel("Gene index")
    plt.tight_layout()
    return plt


def simulate_dataset(
    num_genes=1000,
    n0=1000,
    n1=1000,
    OR=10.0,
    sigma=1.0,
    num_perturbed_mu_genes=20,
    sample_binary=True,
    mu0_range=(0.1, 0.1),
):  # simplest case
    mode = "binary" if sample_binary else "continuous"

    mu0, mu1, perturbed_genes = make_mu_vectors(
        num_genes, high_or_genes=num_perturbed_mu_genes, OR=OR, mu0_range=mu0_range
    )

    mod1_genes = get_module_genes_basic(num_genes, perturbed_genes, frac=0.5)
    excluded_genes = np.union1d(perturbed_genes, mod1_genes)
    mod0_genes = np.setdiff1d(np.arange(num_genes), excluded_genes)[: len(mod1_genes)]

    R1 = make_block_correlation_matrix(num_genes, mod1_genes, sigma)
    R0 = make_block_correlation_matrix(num_genes, mod0_genes, sigma)

    logger.debug("Compute correlation (R) heatmaps...")
    wandb.log(
        {
            "R1 (mod1 genes)": wandb.Image(
                visualize_matrix(R1, gene_indices=mod1_genes, title=f"R1 (mod1 genes), sigma {sigma}")
            )
        }
    )
    wandb.log(
        {
            "R0 (mod0 genes)": wandb.Image(
                visualize_matrix(R0, gene_indices=mod0_genes, title=f"R0 (mod0 genes), sigma {sigma}")
            )
        }
    )
    wandb.log(
        {"R1 (all genes)": wandb.Image(visualize_matrix(R1, gene_indices=None, title=f"R1 (all genes), sigma {sigma}"))}
    )
    plt.close("all")

    Sigma1 = correlation_to_covariance(R1, mu1, mode=mode)
    Sigma0 = correlation_to_covariance(R0, mu0, mode=mode)

    logger.debug("Compute covariance (Sigma) heatmaps...")
    wandb.log(
        {
            "Sigma1 (mod1 genes)": wandb.Image(
                visualize_matrix(Sigma1, gene_indices=mod1_genes, title=f"Sigma1 (mod1 genes, sigma {sigma})")
            )
        }
    )
    wandb.log(
        {
            "Sigma0 (mod0 genes)": wandb.Image(
                visualize_matrix(Sigma0, gene_indices=mod0_genes, title=f"Sigma0 (mod0 genes, sigma {sigma})")
            )
        }
    )
    wandb.log(
        {
            "Sigma1 (all genes)": wandb.Image(
                visualize_matrix(Sigma1, gene_indices=None, title=f"Sigma1 (all genes), sigma {sigma}")
            )
        }
    )
    plt.close("all")

    if sample_binary:
        X1 = sample_binary_genotypes(mu1, Sigma1, n1)
        X0 = sample_binary_genotypes(mu0, Sigma0, n0)
    else:
        X1 = sample_continuous_genotypes(mu1, Sigma1, n1)
        X0 = sample_continuous_genotypes(mu0, Sigma0, n0)

    y = np.concatenate([np.ones(n1), np.zeros(n0)])
    X = np.vstack([X1, X0])

    return X, y, mu0, mu1, mod0_genes, mod1_genes, perturbed_genes


# Function to add gene names to X and update mod0_genes and mod1_genes and deltaMuGenes so they correspond to these gene names
def add_gene_names(X, mod0_genes, mod1_genes, deltaMuGenes, gene_names=None):
    if not gene_names:
        gene_names = [f"Gene_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=gene_names)

    # Update mod0_genes, mod1_genes, and deltaMuGenes to use gene names
    mod0_genes_named = [gene_names[i] for i in mod0_genes]
    mod1_genes_named = [gene_names[i] for i in mod1_genes]
    deltaMuGenes_named = [gene_names[i] for i in deltaMuGenes]

    return X_df, mod0_genes_named, mod1_genes_named, deltaMuGenes_named


def _get_gene_list():
    data_dir = "/mnt/disks/gmiller_data1/pnet_germline/processed/wandb-group-data_prep_germline_tier12_and_somatic/converted-IDs-to-somatic_imputed-germline_True_imputed-somatic_False_paired-samples-True/wandb-run-id-u5yt90p1"
    somatic_f = os.path.join(data_dir, "somatic_mut.csv")
    somatic_df = pd.read_csv(somatic_f, index_col=0)
    somatic_df.head()
    # genes_in_reactome = pd.read_csv("/mnt/disks/gmiller_data1/pnet_germline/data/pnet_database/genes/tcga_prostate_expressed_genes_and_cancer_genes_and_memebr_of_reactome.csv")
    genes = somatic_df.columns.tolist()
    return genes


def stratified_train_val_test_split(labels, train_size=0.7, val_size=0.15, test_size=0.15, seed=None, logger=None):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    # First split: train vs temp (val + test)
    train_inds, temp_inds = train_test_split(
        labels.index, test_size=(val_size + test_size), stratify=labels, random_state=seed
    )

    # Second split: val vs test from temp
    temp_labels = labels.loc[temp_inds]
    val_ratio = val_size / (val_size + test_size)
    val_inds, test_inds = train_test_split(
        temp_inds,
        test_size=(1 - val_ratio),  # 50% test if val:test = 0.15:0.15
        stratify=temp_labels,
        random_state=seed,
    )

    if logger is not None:
        for name, inds in zip(["training", "validation", "test"], [train_inds, val_inds, test_inds]):
            class_counts = labels.loc[inds].value_counts(normalize=True)
            logger.info(f"{name.title()} set: {len(inds)} samples")
            logger.info(f"Class distribution:\n{class_counts.to_string()}")

    return train_inds.tolist(), val_inds.tolist(), test_inds.tolist()


def train_model_pnet(hparams, genetic_data, y, train_inds=None, test_inds=None):
    logger.info("Training PNET model")
    model_save_path = os.path.join(hparams["save_dir"], "model.pt")

    # Pnet.run will do auto 70/30 train/val split if not passed train/test indices
    model, train_losses, test_losses, train_dataset, test_dataset = Pnet.run(
        genetic_data,
        y,
        save_path=model_save_path,
        dropout=hparams["dropout"],
        input_dropout=hparams["input_dropout"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        batch_size=hparams["batch_size"],
        epochs=hparams["epochs"],
        verbose=hparams["verbose"],
        early_stopping=hparams["early_stopping"],
        seed=hparams["random_seed"],
        train_inds=train_inds,
        test_inds=test_inds,
    )

    logger.info("Logging loss curve")
    plt = report_and_eval.get_loss_plot(train_losses=train_losses, test_losses=test_losses)
    wandb.log({"convergence plot": wandb.Image(plt)})
    report_and_eval.savefig(plt, os.path.join(hparams["save_dir"], "loss_over_time"))
    plt.close()
    return model, train_losses, test_losses, train_dataset, test_dataset, model_save_path


# TODO: consider altering to take in train/val/test datasets instead of genetic_data and y, so can just use function from modeling_utils.py
def evaluate_on_train_val_test(model, genetic_data, y, train_inds, val_inds, test_inds, hparams, task="BC"):
    """
    Evaluate a trained model on train, val, and test splits using PnetDataset.

    Args:
        model: Trained model object
        genetic_data: Input feature data (e.g., DataFrame or tensor)
        y: Target labels (Pandas Series or compatible)
        train_inds, val_inds, test_inds: Index arrays for each split
        hparams: Dict of hyperparameters including model_type and save_dir
    """
    y = util.format_target(y, task)

    for split_name, split_inds in zip(["train", "validation", "test"], [train_inds, val_inds, test_inds]):
        logger.info(f"Evaluating model on {split_name} set ({len(split_inds)} samples)")

        split_dataset = pnet_loader.PnetDataset(genetic_data, target=y, indicies=split_inds)

        report_and_eval.evaluate_interpret_save(
            model=model,
            pnet_dataset=split_dataset,
            model_type=hparams["model_type"],
            who=split_name,
            save_dir=hparams["save_dir"],
        )
    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate simulated data by sampling from joint distribution defined by an odds ratio and a sigma."
    )
    parser.add_argument("--odds_ratio", type=float, default=10.0, help="Odds ratio to simulate")
    parser.add_argument(
        "--sigma", type=float, default=0.0, help="Correlation strength between genes in the class-specific module"
    )
    parser.add_argument("--num_class1_samples", type=int, default=500, help="Number of class 1 samples")
    parser.add_argument("--num_class0_samples", type=int, default=500, help="Number of class 0 samples")
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="continuous",
        choices=["binary", "continuous"],
        help="Sampling type: 'binary' or 'continuous' (default: continuous)",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to run ML model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_type", type=str, default="pnet", help="Model type (default: pnet)")
    parser.add_argument("--evaluation_set", type=str, default="validation", help="Evaluation set (default: validation)")
    parser.add_argument("--wandb_group", type=str, default="simulated_data_001", help="wandb group name")
    return parser.parse_args()


def main():
    args = parse_args()

    run_name = f"OR_{args.odds_ratio}_sigma_{args.sigma}_n1_{args.num_class1_samples}_n0_{args.num_class0_samples}"
    logger.info(f"Starting run: {run_name}")
    run = wandb.init(project="prostate_met_status", group=args.wandb_group, name=run_name, reinit=True)

    save_dir = f"/mnt/disks/gmiller_data1/pnet/results/{args.wandb_group}/{args.model_type}_eval_set_{args.evaluation_set}/wandbID_{run.id}"
    logger.info(f"Results will be saved to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Build hparams
    base_hparams = {
        "epochs": args.epochs,
        "early_stopping": True,
        "batch_size": 64,
        "verbose": False,
        "random_seed": args.seed,
        "model_type": args.model_type,
        "evaluation_set": args.evaluation_set,
        "dropout": 0.2,
        "input_dropout": 0.5,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "delete_model_after_training": True,
        "sample_binary": args.sampling_strategy == "binary",
        "mu0_range": (
            0.1,
            0.1,
        ),  # range from which to sample mu0 values (for binary, this is the probability of the reference allele; for continuous, this is the mean of the Gaussian distribution e.g. mean gene expression level)
    }

    logger.info(f"Simulating dataset with OR={args.odds_ratio}, sigma={args.sigma}...")
    X, y, mu0, mu1, mod0_genes, mod1_genes, deltaMuGenes = simulate_dataset(
        num_genes=100,
        n0=args.num_class0_samples,
        n1=args.num_class1_samples,
        OR=args.odds_ratio,
        sigma=args.sigma,
        num_perturbed_mu_genes=20,
        sample_binary=base_hparams["sample_binary"],
        mu0_range=base_hparams["mu0_range"],
    )

    logger.info("Prepping simulated data for PNET...")
    gene_names = _get_gene_list()[: X.shape[1]]  # Ensure gene_names matches the number of columns in X
    assert len(gene_names) == X.shape[1], "Not enough gene names for number of features"
    X, mod0_genes, mod1_genes, deltaMuGenes = add_gene_names(
        X, mod0_genes, mod1_genes, deltaMuGenes, gene_names=gene_names
    )

    # Add sample IDs to X and y
    X.index = [f"Sample_{i}" for i in range(X.shape[0])]

    # add sample IDs to y, make y a pandas DF, ensure class is type int
    y = pd.DataFrame(y.astype(int), index=X.index, columns=["class"])
    genetic_data = {"simulated_binary": X} if base_hparams["sample_binary"] else {"simulated_continuous": X}
    logger.info("Simulated data prepared for PNET.")

    # Make train/val/test indicies with equal class balance in 70/15/15 split
    train_inds, val_inds, test_inds = stratified_train_val_test_split(
        y["class"],
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        seed=base_hparams["random_seed"],
        logger=logger,
    )

    # Update hparams with current OR and sigma
    hparams = {
        "odds_ratio": args.odds_ratio,
        "sigma": args.sigma,
        "num_genes": X.shape[1],
        "num_samples": X.shape[0],
        "num_class1_samples": args.num_class1_samples,
        "num_class0_samples": args.num_class0_samples,
        "mod0_genes": mod0_genes,
        "mod1_genes": mod1_genes,
        "deltaMuGenes": deltaMuGenes,
        "num_perturbed_mu_genes": len(deltaMuGenes),
        "num_genes_in_mod1": len(mod1_genes),
        "save_dir": save_dir,
        "datasets": list(genetic_data.keys()),
        **base_hparams,
    }
    run.config.update(hparams)

    logger.info(f"Running {args.model_type} model on simulated data with hparams: {hparams}")
    if args.model_type == "pnet":
        model, _, _, train_dataset, val_dataset, model_save_path = train_model_pnet(
            hparams, genetic_data, y, train_inds=train_inds, test_inds=val_inds
        )

        evaluate_on_train_val_test(
            model=model,
            genetic_data=genetic_data,
            y=y,
            train_inds=train_inds,
            val_inds=val_inds,
            test_inds=test_inds,
            hparams=hparams,
        )

        modeling_utils.cleanup(
            model_save_path=model_save_path,
            delete_model_after_training=hparams["delete_model_after_training"],
        )
    else:
        raise ValueError(
            f"Model type {args.model_type} not supported. Please implement training logic for this model type."
        )


if __name__ == "__main__":
    main()
