import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# DataLoader object for pytorch. Constructing single loader for all data input modalities.
class PnetDataset(Dataset):
    def __init__(self, genetic_data, target, indicies, additional_data=None, gene_set=None):
        """
        A dataset class for PyTorch that handles the loading and integration of multiple genetic data modalities.

        This class combines data from different genetic modalities (e.g., `mut`, `cnv`), links them to target labels,
        and supports batching for PyTorch. It ensures consistent handling of genes across modalities and can incorporate
        additional sample-specific features.

        Parameters:
        ----------
        genetic_data : Dict(str: pd.DataFrame)
            A dictionary of genetic modalities, where keys are modality names (e.g., 'mut', 'cnv') and values are
            pandas DataFrames with samples as rows and genes as columns. Paired samples must have matching indices
            across all modalities.
        target : pd.Series or pd.DataFrame
            The target variable for each sample. Can be binary or continuous, provided as a pandas Series or DataFrame
            with samples as the index.
        indicies : list of str
            A list of sample indices to include in the dataset.
        additional_data : pd.DataFrame, optional
            Additional features for each sample, indexed by sample names. Default is None.
        gene_set : list of str, optional
            A list of genes to be considered. By default, all overlapping genes across modalities are included.
        """

        assert isinstance(genetic_data, dict), f"input data expected to be a dict, got {type(genetic_data)}"
        for inp in genetic_data:
            assert isinstance(inp, str), f"input data keys expected to be str, got {type(inp)}"
            assert isinstance(genetic_data[inp], pd.DataFrame), (
                f"input data values expected to be a dict, got {type(genetic_data[inp])}"
            )
        self.genetic_data = genetic_data
        self.nbr_genetic_input_types = len(genetic_data)
        self.modalities = list(genetic_data.keys())
        self.target = target
        self.gene_set = gene_set
        self.altered_inputs = []
        self.inds = indicies
        if additional_data is not None:
            self.additional_data = additional_data.loc[self.inds]
        else:
            self.additional_data = pd.DataFrame(index=self.inds)  # create empty dummy dataframe if no additional data
        self.target = self.target.loc[self.inds]
        self.genes = self.get_genes()
        self.input_df = self.unpack_input()
        assert self.input_df.index.equals(self.target.index)
        self.x = torch.tensor(self.input_df.values, dtype=torch.float)
        self.y = torch.tensor(self.target.values, dtype=torch.float)
        self.additional = torch.tensor(self.additional_data.values, dtype=torch.float)

    def __len__(self):
        return self.input_df.shape[0]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        additional = self.additional[index]
        return x, additional, y

    def get_genes(self):
        """
        Identifies overlapping genes across all genetic_data modalities and gene_embeddings (if present).

        Parameters:
        ----------
        genetic_data : dict
            A dictionary of pandas DataFrames where each key is a modality name and
            each value is a DataFrame with samples as rows and genes as columns.
        gene_embeddings : pd.DataFrame
            A DataFrame where rows represent genes and columns represent embedding features.

        Returns:
        -------
        list
            A list of unique gene names found in all genetic_data modalities and gene_embeddings.
        """
        # Get sets of unique gene names from each modality
        gene_sets = [set(df.columns.unique()) for df in self.genetic_data.values()]

        # Additional gene set filter, if specified
        if self.gene_set:
            gene_sets.append(self.gene_set)

        # Find overlapping genes
        overlapping_genes = set.intersection(*gene_sets)

        print(f"Found {len(overlapping_genes)} overlapping genes")
        return list(overlapping_genes)

    def unpack_input(self):
        """
        Unpacks data modalities into one joint pd.DataFrame. Suffixing gene names by their modality name.
        :return: pd.DataFrame; containing n*m columns, where n is the number of modalities and m the number of genes
        considered.
        """
        input_df = pd.DataFrame(index=self.inds)
        for inp in self.genetic_data:
            temp_df = self.genetic_data[inp][self.genes]
            temp_df.columns = temp_df.columns + "_" + inp
            input_df = input_df.join(temp_df, how="inner", rsuffix="_" + inp)
        print(f"generated input DataFrame of size {input_df.shape}")
        return input_df.loc[self.inds]

    def save_indicies(self, path):
        df = pd.DataFrame(data={"indicies": self.inds})
        df.to_csv(path, sep=",", index=False)

    def generate_input_mask(self):
        """
        Generate the input mask for connecting genetic data to the model.
        - The input mask connects the same gene from different
             modalities to the input node.

        Returns:
        -------
        pd.DataFrame
            Input mask connecting genetic data and embeddings to the model.
        """
        expected_row_order = generate_feature_names(
            genes=self.genes,
            modalities=list(self.genetic_data.keys()),
        )

        input_mask = pd.DataFrame(index=expected_row_order, columns=self.genes).fillna(0)

        # Fill connections for modalities
        for modality in self.genetic_data:
            for gene in self.genes:
                row = f"{gene}_{modality}"
                input_mask.loc[row, gene] = 1

        return input_mask.values

    def generate_input_mask_marc_version(self):
        # used to live in ReactomeNetwork.py --> ReactomeNetwork class --> get_masks()
        input_mask = pd.DataFrame(index=len(self.genetic_data) * self.genes, columns=self.genes).fillna(0)
        for col in input_mask.columns:
            input_mask[col].loc[col] = 1
        return input_mask.values


# Dataset class that extends PnetDataset to include global gene embeddings.
class PnetDatasetWithGlobalEmbeddings(PnetDataset):
    """
    A dataset class that extends PnetDataset to include global gene embeddings.

    This class integrates global embeddings for each gene into the dataset.
    Each sample's feature vector contains:
        - Modality-specific features for each gene (e.g., `mut`, `cnv`).
        - A single global embedding for each gene, shared across modalities.

    Attributes:
    ----------
    genetic_data : Dict(str: pd.DataFrame)
        A dictionary where keys are modality names (e.g., 'mut', 'cnv') and values are
        pandas DataFrames containing samples as rows and genes as columns.
    target : pd.Series
        A Series containing the target variable for each sample.
    indices : list
        A list of sample indices to include in the dataset.
    gene_embeddings : pd.DataFrame
        A DataFrame where rows are genes and columns represent the global embedding features.
    additional_data : pd.DataFrame, optional
        Additional features for each sample, indexed by sample names.
    gene_set : list, optional
        A list of genes to include; if None, all overlapping genes across modalities are included.
    gene_embeddings: pd.DataFrame? or maybe pd.Series?
        A Series containing the embedding for each gene. Requires gene names that match those used in genetic_data.

    Methods:
    -------
    unpack_input():
        Combines genetic data and global embeddings into a unified input DataFrame for the model.
    """

    def __init__(self, genetic_data, target, indicies, gene_embeddings, additional_data=None, gene_set=None):
        super().__init__(genetic_data, target, indicies, additional_data, gene_set)
        self.gene_embeddings = gene_embeddings

    def get_genes(self):
        """
        Identifies overlapping genes across all genetic_data modalities and gene_embeddings (if present).

        Parameters:
        ----------
        genetic_data : dict
            A dictionary of pandas DataFrames where each key is a modality name and
            each value is a DataFrame with samples as rows and genes as columns.
        gene_embeddings : pd.DataFrame
            A DataFrame where rows represent genes and columns represent embedding features.

        Returns:
        -------
        list
            A list of unique gene names found in all genetic_data modalities and gene_embeddings.
        """
        # Get sets of unique gene names from each modality
        gene_sets = [set(df.columns.unique()) for df in self.genetic_data.values()]

        # Add unique set of genes we have embedding data for, if available
        if hasattr(self, "gene_embeddings") and self.gene_embeddings is not None:
            gene_sets.append(set(self.gene_embeddings.index.unique()))

        # Additional gene set filter, if specified
        if self.gene_set:
            gene_sets.append(self.gene_set)

        # Find overlapping genes with gene_embeddings
        overlapping_genes = set.intersection(*gene_sets)

        print(f"Found {len(overlapping_genes)} overlapping genes")
        return list(overlapping_genes)

    def unpack_input(self):
        """
        Combines modality-specific genetic data and global gene embeddings into a single DataFrame.
        Also performs gene filtration, filtering each DF down to self.genes.

        The resulting DataFrame includes:
        - Modality-specific features for each gene.
        - A single global embedding for each gene, expanded to match the number of samples.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the combined genetic data and global embeddings for all samples.
            Shape: (samples, n_genes x (n_modalities+embedding_length))
        """
        # Initialize the combined input DataFrame
        input_df = pd.DataFrame(index=self.inds)

        # Add modality-specific genetic data
        for modality_name in self.genetic_data:
            modality_data = self.genetic_data[modality_name][self.genes]
            modality_data.columns = [f"{col}_{modality_name}" for col in modality_data.columns]
            input_df = input_df.join(modality_data, how="inner")

        # Expand global embeddings to match the number of samples
        gene_emb_expanded = pd.DataFrame(
            np.tile(self.gene_embeddings.loc[self.genes].values.flatten(), len(input_df.index)).reshape(
                len(input_df.index), -1
            ),
            index=input_df.index,
            columns=[f"{gene}_embedding{i + 1}" for gene in self.genes for i in range(self.gene_embeddings.shape[1])],
        )

        # Combine genetic data and global embeddings
        input_df = pd.concat([input_df, gene_emb_expanded], axis=1)

        # Enforce column ordering
        expected_feature_order = generate_feature_names(
            genes=self.genes,
            modalities=self.modalities,
            embedding_length=self.gene_embeddings.shape[1],
        )
        input_df = input_df[expected_feature_order]

        return input_df

    def generate_input_mask(self):
        """
        Generate the input mask for connecting genetic data and embeddings to the first layer of the model.

        *Aligned Columns:*
        The expected_row_order ensures the rows in input_mask match the required structure.

        *Dealing with Data Modalities:*
        For each gene, we construct a list of rows corresponding to its modalities (e.g., GeneA_mut, GeneA_cnv).
        Update these rows for the column representing the gene in one operation (input_mask.loc[modality_rows, gene] = 1).
        - We need a connection (value=1) between each modality's gene and the appropriate node in gene layer of the model.
        - I.e., if we have 4 modalities, each gene layer node will have four incoming connections.

        *Embedding Connections:*
        Use np.eye(len(self.genes)) (identity matrix) to efficiently assign diagonal values (1s) for embedding connections.

        Returns:
        -------
        np.array
            Input mask connecting genetic data and embeddings to the first layer of the model.
        """
        embedding_length = self.gene_embeddings.shape[1] if self.use_embeddings else 0

        # Generate the expected row order
        expected_row_order = generate_feature_names(
            genes=self.genes,
            modalities=list(self.genetic_data.keys()),
            embedding_length=embedding_length,
        )

        # Initialize the input mask with zeros
        input_mask = pd.DataFrame(0, index=expected_row_order, columns=self.genes)

        # Fill connections for modalities (vectorized column-wise updates)
        for gene in self.genes:
            modality_rows = [f"{gene}_{modality}" for modality in self.genetic_data]
            input_mask.loc[modality_rows, gene] = 1

        # Fill connections for embeddings (Identity Matrix for Alignment)
        if self.use_embeddings:
            for embedding_idx in range(embedding_length):
                embedding_rows = [f"{gene}_embedding{embedding_idx + 1}" for gene in self.genes]
                input_mask.loc[embedding_rows, self.genes] = np.eye(len(self.genes))

        return input_mask.values

    def SLOW_generate_input_mask(self):  # TODO: delete when verify new one acts the same
        """
        Generate the input mask for connecting genetic data and embeddings to the model.

        Returns:
        -------
        pd.DataFrame
            Input mask connecting genetic data and embeddings to the model.
        """
        embedding_length = self.gene_embeddings.shape[1] if self.use_embeddings else 0
        expected_row_order = generate_feature_names(
            genes=self.genes,
            modalities=list(self.genetic_data.keys()),
            embedding_length=embedding_length,
        )

        input_mask = pd.DataFrame(index=expected_row_order, columns=self.genes).fillna(0)

        # Fill connections for modalities
        for modality in self.genetic_data:
            for gene in self.genes:
                row = f"{gene}_{modality}"
                input_mask.loc[row, gene] = 1

        # Fill connections for embeddings
        if self.use_embeddings:
            for embedding_idx in range(embedding_length):
                for gene in self.genes:
                    row = f"{gene}_embedding{embedding_idx + 1}"
                    input_mask.loc[row, gene] = 1

        return input_mask.values


def get_indicies(genetic_data, target, additional_data=None):
    """
    Generates a list of indicies which are present in all data modalities. Drops duplicated indicies.
    :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
    :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
        paired per sample index. Target can be binary or continuous.
    :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
     genetic data.
    :return: List(str); List of sample names found in all data modalities
    """
    for gd in genetic_data:
        genetic_data[gd].dropna(inplace=True)
    target.dropna(inplace=True)
    ind_sets = [set(genetic_data[inp].index.drop_duplicates(keep=False)) for inp in genetic_data]
    ind_sets.append(target.index.drop_duplicates(keep=False))
    if additional_data is not None:
        ind_sets.append(additional_data.index.drop_duplicates(keep=False))
    inds = list(set.intersection(*ind_sets))
    print(f"Found {len(inds)} overlapping indicies")
    return inds


def generate_dataset_from_indices(
    genetic_data,
    target,
    dset_inds,
    gene_set=None,
    additional_data=None,
    seed=None,
    shuffle_labels=False,
):
    """
    Takes all data modalities to be used and generates a train and test DataSet with a given split.
    :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
    :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
        paired per sample index. Target can be binary or continuous.
    :param dset_inds: List(str); List of sample indices to be included in the dataset.
    :param gene_set: List(str); List of genes to be considered, default is None and considers all genes found in every
        data modality.
    :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
    :param seed: int; Random seed to be used for train/test splits.
    :return:
    """
    print(f"Given {len(genetic_data)} Input modalities")
    inds = get_indicies(genetic_data, target, additional_data)
    random.seed(seed)
    random.shuffle(inds)
    dset_inds = list(set(inds).intersection(dset_inds))

    print(f"Initializing PnetDataset with {len(dset_inds)} samples")
    dset = PnetDataset(genetic_data, target, dset_inds, additional_data=additional_data, gene_set=gene_set)

    # Negative control: Shuffle labels for prediction
    if shuffle_labels:
        dset = shuffle_data_labels(dset)
    return dset


def generate_train_val_test_datasets(
    genetic_data,
    target,
    train_inds,
    validation_inds,
    test_inds,
    gene_set=None,
    additional_data=None,
    seed=None,
    shuffle_labels=False,
):
    """
    Takes all data modalities to be used and generates a train and test DataSet with a given split.
    :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
    :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
        paired per sample index. Target can be binary or continuous.
    :param gene_set: List(str); List of genes to be considered, default is None and considers all genes found in every
        data modality.
    :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
    :param test_split: float; Fraction of samples to be used for testing.
    :param seed: int; Random seed to be used for train/test splits.
    :return:
    """
    print(f"Given {len(genetic_data)} Input modalities")
    inds = get_indicies(genetic_data, target, additional_data)
    random.seed(seed)
    random.shuffle(inds)
    assert train_inds and validation_inds and test_inds, "train_inds, validation_inds and test_inds must be provided"

    train_inds = list(set(inds).intersection(train_inds))
    validation_inds = list(set(inds).intersection(validation_inds))
    test_inds = list(set(inds).intersection(test_inds))

    print("Initializing Train Dataset")
    train_dataset = PnetDataset(genetic_data, target, train_inds, additional_data=additional_data, gene_set=gene_set)
    print("Initializing Validation Dataset")
    validation_dataset = PnetDataset(
        genetic_data, target, validation_inds, additional_data=additional_data, gene_set=gene_set
    )
    print("Initializing Test Dataset")
    test_dataset = PnetDataset(genetic_data, target, test_inds, additional_data=additional_data, gene_set=gene_set)

    # Negative control: Shuffle labels for prediction
    if shuffle_labels:
        train_dataset = shuffle_data_labels(train_dataset)
        validation_dataset = shuffle_data_labels(validation_dataset)
        test_dataset = shuffle_data_labels(test_dataset)
    return train_dataset, validation_dataset, test_dataset


def generate_train_test(
    genetic_data,
    target,
    gene_set=None,
    additional_data=None,
    test_split=0.3,
    seed=None,
    train_inds=None,
    test_inds=None,
    collinear_features=0,
    shuffle_labels=False,
    use_embeddings=False,
    gene_embeddings=None,
):
    """
    Takes all data modalities to be used and generates a train and test DataSet with a given split.
    :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
    :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
        paired per sample index. Target can be binary or continuous.
    :param gene_set: List(str); List of genes to be considered, default is None and considers all genes found in every
        data modality.
    :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
    :param test_split: float; Fraction of samples to be used for testing.
    :param seed: int; Random seed to be used for train/test splits.
    :return:
    """
    print(f"Given {len(genetic_data)} Input modalities")
    inds = get_indicies(genetic_data, target, additional_data)
    random.seed(seed)
    random.shuffle(inds)
    if train_inds and test_inds:
        train_inds = list(set(inds).intersection(train_inds))
        test_inds = list(set(inds).intersection(test_inds))
    elif train_inds:
        train_inds = list(set(inds).intersection(train_inds))
        test_inds = [i for i in inds if i not in train_inds]
    elif test_inds:
        test_inds = list(set(inds).intersection(test_inds))
        train_inds = [i for i in inds if i not in test_inds]
    else:
        test_inds = inds[int((len(inds) + 1) * (1 - test_split)) :]
        train_inds = inds[: int((len(inds) + 1) * (1 - test_split))]

    # Making train and test datasets, following the embedding flag
    if use_embeddings:
        print("Initializing Train Dataset")
        train_dataset = PnetDatasetWithGlobalEmbeddings(
            genetic_data,
            target,
            train_inds,
            gene_embeddings=gene_embeddings,
            additional_data=additional_data,
            gene_set=gene_set,
        )
        print("Initializing Test Dataset")
        test_dataset = PnetDatasetWithGlobalEmbeddings(
            genetic_data,
            target,
            test_inds,
            gene_embeddings=gene_embeddings,
            additional_data=additional_data,
            gene_set=gene_set,
        )
    else:
        print("Initializing Train Dataset")
        train_dataset = PnetDataset(
            genetic_data, target, train_inds, additional_data=additional_data, gene_set=gene_set
        )
        print("Initializing Test Dataset")
        test_dataset = PnetDataset(genetic_data, target, test_inds, additional_data=additional_data, gene_set=gene_set)

    # Positive control: Replace a gene's values with values collinear to the target
    train_dataset, test_dataset = add_collinear(train_dataset, test_dataset, collinear_features)
    # Positive control: Shuffle labels for prediction
    if shuffle_labels:
        train_dataset = shuffle_data_labels(train_dataset)
        test_dataset = shuffle_data_labels(test_dataset)
    return train_dataset, test_dataset


def to_dataloader(train_dataset, test_dataset, batch_size):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(123)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,)
    # val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,)
    # based on https://pytorch.org/docs/stable/notes/randomness.html for reproducibility of DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, val_loader


def add_collinear(train_dataset, test_dataset, collinear_features):
    if isinstance(collinear_features, list):
        for f in collinear_features:
            replace_collinear(train_dataset, test_dataset, f)
    else:
        for n in range(collinear_features):
            r = random.randint(0, len(train_dataset.input_df.columns))
            altered_input_col = train_dataset.input_df.columns[r]
            train_dataset, test_dataset = replace_collinear(train_dataset, test_dataset, altered_input_col)
    return train_dataset, test_dataset


def shuffle_data_labels(dataset):
    print(f"shuffling {dataset.target.shape[0]} labels")
    target_copy = dataset.target.copy()
    target_copy[target_copy.columns[0]] = dataset.target.sample(frac=1).reset_index(drop=True).values
    dataset.target = target_copy
    return dataset


def replace_collinear(train_dataset, test_dataset, altered_input_col):
    train_dataset.altered_inputs.append(altered_input_col)
    test_dataset.altered_inputs.append(altered_input_col)
    print(f"Replace input of: {altered_input_col} with collinear feature.")
    train_dataset.input_df[altered_input_col] = train_dataset.target
    test_dataset.input_df[altered_input_col] = test_dataset.target
    return train_dataset, test_dataset


def generate_feature_names(genes, modalities, embedding_length=0):
    """
    Generate a canonical order of feature names based on genes, modalities, and embeddings.

    Parameters:
    ----------
    genes : list of str
        List of gene names.
    modalities : list of str
        Names of genetic input modalities (e.g., ['mut', 'cnv']).
    embedding_length : int
        Length of the gene embedding vector (default = 0).

    Returns:
    -------
    list of str
        Ordered list of feature names.
    """
    feature_names = []

    for gene in genes:
        # Add modality-specific names
        feature_names.extend([f"{gene}_{modality}" for modality in modalities])
        # Add embedding names
        feature_names.extend([f"{gene}_embedding{i + 1}" for i in range(embedding_length)])

    return feature_names
