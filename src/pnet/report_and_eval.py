"""
Functions for
- reporting information during running of a program, eg details about DFs
- reporting evaluation information for models
"""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing packages related to model performance
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,  # aka the AUC-PRC score
    balanced_accuracy_score,
    classification_report,  # expects true_labels, predicted_labels
    confusion_matrix,  # expects true_labels, predicted_labels
    f1_score,
    mean_squared_error,
    roc_auc_score,  # expects true_labels, predicted_probs
)

import wandb

logger = logging.getLogger(__name__)

#################
# Reporting
#################


def make_path_if_needed(file_path):
    directory = os.path.dirname(file_path)
    make_dir_if_needed(directory)
    return


def make_dir_if_needed(directory):
    if not os.path.isdir(directory) and directory != "":
        logger.debug(f"Directory did not exist; making directory {directory}")
        os.makedirs(directory)
    return


def savefig(plt, save_path, png=True, pdf=True):
    make_path_if_needed(save_path)
    logger.info(f"saving plot to {save_path}")
    if png:
        plt.savefig(save_path, bbox_inches="tight")
    if pdf:
        plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight")


def report_df_info(*dataframes, n=5):  # TODO: use this instead of load_df_verbose
    """
    Report information about an arbitrary number of dataframes.

    Parameters:
    *dataframes (pd.DataFrame): Arbitrary number of dataframes to report information about.
    n (int): Number of columns and indices to display.

    Returns:
    None

    # Example usage:
    data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    data2 = {'X': [7, 8, 9], 'Y': [10, 11, 12]}

    df1 = pd.DataFrame(data1, index=['row1', 'row2', 'row3'])
    df2 = pd.DataFrame(data2, index=['row4', 'row5', 'row6'])

    # Call the function to report information about the dataframes
    report_df_info(df1, df2)
    """

    for idx, df in enumerate(dataframes, start=1):
        print(f"----- DataFrame {idx} Info -----")
        print(f"Shape: {df.shape}")
        print(f"First {n} columns: {df.columns[:n].tolist()}")
        print(f"First {n} indices: {df.index[:n].tolist()}")
        print("-----")
    return


def report_df_info_with_names(df_dict, n=5):
    """
    Report information about dataframes (with DF names provided in a dictionary for convenience).

    Parameters:
    df_dict (dict): A dictionary where keys are names and values are dataframes.
    n (int): Number of columns and indices to display.

    Returns:
    None

    # Example usage:
    data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    data2 = {'X': [7, 8, 9], 'Y': [10, 11, 12]}

    df1 = pd.DataFrame(data1, index=['row1', 'row2', 'row3'])
    df2 = pd.DataFrame(data2, index=['row4', 'row5', 'row6'])

    # Create a dictionary with dataframe names
    dataframes_dict = {"DataFrame 1": df1, "DataFrame 2": df2}

    # Call the function with the dictionary of dataframes
    report_df_info_with_names(dataframes_dict)

    # Alternatively, use dict(zip()) instead of writing out a dictionary
    names = ['Dataframe 1', 'DF2']
    dfs = [df1, df2]
    report_df_info_with_names(dict(zip(names, dfs)))
    """

    for name, df in df_dict.items():
        print(f"----- DataFrame {name} Info -----")
        print(f"Shape: {df.shape}")
        print(f"First {n} columns: {df.columns[:n].tolist()}")
        print(f"First {n} indices: {df.index[:n].tolist()}")
        print("-----")
    return


#################
# Evaluation of model
#################


def get_loss_plot(
    train_losses,
    test_losses,
    train_label="Train loss",
    test_label="Validation loss",
    title="Model Loss",
    ylabel="Loss",
    xlabel="Epochs",
):
    logger.info("Making a loss plot over time")
    # Sample data
    epochs = range(1, len(train_losses) + 1)

    # Plotting the lines
    plt.plot(epochs, train_losses, label=train_label)
    plt.plot(epochs, test_losses, label=test_label)

    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Adding a legend
    plt.legend()
    return plt


def get_pnet_preds_and_probs(model, pnet_dataset):
    """
    Get the predictions and probabilities from a Pnet model.
    Args:
    - model: this is a Pnet model object
    - pnet_dataset: this is a Pnet dataset object (not a DF), and has attributes x, additional, y, etc.

    Outputs:
    - preds: 1D array of the predicted labels, shape: (n_samples,)
    - pred_probas: 1D array pf the predicted probabilities, shape: (n_samples,)
    """
    model.to("cpu")
    x = pnet_dataset.x
    additional = pnet_dataset.additional
    logger.debug("Running `get_pnet_preds_and_probs`.\nSanitize for logging and downstream use (.detach().cpu())")
    pred_probas = model.predict_proba(x, additional).detach().cpu()
    preds = model.predict(x, additional).detach().cpu()

    logger.debug(f"Shape of pred_probas from `model.predict_proba`: {pred_probas.shape}")
    logger.debug(f"Type of pred_probas from `model.predict_proba`: {type(pred_probas)}")

    # Ensure preds is 1D
    if preds.dim() > 1:
        preds = preds.squeeze(1)  # Converts shape from [N, 1] → [N]

    # Ensure pred_probas is 1D
    if pred_probas.dim() > 1:
        pred_probas = pred_probas.squeeze(1)

    # Now convert to NumPy arrays since all downstream tools expect numpy, not tensors
    preds = preds.numpy()
    pred_probas = pred_probas.numpy()

    logger.debug(f"Shape of preds: {preds.shape}")
    logger.debug(f"Shape of pred_probas: {pred_probas.shape}")
    logger.debug(f"Type of preds: {type(preds)}")
    logger.debug(f"Type of pred_probas: {type(pred_probas)}")

    return preds, pred_probas


def prepare_labels(y):
    logger.debug("Preparing labels")
    logger.debug(f"Start y[0:5]: {y[0:5]}")
    if hasattr(y, "detach"):
        y = y.detach().cpu()
    if hasattr(y, "numpy"):
        y = y.numpy()
    y = np.asarray(y).squeeze().tolist()
    logger.debug(f"Finish y[0:5]: {y[0:5]}")
    return y


def get_performance_metrics(who, y_trues, y_preds, y_probas, save_dir=None):
    """
    Get and log useful performance metrics for a given data split (designated by 'who' parameter)
    """
    assert who in [
        "train",
        "test",
        "val",
        "validation",
    ], f"Expected one of train, test, val, or validation but got '{who}'"
    y_trues = prepare_labels(y_trues)
    y_preds = prepare_labels(y_preds)
    y_probas = np.asarray(y_probas)

    metric_dict = {
        f"{who}_acc": accuracy_score(y_trues, y_preds, normalize=True),
        f"{who}_balanced_acc": balanced_accuracy_score(y_trues, y_preds),
        f"{who}_roc_auc_score": roc_auc_score(y_trues, y_probas),
        f"{who}_average_precision_score": average_precision_score(y_trues, y_probas),
        f"{who}_f1_score": f1_score(y_trues, y_preds),
        f"{who}_confusion_matrix\n": confusion_matrix(y_trues, y_preds).tolist(),
        f"{who}_classification report\n": classification_report(y_trues, y_preds, output_dict=True),
    }

    logger.info(f"{who} set metrics:")
    for k, v in metric_dict.items():
        logger.info(f"{k}: {v}")

    if save_dir is not None:
        make_dir_if_needed(save_dir)
        p = os.path.join(save_dir, f"{who}_performance_metrics.json")
        logger.info(f"Saving dictionary of {who} set metrics to {p}")
        with open(p, "w") as json_file:
            json.dump(metric_dict, json_file)

    return metric_dict


def log_metrics_to_wandb(metric_dict):
    """
    Logs a dictionary of metrics to Weights & Biases summary.
    """
    if wandb.run is not None:
        for k, v in metric_dict.items():
            wandb.run.summary[k] = v
        logger.info("Logged metrics to W&B.")
    else:
        logger.warning("W&B is not initialized — skipping logging.")


def log_plots_to_wandb(who, y_trues, y_preds, y_probas_2col):
    """
    Log plots to W&B for the given dataset (train, test, validation).
    Args:
    - who: str, the dataset type (train, test, validation).
    - y_trues: array-like, true labels.
    - y_preds: array-like, predicted labels.
    - y_probas: array-like, predicted probabilities. NOTE: shape must be (n_samples, n_classes) for W&B.
    """
    y_trues = prepare_labels(y_trues)
    y_preds = prepare_labels(y_preds)

    logger.info(f"Logging plots to W&B for {who} set")
    wandb.log(
        {
            f"{who}_confusion_matrix_plot": wandb.plot.confusion_matrix(
                y_true=y_trues, preds=y_preds, class_names=["0", "1"]
            )
        }
    )
    wandb.log({f"{who}_precision_recall_plot": wandb.plot.pr_curve(y_true=y_trues, y_probas=y_probas_2col)})
    wandb.log({f"{who}_roc_auc_plot": wandb.plot.roc_curve(y_true=y_trues, y_probas=y_probas_2col)})
    return


def get_summary_metrics_wandb(model, x_train, y_train, x_test, y_test):
    logger.info("Logging plot summary metrics to W&B")
    wandb.sklearn.plot_summary_metrics(model, x_train, y_train, x_test, y_test)
    return


def get_train_test_manual_split(x, y, train_inds, test_inds):
    """
    Get manual train-test split based on specified indices.

    Parameters:
    - x: Features (input data, e.g., NumPy array or pandas DataFrame).
    - y: Labels (output data, e.g., NumPy array or pandas Series).
    - train_inds: List of indices for the training set.
    - test_inds: List of indices for the testing set.

    Returns:
    - X_train: Features for the training set.
    - X_test: Features for the testing set.
    - y_train: Labels for the training set.
    - y_test: Labels for the testing set.
    """
    logger.info("CAUTION: this `get_train_test_manual_split` is an untested function. TODO: check functionality.")

    # Assuming 'x' is a NumPy array or pandas DataFrame
    X_train = x[train_inds]
    X_test = x[test_inds]

    # Assuming 'y' is a NumPy array or pandas Series
    y_train = y[train_inds]
    y_test = y[test_inds]

    return X_train, X_test, y_train, y_test


def get_model_preds_and_probs(model, who, model_type="pnet", pnet_dataset=None, x=None, verbose=False):
    logger.info(f"Model_type = {model_type}. Computing model predictions on {who} set")
    if model_type == "pnet":
        y_preds, y_probas = get_pnet_preds_and_probs(model, pnet_dataset)
    elif model_type in ["rf", "bdt"]:
        y_preds, y_probas = get_sklearn_model_preds_and_probs(model, x)
    else:
        logger.error(f"We haven't implemented for the model type you specified, which was {model_type}")
    if verbose:
        logger.info(f"Hist of model prediction probabilities on {who} set")
        plt.hist(y_probas)
        plt.show()
    return y_preds, y_probas


def get_sklearn_model_preds_and_probs(sklearn_model, x):
    preds = sklearn_model.predict(x)
    pred_probs = sklearn_model.predict_proba(x)

    logger.debug(f"Shape of preds: {preds.shape}")
    logger.debug(f"Shape of pred_probas: {pred_probs.shape}")
    logger.debug(f"Type of preds: {type(preds)}")
    logger.debug(f"Type of pred_probas: {type(pred_probs)}")

    return preds, pred_probs


def get_sklearn_feature_importances(sklearn_model, who, input_df, save_dir=None):
    importances = sklearn_model.feature_importances_
    gene_feature_importances = pd.Series(
        importances, index=input_df.columns
    )  # TODO: check if this is the correct index
    # TODO: edit so that it's a better format when saved down
    logger.debug(
        "Editing DF format so that we just have two columns to save down. This makes the read-in format much nicer."
    )
    gene_feature_importances = gene_feature_importances.reset_index()
    gene_feature_importances.columns = ["feature", "importance score"]
    if save_dir is not None:
        make_dir_if_needed(save_dir)
        logger.info(f"Saving feature importance information to {save_dir}")
        gene_feature_importances.to_csv(os.path.join(save_dir, f"{who}_gene_feature_importances.csv"), index=False)
        # wandb.save(f'{who}_gene_feature_importances.csv', base_path=save_dir, policy="end") # TODO: problem. save_dir is above current dir, and this isn't allowed.
    return gene_feature_importances


def get_pnet_feature_importances(model, who, pnet_dataset, save_dir=None):
    """
    Args:
    - model: this is a Pnet model object
    - who: train, test, validation, or val. Which dataset are you using?
    - pnet_dataset: this is a Pnet dataset object (not a DF), and has attributes x, additional, y, etc.
    """
    logger.info(f"Getting feature importances for {who} set")
    gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = (
        model.interpret(pnet_dataset)
    )

    if save_dir is not None:
        make_dir_if_needed(save_dir)
        logger.info(f"Saving feature importance information to {save_dir}")
        gene_feature_importances.to_csv(os.path.join(save_dir, f"{who}_gene_feature_importances.csv"))
        additional_feature_importances.to_csv(os.path.join(save_dir, f"{who}_additional_feature_importances.csv"))
        gene_importances.to_csv(os.path.join(save_dir, f"{who}_gene_importances.csv"))
        for i, layer in enumerate(layer_importance_scores):
            layer.to_csv(os.path.join(save_dir, f"{who}_layer_{i}_importances.csv"))

    return gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores


def save_predictions_and_probs(save_dir, who, y_true, y_preds, y_probas, indices=None):
    """
    Save predictions, probabilities, true labels, and optionally sample indices for analysis.

    Parameters:
        save_dir (str): Directory to save the results.
        who (str): Identifier for the dataset (e.g., 'test', 'validation').
        y_true (array-like): True labels.
        y_preds (array-like): Predicted labels.
        y_probas (array-like): Predicted probabilities.
        indices (array-like, optional): Indices of the samples.
    """
    # Ensure formatted as numpy array
    y_true = np.asarray(y_true)
    y_preds = np.asarray(y_preds)
    y_probas = np.asarray(y_probas)

    # Extract probability of class 1 (we're assuming binary classification)
    y_probas = y_probas[:, 1] if y_probas.ndim > 1 and y_probas.shape[1] == 2 else y_probas
    results_dict = {
        "true": y_true.squeeze(),
        "predicted": y_preds.squeeze(),
        "probability": y_probas.squeeze(),
    }

    if indices is not None:
        results_dict["index"] = indices

    results_df = pd.DataFrame(results_dict)
    results_path = os.path.join(save_dir, f"{who}_predictions_and_probs.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved predictions, probabilities, and indices to {results_path}.")
    return results_df


def evaluate_interpret_save(model, who, model_type, pnet_dataset=None, x=None, y=None, save_dir=None):
    """
    For a given trained model of type `model_type' (e.g. P-NET, RF, BDT), get the model predictions, performance metrics, feature importances, and (optionally) save the results.
    The model is evaluated on the `dataset`.
    """
    if save_dir is not None:
        make_dir_if_needed(save_dir)
        logger.info(f"Results will be saved to {save_dir}.")

    # TODO: need a universal way to get the X vs y components of the `pnet_dataset`
    if model_type == "pnet":
        y = pnet_dataset.y.detach().cpu().squeeze().numpy()
        logger.info(
            f"Getting the {model_type} model predictions on the {who} set, performance metrics, and feature importances (if applicable)"
        )
        y_preds, y_probas = get_model_preds_and_probs(
            model=model, pnet_dataset=pnet_dataset, who=who, model_type=model_type
        )
        logger.debug(f"Shape of y_preds: {y_preds.shape}")
        logger.debug(f"Shape of y_probas: {y_probas.shape}")
        logger.debug(f"Shape of y: {y.shape}")
        logger.debug(f"Type of y_preds (should be numpy): {type(y_preds)}")
        logger.debug(f"Type of y_probas (should be numpy): {type(y_probas)}")
        logger.debug(f"Type of y (should be numpy): {type(y)}")

        save_predictions_and_probs(
            save_dir, who, y, y_preds, y_probas, indices=pnet_dataset.input_df.index.tolist()
        )  # TODO: this is universal, and should get pulled out

        metric_dict = get_performance_metrics(
            who, y, y_preds, y_probas, save_dir
        )  # TODO: this is universal, and should get pulled out
        log_metrics_to_wandb(metric_dict)

        if who != "train":
            gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = (
                get_pnet_feature_importances(model, who, pnet_dataset, save_dir)
            )

            proba_2col = np.stack([1 - y_probas, y_probas], axis=1)
            log_plots_to_wandb(who, y, y_preds, proba_2col)
        elif who == "train":
            logger.warn(
                "Skipping feature importances for the training set. Causing runs to crash due to OOM errors, and don't think I need these anyway."
            )
            return metric_dict, None, None, None, None

        return (
            metric_dict,
            gene_feature_importances,
            additional_feature_importances,
            gene_importances,
            layer_importance_scores,
        )

    elif model_type in ["rf", "bdt"]:
        logger.info(
            f"Getting the {model_type} model predictions on the {who} set, performance metrics, and feature importances (if applicable)"
        )
        x = pnet_dataset.x
        y = pnet_dataset.y.ravel().numpy()
        input_df = pnet_dataset.input_df

        y_preds, y_probas = get_model_preds_and_probs(model=model, x=x, who=who, model_type=model_type)

        save_predictions_and_probs(save_dir, who, y, y_preds, y_probas[:, 1], indices=input_df.index.tolist())
        metric_dict = get_performance_metrics(who, y, y_preds, y_probas[:, 1], save_dir)
        log_metrics_to_wandb(metric_dict)
        gene_feature_importances = get_sklearn_feature_importances(model, who=who, input_df=input_df, save_dir=save_dir)
        if who != "train":
            log_plots_to_wandb(who, y, y_preds, y_probas)
        return gene_feature_importances

    else:
        logger.error(f"We haven't implemented for the model type you specified, which was {model_type}")
    return


def save_as_file_to_wandb(data, filename, policy="now", delete_local=True):
    logger.info(f"Temporarily save down to {filename}, upload to WandB.")
    if not isinstance(data, pd.DataFrame):
        logger.warn(f"Expected a DF as input; converting {type(data)} object to pandas DF before saving.")
        data = pd.DataFrame(data)
    data.to_csv(filename)
    wandb.save(filename, policy=policy)
    if delete_local:
        logger.info(f"Deleting the temporary file at {filename}")
        os.remove(filename)
    return


def get_deviance(clf, x_test, y_test):
    """
    TODO: check functionality for RF. Works for BDT.
    Thinking about the "Plot training deviance" section from https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
    """
    test_score = np.zeros((clf.n_estimators_,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict(x_test)):
        test_score[i] = mean_squared_error(y_test, y_pred)
    return clf.train_score_, test_score
