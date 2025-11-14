import logging
import os

import configargparse
import pandas as pd
import yaml

import wandb
from pnet import Pnet, model_selection, report_and_eval

logging.basicConfig(
    encoding="utf-8",
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

logger = logging.getLogger(__name__)


# adapted from https://github.com/wandb/wandb/issues/2939 to help W&B sweep
class ParseAction(configargparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = list(map(str, values.split()))
        setattr(namespace, self.dest, values)


def none_or_float(value):
    if value == "None" or value is None:
        return None
    return float(value)


def read_config(filename):
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config


def get_train_val_test_indices(
    split_dir, train_ind_f="training_set.csv", validation_ind_f="validation_set.csv", test_ind_f="test_set.csv"
):
    """
    Load train, validation, and test indices from CSV files in the specified directory.
    The CSV files should contain 'id' and 'response' columns.
    """
    train_inds = pd.read_csv(
        os.path.join(split_dir, train_ind_f), usecols=["id", "response"], index_col="id"
    ).index.tolist()
    validation_inds = pd.read_csv(
        os.path.join(split_dir, validation_ind_f), usecols=["id", "response"], index_col="id"
    ).index.tolist()
    test_inds = pd.read_csv(
        os.path.join(split_dir, test_ind_f), usecols=["id", "response"], index_col="id"
    ).index.tolist()

    return train_inds, validation_inds, test_inds


def get_train_eval_indices(split_dir, eval_set):
    train_f = os.path.join(split_dir, "training_set.csv")
    eval_f = os.path.join(split_dir, f"{eval_set}_set.csv")

    train_ids = pd.read_csv(train_f, usecols=["id", "response"], index_col="id").index.tolist()
    eval_ids = pd.read_csv(eval_f, usecols=["id", "response"], index_col="id").index.tolist()

    return train_ids, eval_ids, train_f, eval_f


def setup_save_dir(model_type, eval_set, wandb_group, run_id, base_dir="../results"):
    base = os.path.join(base_dir, f"{model_type}_eval_set_{eval_set}/wandbID_{run_id}")
    if wandb_group:
        base = os.path.join(base_dir, f"{wandb_group}/{model_type}_eval_set_{eval_set}/wandbID_{run_id}")
    report_and_eval.make_dir_if_needed(base)
    logger.debug(f"Save directory: {base}")
    return base


# def train_model_rf(train_dataset, min_samples_split, max_depth=None, random_seed=None, min_samples_leaf=1):
def train_model_rf(train_dataset, min_samples_split=None, max_depth=None, min_samples_leaf=1, random_seed=None):
    logger.info("Training Random Forest model")
    x_train, y_train = train_dataset.x, train_dataset.y.ravel()

    model = model_selection.run_rf(
        x_train,
        y_train,
        random_seed=random_seed,
        min_samples_split=min_samples_split,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    return model


def train_model_bdt(train_dataset, test_dataset, evaluation_set):
    logger.info("Training Gradient Boosting model")
    x_train, y_train = train_dataset.x, train_dataset.y.ravel()
    x_test, y_test = test_dataset.x, test_dataset.y.ravel()

    model = model_selection.run_bdt(x_train, y_train, random_seed=None)

    logger.info("Generating deviance plot to check convergence/overfitting")
    train_scores, test_scores = report_and_eval.get_deviance(model, x_test, y_test)
    plt = report_and_eval.get_loss_plot(
        train_losses=train_scores,
        test_losses=test_scores,
        train_label="Train deviance",
        test_label=f"{evaluation_set} deviance",
        title="Model Deviance",
        ylabel="Deviance (MSE)",
        xlabel="Boosting iterations",
    )
    wandb.log({"convergence plot": plt})
    report_and_eval.savefig(plt, os.path.join(wandb.run.dir, "deviance_per_boosting_iteration"))
    return model, train_scores, test_scores


def train_model_pnet(hparams, genetic_data, additional, y):
    logger.info("Training PNET model")
    model_save_path = os.path.join(hparams["save_dir"], "model.pt")
    model, train_losses, test_losses, train_dataset, test_dataset = Pnet.run(
        genetic_data,
        y,
        save_path=model_save_path,
        additional_data=additional,
        dropout=hparams["dropout"],
        input_dropout=hparams["input_dropout"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        batch_size=hparams["batch_size"],
        epochs=hparams["epochs"],
        verbose=hparams["verbose"],
        early_stopping=hparams["early_stopping"],
        train_inds=hparams["train_set_indices"],
        test_inds=hparams["evaluation_set_indices"],
    )

    logger.info("Logging loss curve")
    plt = report_and_eval.get_loss_plot(train_losses=train_losses, test_losses=test_losses)
    wandb.log({"convergence plot": plt})
    report_and_eval.savefig(plt, os.path.join(hparams["save_dir"], "loss_over_time"))

    return model, train_losses, test_losses, train_dataset, test_dataset, model_save_path


def evaluate_and_log_results(model, train_dataset, test_dataset, model_type, save_dir, eval_set_name):
    logger.info("Evaluating model on training and evaluation sets")
    report_and_eval.evaluate_interpret_save(
        model=model, pnet_dataset=train_dataset, model_type=model_type, who="train", save_dir=save_dir
    )
    report_and_eval.evaluate_interpret_save(
        model=model, pnet_dataset=test_dataset, model_type=model_type, who=eval_set_name, save_dir=save_dir
    )


def evaluate_on_train_val_test(model, train_dset, val_dset, test_dset, hparams):
    """
    Evaluate a trained model on train, val, and test splits using PnetDataset.

    Args:
        model: Trained model object
        genetic_data: Input feature data (e.g., DataFrame or tensor)
        y: Target labels (Pandas Series or compatible)
        train_dset, val_dset, test_dset: PnetDataset objects for train, validation, and test sets
        hparams: Dict of hyperparameters including model_type and save_dir
    """
    for split_name, split_dset in zip(["train", "validation", "test"], [train_dset, val_dset, test_dset]):
        logger.info(f"Evaluating model on {split_name} set")

        report_and_eval.evaluate_interpret_save(
            model=model,
            pnet_dataset=split_dset,
            model_type=hparams["model_type"],
            who=split_name,
            save_dir=hparams["save_dir"],
        )
    return


def cleanup(model_save_path, delete_model_after_training=False):
    """
    Delete the saved model if delete_model_after_training is True.
    """
    if delete_model_after_training:
        try:
            os.remove(model_save_path)
            logger.info(f"Deleted model file at: {model_save_path}")
        except OSError as e:
            logger.warning(f"Failed to delete model file: {e}")
    return
