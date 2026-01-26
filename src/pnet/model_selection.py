"""
Script containing functions for running different model types.
E.g. P-NET, RF, Boosted Decision Tree, etc.

Each function currently takes
Inputs:
Train/test indices, data, y

Outputs:
trained model, train and test datasets (P-NET returns a P-NET data object)
"""

import logging

from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from pnet import Pnet

logger = logging.getLogger(__name__)


def run_pnet(genetic_data, y, train_inds, test_inds):
    logger.info("Running P-NET model")
    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
        genetic_data,
        y,
        verbose=False,
        early_stopping=False,
        train_inds=train_inds,
        test_inds=test_inds,
    )
    model.to("cpu")
    x_test = test_dataset.x
    additional_test = test_dataset.additional
    y_test = test_dataset.y
    pred = model(x_test, additional_test)
    y_pred_proba = pred[0].detach().numpy().squeeze()

    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    gene_imp = model.gene_importance(test_dataset)
    layerwise_imp = model.layerwise_importance(test_dataset)
    logger.debug(
        f"gene_imp: \n{model.gene_importance(test_dataset)}\nlayerwise_imp: \n{model.layerwise_importance(test_dataset)}\nauc: \n{metrics.roc_auc_score(y_test, y_pred_proba)}"
    )

    return (
        model,
        train_dataset,
        test_dataset,
        train_scores,
        test_scores,
        auc,
        gene_imp,
        layerwise_imp,
    )


def run_rf(
    x_train,
    y_train,
    max_depth=None,
    random_seed=None,
    min_samples_split=2,
    n_estimators=100,
    min_samples_leaf=1,
):
    logger.info("Running Random Forest (RF) model")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(x_train, y_train)
    return model


def run_bdt(x_train, y_train, max_depth=3, n_estimators=100, lr=0.1, random_seed=None):
    logger.info("Running Boosted Decision Tree (BDT) model")
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=max_depth,
        random_state=random_seed,
    )
    model.fit(x_train, y_train)
    return model


def run_logistic_regression(
    x_train,
    y_train,
    penalty="l2",
    C=1.0,
    l1_ratio=None,
    max_iter=1000,
    random_seed=None,
):
    """
    Run Logistic Regression model with optional L1 or elastic net regularization.

    Args:
        x_train: Training feature data
        y_train: Training labels
        penalty: Type of regularization - "l1", "l2", or "elasticnet" (default: "l2")
        C: Inverse of regularization strength (default: 1.0). Lower values indicate stronger regularization.
        l1_ratio: Balance parameter for elastic net (0 to 1). Only used when penalty="elasticnet".
                 0 = pure L2, 1 = pure L1 (default: 0.5)
        max_iter: Maximum number of iterations for solver convergence (default: 1000)
        random_seed: Random seed for reproducibility (default: None)

    Returns:
        Trained LogisticRegression model
    """
    logger.info(f"Running Logistic Regression model with penalty type ={penalty}")

    solver = (
        "saga" if penalty == "elasticnet" else "saga"
    )  # saga supports all penalty types

    model = LogisticRegression(
        penalty=penalty,
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        max_iter=max_iter,
        random_state=random_seed,
    )
    model.fit(x_train, y_train)
    return model


def run_logistic_regression_sgd(
    x_train,
    y_train,
    loss="log_loss",
    penalty="l2",
    C=1.0,
    alpha=None,
    l1_ratio=None,
    class_weight=None,
    random_seed=None,
):
    if alpha is not None:
        logger.info(
            f"Using provided alpha={alpha} for SGD logistic regression with penalty={penalty}, class_weight={class_weight}"
        )
    else:
        # Calculate alpha from C
        n_samples = x_train.shape[0]
        alpha = 1.0 / (n_samples * C)
        logger.info(
            f"Running SGD logistic regression with penalty={penalty}, user-provided C={C} (alpha={alpha}), class_weight={class_weight}"
        )

    model = SGDClassifier(
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        random_state=random_seed,
    )

    model.fit(x_train, y_train)
    return model
