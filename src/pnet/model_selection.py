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

from pnet import Pnet

logger = logging.getLogger(__name__)


def run_pnet(genetic_data, y, train_inds, test_inds):
    logger.info("Running P-NET model")
    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
        genetic_data, y, verbose=False, early_stopping=False, train_inds=train_inds, test_inds=test_inds
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

    return model, train_dataset, test_dataset, train_scores, test_scores, auc, gene_imp, layerwise_imp


def run_rf(
    x_train, y_train, max_depth=None, random_seed=None, min_samples_split=2, n_estimators=100, min_samples_leaf=1
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
        n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, random_state=random_seed
    )
    model.fit(x_train, y_train)
    return model
