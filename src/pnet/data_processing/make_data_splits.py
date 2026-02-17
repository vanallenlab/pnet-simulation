import logging
import os
import pickle
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, "../../..")  # add project_config to path
import project_config

logging.basicConfig(
    encoding="utf-8",
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="make_data_splits.log",
)
logger = logging.getLogger(__name__)

harmonized_data_dir = "/mnt/disks/gmiller_data1/pnet_germline/processed/wandb-group-data_prep_germline_tier12_and_somatic/converted-IDs-to-somatic_imputed-germline_True_imputed-somatic_False_paired-samples-True/wandb-run-id-u5yt90p1"
data_split_dir = project_config.DATA_DIR / "pnet_database/prostate/splits"

# load labels
y = pd.read_csv(os.path.join(harmonized_data_dir, "y.csv"), index_col=0)
y.rename(columns={"is_met": "class"}, inplace=True)

# desired sizes
N_TRAIN, N_VAL, N_TEST = 759, 89, 95
N_SPLITS = 10
SEED_BASE = 123
logger.info(
    f"Going to make {N_SPLITS} splits: {N_TRAIN} train, {N_VAL} val, {N_TEST} test samples."
)

ids = y.index.to_numpy()
labels = y["class"].to_numpy()

splits = []
for split_id in range(N_SPLITS):
    seed = SEED_BASE + split_id

    # 1) test set (size 95)
    trainval_ids, test_ids, trainval_y, _ = train_test_split(
        ids,
        labels,
        test_size=N_TEST,
        stratify=labels,
        random_state=seed,
    )

    # 2) val set (size 89) from remaining trainval pool (size 848)
    val_frac_of_trainval = N_VAL / (N_TRAIN + N_VAL)  # 89 / 848
    train_ids, val_ids = train_test_split(
        trainval_ids,
        test_size=val_frac_of_trainval,
        stratify=trainval_y,
        random_state=seed,
    )

    splits.append(
        {
            "split_id": split_id,
            "train_ids": list(train_ids),
            "val_ids": list(val_ids),
            "test_ids": list(test_ids),
        }
    )

out = {
    "scheme": "repeated_stratified_train_val_test",
    "sizes": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST, "total": len(ids)},
    "n_splits": N_SPLITS,
    "seed_base": SEED_BASE,
    "splits": splits,
}

os.makedirs(data_split_dir, exist_ok=True)
out_path = os.path.join(
    data_split_dir,
    f"splits_{N_SPLITS}x_train{N_TRAIN}_val{N_VAL}_test{N_TEST}_seed{SEED_BASE}.pkl",
)

with open(out_path, "wb") as f:
    pickle.dump(out, f)

logger.info(f"Saved splits to: {out_path}")
