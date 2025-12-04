# Script to prepare each data modality to be directly loaded into P-NET

"""
Here, we prepare our inputs so they can be easily loaded into P-NET (and other models) without requiring any additional work beyond creating train/test/val splits.
Specifically, we take care of issues including:
1. Harmonizing the IDs
2. Performing imputation as necessary (to keep non-overlapping genes)

Load each of your data modalities of interest.
Format should be samples x genes. Set the sample IDs as the index.

Data modalities:
1. somatic amp
1. somatic del
1. somatic mut
1. germline mut (subset to a small number of genes).

Our somatic data has information for many more genes compared to the germline data. We will need to either:
1. impute zeros for the excluded germline genes, or
2. subset the somatic datasets down to the ones that overlap with the germline data.


We will be subsetting to the 943 samples that we have matched somatic and germline data for.
"""

import argparse
import logging
import os
import sys

import wandb
from pnet import report_and_eval
from pnet.data_processing import data_manipulation, prostate_data_loaders

sys.path.insert(0, "../../..")  # add project_config to path
import project_config


def configure_logging(log_file="prep_vcfs_as_model_input.log"):
    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare the P1000 somatic data with germline data as model inputs. Harmonize the identifiers used, and optionally zero-impute non-overlapping genes to preserve the union of genes."
    )
    parser.add_argument("--wandb_group", default="", help="Wandb group name")
    parser.add_argument("--use_only_paired", action="store_true", help="Use only paired")
    parser.add_argument("--convert_ids_to", default="somatic", help="Convert IDs to")
    parser.add_argument("--zero_impute_germline", action="store_true", help="Zero impute germline")
    parser.add_argument("--zero_impute_somatic", action="store_true", help="Zero impute somatic")
    parser.add_argument(
        "--somatic_datadir",
        default=project_config.PNET_DATABASE_DIR
        / "prostate/processed",  # directory containing the P1000 somatic data as published in the original P-NET paper
        help="Somatic data directory",
    )
    parser.add_argument(
        "--germline_datadir",
        default=project_config.GERMLINE_DATA_DIR,
        help="Germline data directory containing the germline datasets that require harmonization with the somatic data",
    )
    parser.add_argument(
        "--input_data_wandb_id",
        default="",
        help="W&B run ID that created the data in the input_data_dir, if applicable",
    )
    parser.add_argument(
        "--save_dir", default=project_config.PROCESSED_DATA_DIR, help="Directory storing model-ready input"
    )
    return parser.parse_args()


def initialize_wandb(wandb_group):
    wandb.login()
    wandb.init(project="prostate_met_status", group=wandb_group)
    return wandb.run.id


def log_parameters_to_wandb(params):
    logging.info("Adding parameters to Weights and Biases")
    wandb.config.update(params)


def load_data(somatic_datadir, germline_datadir):
    somatic_datasets = {
        "somatic_mut": prostate_data_loaders.get_somatic_mutation(
            os.path.join(somatic_datadir, "P1000_final_analysis_set_cross_important_only.csv")
        ),
        "somatic_amp": prostate_data_loaders.get_somatic_amp_and_del(
            os.path.join(somatic_datadir, "P1000_data_CNA_paper.csv")
        )[0],
        "somatic_del": prostate_data_loaders.get_somatic_amp_and_del(
            os.path.join(somatic_datadir, "P1000_data_CNA_paper.csv")
        )[1],
    }

    processed_germline_vcfs_dir = os.path.join(germline_datadir, "processed_germline_vcfs")
    germline_datasets = {
        "germline_rare_lof": prostate_data_loaders.get_germline_mutation(
            os.path.join(
                processed_germline_vcfs_dir,
                "prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_patho-vars-only_rare_high-impact.txt",
            )
        ),
        "germline_rare_missense": prostate_data_loaders.get_germline_mutation(
            os.path.join(
                processed_germline_vcfs_dir,
                "prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_patho-vars-only_rare_moderate-impact.txt",
            )
        ),
        # "germline_common_lof": prostate_data_loaders.get_germline_mutation(
        #     os.path.join(
        #         processed_germline_vcfs_dir,
        #         "prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_patho-vars-only_common_high-impact.txt",
        #     )
        # ),
        "germline_common_missense": prostate_data_loaders.get_germline_mutation(
            os.path.join(
                processed_germline_vcfs_dir,
                "prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_patho-vars-only_common_moderate-impact.txt",
            )
        ),
        "germline_rare_common_lof": prostate_data_loaders.get_germline_mutation(
            os.path.join(
                processed_germline_vcfs_dir,
                "prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_patho-vars-only_rare_common_high-impact.txt",
            )
        ),
        "germline_rare_common_missense": prostate_data_loaders.get_germline_mutation(
            os.path.join(
                processed_germline_vcfs_dir,
                "prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_patho-vars-only_rare_common_moderate-impact.txt",
            )
        ),
        "germline_rare_lof_missense": prostate_data_loaders.get_germline_mutation(
            os.path.join(
                processed_germline_vcfs_dir,
                "prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_patho-vars-only_rare_high-impact_moderate-impact.txt",
            )
        ),
        "germline_common_lof_missense": prostate_data_loaders.get_germline_mutation(
            os.path.join(
                processed_germline_vcfs_dir,
                "prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_patho-vars-only_common_high-impact_moderate-impact.txt",
            )
        ),
        "germline_rare_common_lof_missense": prostate_data_loaders.get_germline_mutation(
            os.path.join(
                processed_germline_vcfs_dir,
                "prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_patho-vars-only_rare_common_high-impact_moderate-impact.txt",
            )
        ),
    }
    return somatic_datasets, germline_datasets


def harmonize_ids(somatic_datasets, germline_datasets, additional, y, convert_ids_to):
    logging.info(f"Harmonizing IDs (switching to {convert_ids_to} IDs)")

    germline_list = list(germline_datasets.values())
    somatic_list = list(somatic_datasets.values())

    germline_list, somatic_list = prostate_data_loaders.harmonize_prostate_ids(
        datasets_w_germline_ids=list(germline_datasets.values()),
        datasets_w_somatic_ids=list(somatic_datasets.values()) + [additional, y],
        convert_ids_to=convert_ids_to,
    )

    germline_datasets = {key: value for key, value in zip(germline_datasets.keys(), germline_list)}
    somatic_datasets = {key: value for key, value in zip(somatic_datasets.keys(), somatic_list)}

    return germline_datasets, somatic_datasets


def restrict_to_paired_samples(somatic_datasets, germline_datasets, additional, y):
    logging.info("Restricting to overlapping samples (the rows/indices)")
    restricted_datasets = data_manipulation.restrict_to_overlapping_indices(
        *somatic_datasets.values(), *germline_datasets.values(), additional, y
    )
    keys = list(somatic_datasets.keys()) + list(germline_datasets.keys()) + ["additional", "y"]
    return dict(zip(keys, restricted_datasets))


def zero_impute_datasets(germline_datasets, somatic_datasets, zero_impute_germline, zero_impute_somatic):
    logging.info(
        f"Zero-imputing columns (genes) as defined by user (impute germline: {zero_impute_germline}, impute somatic: {zero_impute_somatic})"
    )
    germline_list = list(germline_datasets.values())
    somatic_list = list(somatic_datasets.values())

    if zero_impute_germline:
        germline_list = prostate_data_loaders.zero_impute_germline_datasets(
            germline_datasets=germline_list,
            somatic_datasets=somatic_list,
            zero_impute_germline=zero_impute_germline,
        )

    if zero_impute_somatic:
        somatic_list = prostate_data_loaders.zero_impute_somatic_datasets(
            germline_datasets=germline_list,
            somatic_datasets=somatic_list,
            zero_impute_somatic=zero_impute_somatic,
        )

    germline_datasets = {key: value for key, value in zip(germline_datasets.keys(), germline_list)}
    somatic_datasets = {key: value for key, value in zip(somatic_datasets.keys(), somatic_list)}
    return germline_datasets, somatic_datasets


def restrict_to_common_genes(somatic_datasets, germline_datasets):
    all_datasets = list(somatic_datasets.values()) + list(germline_datasets.values())
    restricted_datasets = prostate_data_loaders.restrict_to_genes_in_common(*all_datasets)
    keys = list(somatic_datasets.keys()) + list(germline_datasets.keys())
    return dict(zip(keys, restricted_datasets))


def process_data(somatic_datasets, germline_datasets, additional, y, args):
    if args.convert_ids_to:
        # harmonize the IDs of the datasets
        germline_datasets, somatic_datasets = harmonize_ids(
            somatic_datasets, germline_datasets, additional, y, args.convert_ids_to
        )
    if args.use_only_paired:
        # restrict DFs to overlapping samples
        logging.debug(f"Somatic dataset indices: {[df.index.tolist() for df in somatic_datasets.values()]}")
        logging.debug(f"Germline dataset indices: {[df.index.tolist() for df in germline_datasets.values()]}")
        logging.debug(f"Additional dataset indices: {additional.index.tolist()}")
        logging.debug(f"Target dataset indices: {y.index.tolist()}")
        datasets = restrict_to_paired_samples(somatic_datasets, germline_datasets, additional, y)
        somatic_datasets = {k: datasets[k] for k in somatic_datasets}
        germline_datasets = {k: datasets[k] for k in germline_datasets}
        additional, y = datasets["additional"], datasets["y"]

    # perform zero-imputation according to args
    germline_datasets, somatic_datasets = zero_impute_datasets(
        germline_datasets,
        somatic_datasets,
        args.zero_impute_germline,
        args.zero_impute_somatic,
    )
    # restrict DFs to genes in common, and also restrict to TCGA prostate genes
    all_datasets = restrict_to_common_genes(somatic_datasets, germline_datasets)
    somatic_datasets = {k: all_datasets[k] for k in somatic_datasets}
    germline_datasets = {k: all_datasets[k] for k in germline_datasets}

    # return the updated DFs
    return somatic_datasets, germline_datasets, additional, y


def save_datasets(datasets, save_dir):
    report_and_eval.make_dir_if_needed(save_dir)
    for name, df in datasets.items():
        save_path = os.path.join(save_dir, f"{name}.csv")
        df.to_csv(save_path, index=True)
        wandb.config.update({f"{name}_output_f": save_path})


def main():
    args = parse_arguments()
    configure_logging()
    wandb_run_id = initialize_wandb(args.wandb_group)

    SAVE_DIR = os.path.join(
        args.save_dir,
        f"wandb-group-{args.wandb_group}/converted-IDs-to-{args.convert_ids_to}_imputed-germline_{args.zero_impute_germline}_imputed-somatic_{args.zero_impute_somatic}_paired-samples-{args.use_only_paired}/wandb-run-id-{wandb_run_id}",
    )

    logging.info("Loading data")
    somatic_datasets, germline_datasets = load_data(args.somatic_datadir, args.germline_datadir)
    y = prostate_data_loaders.get_target(
        os.path.join(args.germline_datadir, "metadata/germline_somatic_id_map_outer_join.csv"),
        os.path.join(
            args.germline_datadir,
            "metadata/pathogenic_variants_with_clinical_annotation_1341_aug2021_correlation.csv",
        ),
        id_to_use="Tumor_Sample_Barcode",
        target_col="is_met",
    )
    additional = prostate_data_loaders.get_additional_data(
        os.path.join(
            args.germline_datadir,
            "metadata/pathogenic_variants_with_clinical_annotation_1341_aug2021_correlation.csv",
        ),
        os.path.join(args.germline_datadir, "metadata/germline_somatic_id_map_outer_join.csv"),
        cols_to_include=[
            "PCA1",
            "PCA2",
            "PCA3",
            "PCA4",
            "PCA5",
            "PCA6",
            "PCA7",
            "PCA8",
            "PCA9",
            "PCA10",
        ],
    )

    params = {
        "wandb_run_id_that_created_inputs": args.input_data_wandb_id,
        "zero_impute_germline": args.zero_impute_germline,
        "zero_impute_somatic": args.zero_impute_somatic,
        "restricted_to_pairs": args.use_only_paired,
        "somatic_datadir": args.somatic_datadir,
        "germline_datadir": args.germline_datadir,
        "save_dir": SAVE_DIR,
        "id_map_f": os.path.join(args.germline_datadir, "metadata/germline_somatic_id_map_outer_join.csv"),
        "sample_metadata_f": os.path.join(
            args.germline_datadir,
            "metadata/pathogenic_variants_with_clinical_annotation_1341_aug2021_correlation.csv",
        ),
    }
    log_parameters_to_wandb(params)

    logging.info("Processing datasets")
    somatic_datasets, germline_datasets, additional, y = process_data(
        somatic_datasets, germline_datasets, additional, y, args
    )

    logging.info("Printing some basic info for each of our datasets")
    df_dict = {
        **somatic_datasets,
        **germline_datasets,
        "additional": additional,
        "y": y,
    }
    report_and_eval.report_df_info_with_names(df_dict, n=5)

    logging.info(f"Saving each DF to {SAVE_DIR}")
    report_and_eval.make_dir_if_needed(SAVE_DIR)
    save_datasets(df_dict, SAVE_DIR)

    logging.info("Ending wandb run")
    wandb.finish()
    return wandb_run_id


if __name__ == "__main__":
    main()
