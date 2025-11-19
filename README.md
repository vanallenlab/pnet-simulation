# Simulation and empirical evaluation of biologically-informed neural network performance
This codebase contains all the code associated with the paper (in submission).
This documentation is a work-in-progress and will continue to be updated.

## Installation and environment setup
1. Clone this git repository, and navigate inside.
2.  Make a conda env using <pnet.yaml>
	1. Run `conda env create -f pnet.yaml --name YOUR-ENV`
	2. (Optional) Make the environment visible to VSCode jupyter notebooks `ipython kernel install --user --name=YOUR-ENV`
3. Activate the conda env
```conda activate YOUR-ENV```
4. Install the 'pnet' package:
```bash
# From your repo root
python -m pip install -e .
```
5. (Optional) Set up nbstripout to clear jupyter notebook outputs when pushing to github: `conda install -c conda-forge nbstripout`
```bash
# From your repo root
nbstripout --uninstall           # removes the repo’s filter config
nbstripout --install --attributes .gitattributes
nbstripout --status
```


## Reproducing analyses
### Figure 1: Joint simulation with linear and nonlinear signal
For the joint simulation (also called "2D" sometimes since both linear and nonlinear signal are simulated jointly):  
1. Run model with `config_2D_*` yamls: `src/run_model_on_pure_simulated_data.py`
2. Analyze with `notebooks/analysis/get_2D_simulation_results.ipynb`. The Figure 1 panels will be saved to the directory specified in `project_config.py` as `FIGURE_1_DIR`.

### Figure 2: Single-gene spike-in simulation
For the single-gene spike-in (also called "1D" at times since simulating linear signal):  
1. Make & save perturbed datasets to a folder: `src/pnet/make_perturbed_genotype_datasets.py`
2. Make configs: `notebooks/make_config_files_for_single-gene_perturbed_data.ipynb`
3. Run model using configs: `src/pnet/run_model_on_perturbed_data.py`
4. Analyze with `notebooks/analysis/get_1D_simulation_results.ipynb`. The Figure 2 panels will be saved to the directory specified in `project_config.py` as `FIGURE_2_DIR`.


### Figure 3: Empirical assessment on the P1000 prostate cancer dataset (matched somatic and germline WES sequencing)
#### Data download and access
1. The somatic data was sourced from the original [P-NET paper](https://www.nature.com/articles/s41586-021-03922-4) [(GitHub)](https://github.com/marakeby/pnet_prostate_paper), and can be located here: https://zenodo.org/records/10775529.
2. The germline raw sequence data (BAM files) can be obtained through dbGaP (https://www.ncbi.nlm.nih.gov/gap) as described in Supplementary Table 1 from Armenia et al.. The germline data used in this manuscript can be generated based on Armenia et al. and AlDubayan, S.H. et al. with our methods section and code.
   
#### Data preprocessing
We used matched germline and somatic WES samples from a prostate cancer genomics dataset (hereafter “P1000” from Armenia et al.). 
1. Make filtered germline VCFs with `src/pipeline_full_vcf_to_germline_patho_vcfs.ipynb`. This script first subsets the germline data to a gene subset, and then performs quality and pathogenicity filtering at the variant level, and creates different subsets of the germline data.
	1. Output saved to the directory specified in `project_config.py` as `PROCESSED_GERMLINE_VCFS_DIR`
2. Make harmonized, model-ready datasets.
	1. Run command `python prep_vcfs_as_model_inputs.py --zero_impute_germline --use_only_paired --wandb_group data_prep_germline_tier12_and_somatic`

#### Modeling
For empirical P1000 results, after you have generated the model-ready datasets:
1. Run model according to `config_p1000_empirical.yaml`: `pnet/src/run_moded_on_preharmonized_data.py`
2. Analyze with `notebooks/analysis/get_P1000_simulation_results.ipynb`. The Figure 3 panels will be saved to the directory specified in `project_config.py` as `FIGURE_3_DIR`.

### Supplementary tables: additional performance metrics
1. After running the analyses for Figures 1-3, you can generate the supplementary tables with `notebooks/analysis/make_supplementary_tables.ipynb`. The supplementary tables will be saved to the directory specified in `project_config.py` as `SUPPLEMENTARY_TABLES_DIR`.

## References
- Elmarakeby, H. A. et al. Biologically informed deep neural network for prostate cancer discovery. Nature 598, 348–352 (2021).
  - Original P-NET model paper. Ran on the P1000 somatic WES data.
- Armenia, J. et al. The long tail of oncogenic drivers in prostate cancer. Nat. Genet. 50, 645–651 (2018).
  - Assembled the P1000 dataset, which consists of tumor and matched germline prostate cancer patient whole-exome sequencing (WES) samples.
- AlDubayan, S. H. et al. Detection of Pathogenic Variants With Germline Genetic Testing Using Deep Learning vs Standard Methods in Patients With Prostate Cancer and Melanoma. JAMA 324, 1957–1969 (2020).
  - Initial processing of the P1000 germline data (stringent quality control, detection of germline variants with DeepVariant18 (v0.6.0), functional annotation with Ensembl’s Variant Effect Predictor (VEP, v92.0), and genetic ancestry inference via principal component analysis)
- The P-NET model code is adapted from a PyTorch implementation of P-NET by Marc Glettig: https://github.com/vanallenlab/pnet.
