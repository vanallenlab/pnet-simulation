# Process for making model-ready data

## Description of process
1. First, ensure that you've updated file paths in `notebooks/preprocessing/pipeline_full_vcf_to_germline_patho_vcfs.ipynb` and `src/pnet/preprocessing/prep_vcfs_as_model_inputs.py`.

2. Code to go from full VCF all the way to variant-subsetted VCF is in `notebooks/preprocessing/pipeline_full_vcf_to_germline_patho_vcfs.ipynb`. This does gene subsetting, then applies universal variant quality filters, then germline pathogenicity filters, then makes variant-subsetted VCFs (rare vs common, LOF vs missense). This script replaces `make_vcf_subsets.py` and `make_germline_vcf_subsets.py`.


3. To harmonize the P1000 somatic and germline datasets, run `src/pnet/preprocessing/prep_vcfs_as_model_inputs.py`. This ensures a common set of identifiers are used across the P1000 somatic and germline datasets, and zero-imputes genes as needed to preserve the union of all genes.


