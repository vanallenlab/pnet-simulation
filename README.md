# Pnet
Implementation of P-Net as a flexible deep learning tool to generate insights from genetic features.

Current pytorch implementation in revision.

## Model
Pnet uses the Reactome hierarchical graph as underlying structure to reduce the number of connections in a fully connected feed forward neural network. The sparse layers connect only known pathways. This limits the number of parameters to be learnt in a meaningful way and facilitate learning via gradient descent and leads to more generalizable models. 


## Application
Pnet can be used to predict target variables from genetic or transcriptional data. Example modalities include mutation data, copy number data, expression data. We are currently working on providing a exhaustive example notebook as well as a conda environment to install. In the meantime, users can check out the [prostate validation notebook](https://github.com/vanallenlab/pnet/blob/main/src/prostate_validation.ipynb). 