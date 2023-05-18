![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

# An industrial evaluation of proteochemometric modelling: predicting drug-target affinities for kinases 
GraphDTA (https://github.com/thinng/GraphDTA) was adapted for the purpose of an industrial evaluation of deep learning proteochemometric models as part of the paper "An industrial evaluation of proteochemometric modelling: predicting drug-target affinities for kinases"

## Installation
For python version 3.8, cuda version 11.3 and pytorch version 1.12.0, create a conda environment:

```sh
conda env create -f environment.yml
```

Activate the environment:

```sh
conda activate GraphDTAadapted
```
OR: Install Python libraries needed
Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
Install rdkit: conda install -y -c conda-forge rdkit
Install networkx: pip install networkx
Install prettytable: pip install prettytable


## Example usage
Training the model:
```console
python training.py -h 
usage: training.py [-h] run data params

positional arguments:
  run         Name of the run.
  data        Name of the data.
  params      Config file.

python training.py  'test' 'BindingDB/RandomLigandSplit/Fold0' 'params.json'
```
Testing:
```console
python testing.py -h
usage: testing.py [-h] model_dir run data params

positional arguments:
  model_dir   model directory
  run         Name of the run.
  data        Name of the data.
  params      Config file.
```
