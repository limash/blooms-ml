# blooms-ml
Identification of phytoplankton blooms from observational data.

## Overview
The repo contains two parts:
1. Data preprocessing:
The modelled data is used to generate observations for training.
Essentially these are nutrients point values and density profiles data.
The modelled data comes from ROMS+NERSEM models.
2. Training:
[JAX](https://github.com/google/jax/blob/main/README.md)/[FLAX](https://github.com/google/flax/blob/main/README.md) based routines.

## Quick install
Install and activate a python environment, for example:
```
python3 -m venv venv/blooms-ml
source venv/blooms-ml/bin/activate
```

Then install blooms-ml:
```
git clone https://github.com/limash/blooms-ml
cd blooms-ml
pip install -e .[ml-cpu]
```
