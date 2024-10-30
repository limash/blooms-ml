# blooms-ml

[![Report](https://img.shields.io/badge/Report-8A2BE2)](https://limash.github.io/blooms-ml/)

Identification of phytoplankton blooms using machine learning methods.

## Overview

The description can be found in the [report](https://limash.github.io/blooms-ml/).
The implementation of machine learning methods is based on [JAX](https://github.com/google/jax/blob/main/README.md)/[FLAX](https://github.com/google/flax/blob/main/README.md).

## Quick install

Install and activate a python environment, for example:

```bash
python3 -m venv venv/blooms-ml
source venv/blooms-ml/bin/activate
```

Then install blooms-ml:

```bash
git clone https://github.com/limash/blooms-ml
cd blooms-ml
pip install -e .[ml-cpu]
```
