# Constraint-Aware Neural Networks

Code of the paper "Constraint-Aware Neural Networks for Riemann Problems" for training constraint-aware neural networks for various model problems, see [JCP](https://doi.org/10.1016/j.jcp.2020.109345) or [arxiv](https://doi.org/10.48550/arXiv.1904.12794).

## Installation

### Using uv

Install [uv](https://github.com/astral-sh/uv), for example via pipx:

        python -m pip install --user pipx
        pipx install uv

Then run uv to set everything up:

        uv venv
        uv pip install -e .[dev]

### Using pip

Install all dependencies via [pip](https://pip.pypa.io/en/stable/) from the project directory

        pip install .[dev]

### Post-Installation Steps

Setup [git-lfs](https://git-lfs.com/) and [pre-commit](https://pre-commit.com/hooks.html) hooks:

        git lfs install  
        pre-commit install

## Replication Data

The replication data is available at
[DaRUS](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3869),
[doi: 10.18419/darus-3869](https://doi.org/10.18419/darus-3869).
Download it and put it into `data/`.

## Usage

Run the CLI-tool:

        uv run constraint_aware_neural_networks train --help

Example:

        uv run constraint_aware_neural_networks train --model=isothermal_euler --algorithm=standard --data ./data/iso_euler_datasets/iso_euler_data_2000_n0/

Show tensorboard log:

        uv run tensorboard --logdir=./output/logs
