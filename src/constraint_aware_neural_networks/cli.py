import logging
from pathlib import Path

import click

from constraint_aware_neural_networks.common.algorithm_types import Algorithm_Type
from constraint_aware_neural_networks.common.train import train_network
from constraint_aware_neural_networks.models.cubic.config import (
    setup_cubic_model_configuration,
)
from constraint_aware_neural_networks.models.euler.config import (
    setup_euler_model_configuration,
)
from constraint_aware_neural_networks.models.iso_euler.config import (
    setup_iso_euler_model_configuration,
)
from constraint_aware_neural_networks.models.model_types import Model_Type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(package_name="constraint_aware_neural_networks")
def cli():
    """
    Command-line tool for constraint-aware neural networks.

    Run one of the commands below (with --help) for further information.
    """
    pass


@click.command()
@click.option(
    "data_directory",
    "--data",
    type=click.Path(path_type=Path),
    help="training data path.",
)
@click.option(
    "--model",
    type=click.Choice([model.value for model in Model_Type], case_sensitive=False),
    help="PDE model type.",
)
@click.option(
    "--algorithm",
    type=click.Choice([alg.value for alg in Algorithm_Type], case_sensitive=False),
    default=Algorithm_Type.STANDARD,
    show_default=True,
    help="constraint-aware neural network algorithm.",
)
@click.option(
    "output_directory",
    "--output_directory",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    show_default=True,
    help="output directory.",
)
@click.option(
    "--constraint-coefficient",
    type=float,
    default=0.0,
    show_default=True,
    help="constraint coefficient λ for the custom_loss algorithm.",
)
@click.option(
    "--use-gpu/--use-cpu",
    default=True,
    show_default=True,
)
def train(
    data_directory, model, algorithm, output_directory, constraint_coefficient, use_gpu
):
    """
    Start the neural network training.
    """

    match model:
        case Model_Type.CUBIC:
            setup_function = setup_cubic_model_configuration
        case Model_Type.ISOTHERMAL_EULER:
            setup_function = setup_iso_euler_model_configuration
        case Model_Type.EULER:
            setup_function = setup_euler_model_configuration
        case _:
            logger.error(f"no configuration found for {model=}!")
    network, parameters, data_handle = setup_function(
        data_directory=data_directory,
        output_directory=output_directory,
        algorithm_type=algorithm,
        constraint_coefficient=constraint_coefficient,
    )

    train_network(
        network=network,
        parameters=parameters,
        data_handle=data_handle,
        use_gpu=use_gpu,
    )


cli.add_command(train)

if __name__ == "__main__":
    cli()
