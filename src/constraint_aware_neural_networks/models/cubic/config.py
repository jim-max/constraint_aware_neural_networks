from pathlib import Path

import torch

from constraint_aware_neural_networks.common.config_helper import (
    setup_network,
    setup_scaled_loss_function,
)
from constraint_aware_neural_networks.common.data_handle import Data_Handle
from constraint_aware_neural_networks.common.parameters import NN_Parameters
from constraint_aware_neural_networks.models.cubic.constraint import (
    Cubic_Constraint_Loss,
    Cubic_Constraint_Resolving_Layer,
    compute_cubic_constraint_deviation,
)
from constraint_aware_neural_networks.models.model_types import Model_Type


def setup_cubic_model_configuration(
    data_directory: Path,
    output_directory: Path,
    algorithm_type: str,
    constraint_coefficient: float = 0.0,
) -> tuple[torch.nn.Module, NN_Parameters, Data_Handle]:
    """
    Sets everything up for the training of the cubic flux model.

    .. note::
        Modify the values in this function if you want to adapt something.
    """
    feature_names = ["ul", "ur"]
    label_names = ["ul_s", "ur_s", "s"]

    parameters = NN_Parameters(
        algorithm_name=algorithm_type,
        model_name=Model_Type.CUBIC.value,
        data_directory=data_directory,
        output_directory=output_directory,
        label_names=label_names,
        feature_names=feature_names,
        constraint_coefficient_lambda=constraint_coefficient,
        # custom parameters:
        epochs=50000,
        patience=50000,
        learning_rate=0.0005,
        constraint_deviation_function=compute_cubic_constraint_deviation,
        constraint_loss_function=Cubic_Constraint_Loss(),
        constraint_resolving_layer=Cubic_Constraint_Resolving_Layer(),
        net_topology=[20] * 5,
    )

    data_handle = Data_Handle(
        train_data_file=parameters.train_data_file,
        test_data_file=parameters.test_data_file,
        label_names=label_names,
        feature_names=feature_names,
        validation_split_ratio=parameters.validation_split_ratio,
    )

    parameters.loss_function = setup_scaled_loss_function(parameters, data_handle)
    network = setup_network(parameters, data_handle)

    return network, parameters, data_handle
