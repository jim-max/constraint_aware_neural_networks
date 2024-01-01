import logging

import numpy as np
import torch

from constraint_aware_neural_networks.common.algorithm_types import Algorithm_Type
from constraint_aware_neural_networks.common.data_handle import Data_Handle
from constraint_aware_neural_networks.common.loss_functions import (
    Constraint_Adapted_Loss,
    Constraint_Adapted_Loss_Scaled,
    MSE_Loss_Scaled,
)
from constraint_aware_neural_networks.common.network_architectures import (
    Resolving_Net,
    Standard_Net,
)
from constraint_aware_neural_networks.common.parameters import NN_Parameters

logger = logging.getLogger(__name__)


def setup_scaled_loss_function(parameters: NN_Parameters, data_handle: Data_Handle):
    """
    Get the scaled loss function used for training the network, depending on the
    algorithm type
    :class:`~constraint_aware_neural_networks.common.algorithm_types.Algorithm_Type`
    in parameters.
    The loss output is scaled by the squared reciprocal standard deviation
    of each label component of the data set.
    """
    # compute loss scaling factors
    reciprocal_label_sq_std = np.reciprocal(data_handle.y_std.to_numpy() ** 2)

    loss_function: torch.nn.Module
    if parameters.algorithm_name == Algorithm_Type.CONSTRAINT_ADAPTED_LOSS:
        logger.info(
            f"running {parameters.algorithm_name} with coefficient "
            f"λ = {parameters.constraint_coefficient_lambda}."
        )
        loss_function = Constraint_Adapted_Loss_Scaled(
            parameters=parameters, scaling_factors=reciprocal_label_sq_std
        )
    else:
        loss_function = MSE_Loss_Scaled(scaling_factors=reciprocal_label_sq_std)
    return loss_function


def setup_loss_function(parameters: NN_Parameters):
    """
    Get the loss function used for training the network, depending on the
    algorithm type
    :class:`~constraint_aware_neural_networks.common.algorithm_types.Algorithm_Type`
    in parameters.
    """
    loss_function: torch.nn.Module
    if parameters.algorithm_name == Algorithm_Type.CONSTRAINT_ADAPTED_LOSS:
        logger.info(
            f"running {parameters.algorithm_name} with coefficient "
            f"λ = {parameters.constraint_coefficient_lambda}."
        )
        loss_function = Constraint_Adapted_Loss(parameters)
    else:
        loss_function = torch.nn.MSELoss()
    return loss_function


def setup_network(parameters: NN_Parameters, data_handle: Data_Handle):
    """
    Defines the neural network architecture depending on the
    algorithm type
    :class:`~constraint_aware_neural_networks.common.algorithm_types.Algorithm_Type`
    and adds the data set distribution parameters for input value standardization.
    """
    network: torch.nn.Module
    if parameters.algorithm_name == Algorithm_Type.CONSTRAINT_RESOLVING:
        network = Resolving_Net(
            parameters,
            input_mean=data_handle.x_mean.to_numpy(),
            input_std=data_handle.x_std.to_numpy(),
        )
    else:
        network = Standard_Net(
            parameters,
            input_mean=data_handle.x_mean.to_numpy(),
            input_std=data_handle.x_std.to_numpy(),
        )
    return network
