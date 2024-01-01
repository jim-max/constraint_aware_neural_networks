import torch

from constraint_aware_neural_networks.common.parameters import NN_Parameters


class MSE_Loss_Scaled(torch.nn.Module):
    """
    Mean-squared error loss with custom scaling factors
    for each label component.
    """

    def __init__(self, scaling_factors):
        super().__init__()
        self.register_buffer("scaling_factors", torch.tensor(scaling_factors))

    def forward(self, input, target):
        mse = torch.mean((input - target) ** 2, dim=0)
        mse = torch.dot(self.scaling_factors, mse)
        return mse


class Constraint_Adapted_Loss(torch.nn.Module):
    """
    Constraint adapted loss function, consisting of the sum of

    * the mean-squared error loss
    * the constraint loss, scaled by the constraint loss parameter λ
    """

    def __init__(self, parameters: NN_Parameters):
        super().__init__()
        self.constraint_loss_parameter = parameters.constraint_coefficient_lambda
        self.constraint_loss_function = parameters.constraint_loss_function

    def forward(self, input, target):
        mse = torch.mean((input - target) ** 2, dim=0)
        constraint_loss_value = torch.mean(self.constraint_loss_function(input))
        return mse + self.constraint_loss_parameter * constraint_loss_value


class Constraint_Adapted_Loss_Scaled(torch.nn.Module):
    """
    Constraint adapted loss function, consisting of the sum of

    * the mean-squared error loss with custom scaling factors
    for each label component
    * the constraint loss, scaled by the constraint loss parameter λ
    """

    def __init__(self, parameters: NN_Parameters, scaling_factors):
        super().__init__()
        self.constraint_loss_parameter = parameters.constraint_coefficient_lambda
        self.constraint_loss_function = parameters.constraint_loss_function
        self.register_buffer("scaling_factors", torch.tensor(scaling_factors))

    def forward(self, input, target):
        mse = torch.mean((input - target) ** 2, dim=0)
        mse = torch.dot(self.scaling_factors, mse)
        constraint_loss_value = torch.mean(self.constraint_loss_function(input))
        return mse + self.constraint_loss_parameter * constraint_loss_value
