import torch

from constraint_aware_neural_networks.common.parameters import NN_Parameters


class Standard_Net(torch.nn.Module):
    """
    Standard feedforward neural network architecture.
    """

    def __init__(self, parameters: NN_Parameters, input_mean=None, input_std=None):
        super().__init__()
        input_mean_tensor = (
            torch.tensor(input_mean)
            if input_mean is not None
            else torch.zeros(parameters.input_dimension)
        )
        input_std_tensor = (
            torch.tensor(input_std)
            if input_std is not None
            else torch.ones(parameters.input_dimension)
        )
        self.register_buffer("input_mean", input_mean_tensor)
        self.register_buffer("input_std", input_std_tensor)

        self.activation_function = parameters.activation_function

        self.input_layer = torch.nn.Linear(
            parameters.input_dimension, parameters.net_topology[0]
        )
        self.hidden = torch.nn.ModuleList()
        for n_nodes in parameters.net_topology:
            self.hidden.append(torch.nn.Linear(n_nodes, n_nodes))
        self.output_layer = torch.nn.Linear(
            parameters.net_topology[-1], parameters.output_dimension
        )

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std
        x = self.activation_function(self.input_layer(x))
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        x = self.output_layer(x)
        return x


class Resolving_Net(torch.nn.Module):
    """
    Constraint-resolving neural network architecture.
    """

    def __init__(self, parameters: NN_Parameters, input_mean=None, input_std=None):
        super().__init__()
        input_mean_tensor = (
            torch.tensor(input_mean)
            if input_mean is not None
            else torch.zeros(parameters.input_dimension)
        )
        input_std_tensor = (
            torch.tensor(input_std)
            if input_std is not None
            else torch.ones(parameters.input_dimension)
        )
        self.register_buffer("input_mean", input_mean_tensor)
        self.register_buffer("input_std", input_std_tensor)

        self.constraint_resolving_layer = parameters.constraint_resolving_layer
        self.activation_function = parameters.activation_function

        self.input_layer = torch.nn.Linear(
            parameters.input_dimension, parameters.net_topology[0]
        )
        self.hidden = torch.nn.ModuleList()
        for n_nodes in parameters.net_topology:
            self.hidden.append(torch.nn.Linear(n_nodes, n_nodes))
        self.output_layer = torch.nn.Linear(
            parameters.net_topology[-1], self.constraint_resolving_layer.input_dimension
        )

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std
        x = self.activation_function(self.input_layer(x))
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        x = self.output_layer(x)
        x = self.constraint_resolving_layer(x)
        return x
