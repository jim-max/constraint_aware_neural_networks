import numpy as np
import pandas as pd
import torch


def compute_cubic_constraint_deviation(input: pd.DataFrame):
    """
    Compute the deviation from the cubic flux constraint.
    """
    ul = input["ul_s"].to_numpy()
    ur = input["ur_s"].to_numpy()
    s = input["s"].to_numpy()
    val = np.multiply(s, (ul - ur)) - (np.power(ul, 3) - np.power(ur, 3))
    return np.abs(val)


class Cubic_Constraint_Loss(torch.nn.Module):
    """
    PyTorch module for computing the constraint loss for the cubic flux model.
    """

    def forward(self, input):
        ul = input[:, 0]
        ur = input[:, 1]
        s = input[:, 2]
        val = s * (ul - ur) - (ul**3.0 - ur**3.0)
        return torch.abs(val)


class Cubic_Constraint_Resolving_Layer(torch.jit.ScriptModule):
    """
    The constraint-resolving layer for the cubic flux model.
    """

    def __init__(self):
        super().__init__()
        self.input_dimension = 2
        self._finite_difference_tolerance = 1e-9

    @torch.jit.script_method
    def forward(self, input):
        ul = input[:, 0]
        ur = input[:, 1]

        delta_u = torch.abs(ul - ur)
        s_val = (ul**3 - ur**3) / (ul - ur)
        flux_derivative = 3.0 * (0.5 * (ul + ur)) ** 2
        s = torch.where(
            delta_u < self._finite_difference_tolerance, flux_derivative, s_val
        )

        ul = ul.unsqueeze(1)
        ur = ur.unsqueeze(1)
        s = s.unsqueeze(1)
        output = torch.cat((ul, ur, s), dim=1)
        return output
