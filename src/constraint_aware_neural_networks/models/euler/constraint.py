import numpy as np
import pandas as pd
import torch

CONSTANT_GAMMA = 1.4


def compute_euler_constraint_deviation(input: pd.DataFrame):
    """
    Compute the deviation from the Euler model constraint.
    """
    rhol = input["rhol_s"].to_numpy()
    ml = input["ml_s"].to_numpy()
    el = input["El_s"].to_numpy()
    rhor = input["rhor_s"].to_numpy()
    mr = input["mr_s"].to_numpy()
    er = input["Er_s"].to_numpy()
    s = input["s"].to_numpy()

    gamma = CONSTANT_GAMMA
    pl = (gamma - 1.0) * (el - 0.5 * np.divide(np.power(ml, 2), rhol))
    pr = (gamma - 1.0) * (er - 0.5 * np.divide(np.power(mr, 2), rhor))

    val_0 = np.multiply(s, (rhol - rhor)) - (ml - mr)
    val_1 = np.multiply(s, (ml - mr)) - (
        (np.divide(np.power(ml, 2), rhol) + pl)
        - (np.divide(np.power(mr, 2), rhor) + pr)
    )
    val_2 = np.multiply(s, (el - er)) - (
        np.multiply(el + pl, np.divide(ml, rhol))
        - np.multiply(er + pr, np.divide(mr, rhor))
    )

    val = np.abs(val_0) + np.abs(val_1) + np.abs(val_2)
    return val


class Euler_Constraint_Loss(torch.nn.Module):
    """
    PyTorch module for computing the constraint loss for the Euler model.
    """

    def forward(self, input):
        rhol = input[:, 0]
        ml = input[:, 1]
        el = input[:, 2]
        rhor = input[:, 3]
        mr = input[:, 4]
        er = input[:, 5]
        s = input[:, 6]

        gamma = CONSTANT_GAMMA
        pl = (gamma - 1.0) * (el - 0.5 * ml**2 / rhol)
        pr = (gamma - 1.0) * (er - 0.5 * mr**2 / rhor)

        val_0 = s * (rhol - rhor) - (ml - mr)
        val_1 = s * (ml - mr) - ((((ml**2) / rhol) + pl) - (((mr**2) / rhor) + pr))
        val_2 = s * (el - er) - (((el + pl) * (ml / rhol)) - ((er + pr) * (mr / rhor)))

        val = torch.abs(val_0) + torch.abs(val_1) + torch.abs(val_2)
        return val


class Euler_Constraint_Resolving_Layer(torch.jit.ScriptModule):
    """
    The constraint-resolving layer for the Euler model.
    """

    def __init__(self):
        super().__init__()
        self.input_dimension = 4

    @torch.jit.script_method
    def forward(self, input):
        rhol = input[:, 0]
        rho_epsl = input[:, 1]
        rhor = input[:, 2]
        s = input[:, 3]

        ml = rhol * s
        el = rho_epsl + 0.5 * rhol * s * s
        mr = rhor * s
        er = rho_epsl + 0.5 * rhor * s * s

        rhol = rhol.unsqueeze(1)
        ml = ml.unsqueeze(1)
        el = el.unsqueeze(1)
        rhor = rhor.unsqueeze(1)
        mr = mr.unsqueeze(1)
        er = er.unsqueeze(1)
        s = s.unsqueeze(1)
        output = torch.cat((rhol, ml, el, rhor, mr, er, s), dim=1)
        return output
