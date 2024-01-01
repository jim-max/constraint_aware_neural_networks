import numpy as np
import pandas as pd
import torch

VAN_DER_WAALS_CONSTANT_R = 8.0 / 3.0
VAN_DER_WAALS_CONSTANT_A = 3.0
VAN_DER_WAALS_CONSTANT_B = 1.0 / 3.0
VAN_DER_WAALS_REFERENCE_TEMPERATURE = 0.85


def compute_iso_euler_constraint_deviation(input: pd.DataFrame):
    """
    Compute the deviation from the isothermal Euler model constraint.
    """
    rhol = input["rhol_s"].to_numpy()
    ml = input["ml_s"].to_numpy()
    rhor = input["rhor_s"].to_numpy()
    mr = input["mr_s"].to_numpy()
    s = input["s"].to_numpy()

    R = VAN_DER_WAALS_CONSTANT_R
    a = VAN_DER_WAALS_CONSTANT_A
    b = VAN_DER_WAALS_CONSTANT_B
    T_ref = VAN_DER_WAALS_REFERENCE_TEMPERATURE

    pl = np.divide(
        np.multiply(R * T_ref, rhol), (1.0 - np.multiply(b, rhol))
    ) - np.multiply(a, np.power(rhol, 2))
    pr = np.divide(
        np.multiply(R * T_ref, rhor), (1.0 - np.multiply(b, rhor))
    ) - np.multiply(a, np.power(rhor, 2))

    val_0 = np.multiply(s, (rhol - rhor)) - (ml - mr)
    val_1 = np.multiply(s, (ml - mr)) - (
        (np.divide(np.power(ml, 2), rhol) + pl)
        - (np.divide(np.power(mr, 2), rhor) + pr)
    )

    val = np.abs(val_0) + np.abs(val_1)

    return np.abs(val)


class Iso_Euler_Constraint_Loss(torch.nn.Module):
    """
    PyTorch module for computing the constraint loss for the isothermal Euler model.
    """

    def forward(self, input):
        rhol = input[:, 0]
        ml = input[:, 1]
        rhor = input[:, 2]
        mr = input[:, 3]
        s = input[:, 4]

        R = VAN_DER_WAALS_CONSTANT_R
        a = VAN_DER_WAALS_CONSTANT_A
        b = VAN_DER_WAALS_CONSTANT_B
        T_ref = VAN_DER_WAALS_REFERENCE_TEMPERATURE

        pl = R * T_ref * rhol / (torch.ones_like(rhol) - b * rhol) - a * rhol**2
        pr = R * T_ref * rhor / (torch.ones_like(rhor) - b * rhor) - a * rhor**2

        val0 = s * (rhol - rhor) - (ml - mr)
        val1 = s * (ml - mr) - (((ml**2 / rhol) + pl) - ((mr**2 / rhor) + pr))

        val = torch.abs(val0) + torch.abs(val1)
        return val


class Iso_Euler_Constraint_Resolving_Layer(torch.jit.ScriptModule):
    """
    The constraint-resolving layer for the isothermal Euler model.
    """

    def __init__(self):
        super().__init__()
        self.input_dimension = 4

    @torch.jit.script_method
    def forward(self, input):
        rhol = input[:, 0]
        ml = input[:, 1]
        rhor = input[:, 2]
        s = input[:, 3]

        mr = ml - s * (rhol - rhor)

        rhol = rhol.unsqueeze(1)
        rhor = rhor.unsqueeze(1)
        ml = ml.unsqueeze(1)
        mr = mr.unsqueeze(1)
        s = s.unsqueeze(1)
        output = torch.cat((rhol, ml, rhor, mr, s), dim=1)
        return output


class Approximate_Iso_Euler_Constraint_Resolving_Layer(torch.jit.ScriptModule):
    """
    The approximative constraint-resolving layer for the isothermal Euler model.
    """

    def __init__(self):
        super().__init__()
        self.input_dimension = 4
        self._finite_difference_tolerance = 1e-9
        self._R = VAN_DER_WAALS_CONSTANT_R
        self._a = VAN_DER_WAALS_CONSTANT_A
        self._b = VAN_DER_WAALS_CONSTANT_B
        self._T_ref = VAN_DER_WAALS_REFERENCE_TEMPERATURE

    @torch.jit.script_method
    def forward(self, input):
        rhol = input[:, 0]
        ml = input[:, 1]
        rhor = input[:, 2]
        mr = input[:, 3]

        pl = (
            self._R * self._T_ref * rhol / (torch.ones_like(rhol) - self._b * rhol)
            - self._a * rhol**2
        )
        pr = (
            self._R * self._T_ref * rhor / (torch.ones_like(rhor) - self._b * rhor)
            - self._a * rhor**2
        )

        squared_diff = (rhol - rhor) ** 2 + (ml - mr) ** 2
        s_val = (
            (ml - mr) * (rhol - rhor)
            + (((ml**2 / rhol) + pl) - ((mr**2 / rhor) + pr)) * (ml - mr)
        ) / squared_diff

        flux_derivative = 0.5 * (ml / rhol + mr / rhor)
        s_approximation = torch.where(
            squared_diff < self._finite_difference_tolerance, flux_derivative, s_val
        )

        rhol = rhol.unsqueeze(1)
        ml = ml.unsqueeze(1)
        rhor = rhor.unsqueeze(1)
        mr = mr.unsqueeze(1)
        s_approximation = s_approximation.unsqueeze(1)
        output = torch.cat((rhol, ml, rhor, mr, s_approximation), dim=1)
        return output
