import dataclasses
import datetime
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class NN_Parameters:
    """
    Collection of all parameters and options used in the neural network training.
    """

    algorithm_name: str
    model_name: str

    data_directory: Path
    label_names: list[str]
    feature_names: list[str]

    constraint_coefficient_lambda: float = 0.0

    validation_split_ratio: float = 0.2

    epochs: int = 10000
    patience: int = 1000
    learning_rate: float = 2.0e-4
    enable_learning_rate_scheduler: bool = False

    model_file_in: None | Path = None
    net_topology: list[int] = field(default_factory=lambda: [20, 20, 20])
    activation_function: Callable = F.elu

    constraint_deviation_function: Any = None
    constraint_loss_function: Any = None
    constraint_resolving_layer: Any = None
    loss_function: Any = field(default_factory=lambda: torch.nn.MSELoss())

    batch_size: None | int = None
    weight_penalty: float = 1.0e-7
    clipnorm: float = 1.0

    timestamp: str = field(init=False)

    output_directory: Path = Path("./output")

    network_output_directory: Path = field(init=False)
    network_id: str = field(init=False)
    network_file_out: Path = field(init=False)

    enable_tensorboard_logging: bool = True
    log_path: Path = field(init=False)

    def __post_init__(self):
        uid = str(uuid.uuid4().hex)[:3]
        time_now = datetime.datetime.now(tz=datetime.timezone.utc)
        time_str = str(time_now.strftime("%Y_%m_%d_%H_%M"))
        self.timestamp = time_now.replace(microsecond=0).isoformat()

        dataset_name = self.data_directory.name
        self.network_id = (
            f"{self.model_name}_{self.algorithm_name}_{dataset_name}_{time_str}_{uid}"
        )
        self.network_output_directory = self.output_directory / self.network_id
        self.network_file_out = self.network_output_directory / "network.pth"
        self.log_path = self.output_directory / "logs" / self.network_id

    @property
    def train_data_file(self) -> Path:
        return self.data_directory / "data.csv.gz"

    @property
    def test_data_file(self) -> Path:
        return self.data_directory / "testgrid_data.csv.gz"

    @property
    def input_dimension(self) -> int:
        return len(self.feature_names)

    @property
    def output_dimension(self) -> int:
        return len(self.label_names)


class NN_Parameters_JSON_Encoder(json.JSONEncoder):
    """
    Helper class to save the parameter configuration as a json file.
    """

    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.nn.Module):
            return obj.__class__.__name__
        if callable(obj):
            return obj.__name__
        return super().default(obj)
