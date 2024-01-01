import numpy as np
import pandas as pd
import torch

from constraint_aware_neural_networks.common.data_handle import Data_Handle
from constraint_aware_neural_networks.common.parameters import NN_Parameters
from constraint_aware_neural_networks.util.plotting import (
    plot_predictions_vs_references,
)


def evaluate_on_data_set(
    network: torch.nn.Module,
    parameters: NN_Parameters,
    input_data: pd.DataFrame,
    labels: pd.DataFrame,
    prefix: None | str = None,
    device="cpu",
    dtype=torch.double,
) -> dict:
    """
    Helper function to evaluate the network on a single data set.
    """
    torch_input_data = torch.from_numpy(input_data.to_numpy())
    torch_input_data = torch_input_data.type(dtype).to(device)
    torch_predictions = network(torch_input_data)
    np_predictions = torch_predictions.detach().cpu().numpy()

    predictions = pd.DataFrame(np_predictions, columns=labels.columns)

    if parameters.constraint_deviation_function is not None:
        constraint_errors = parameters.constraint_deviation_function(predictions)

        constraint_ma_error = np.mean(constraint_errors)
        constraint_ms_error = np.mean(np.square(constraint_errors))
        constraint_max_error = np.max(constraint_errors)
    else:
        constraint_ma_error = None
        constraint_ms_error = None
        constraint_max_error = None

    difference = np.abs(predictions - labels)

    ma_error = np.mean(difference)
    ms_error = np.mean(np.square(difference))
    max_error = np.max(difference)

    output = {
        "constraint_ma_error": constraint_ma_error,
        "constraint_ms_error": constraint_ms_error,
        "constraint_max_error": constraint_max_error,
        "ma_error": ma_error,
        "ms_error": ms_error,
        "max_error": max_error,
    }

    if prefix is not None:
        output = {f"{prefix}_{key}": value for key, value in output.items()}
    return output


def eval_network(
    network: torch.nn.Module,
    parameters: NN_Parameters,
    data_handle: Data_Handle,
    device="cpu",
    dtype=torch.double,
) -> tuple[dict, dict]:
    """
    Evaluate the neural network on the train, validation, and test set.
    Returns two dictionaries for the hyperparameters and metrics.
    """
    # -------------------------------
    network = network.type(dtype).to(device)
    network.eval()

    metrics: dict = {}
    # -------------------------------

    # plot test predictions:
    if data_handle.test_features is not None:
        torch_test_features = torch.from_numpy(data_handle.test_features.to_numpy())
        torch_test_features = torch_test_features.type(dtype).to(device)
        torch_test_predictions = network(torch_test_features)
        test_predictions = torch_test_predictions.detach().cpu().numpy()

        plot_predictions_vs_references(
            predictions=test_predictions,
            references=data_handle.test_labels.to_numpy(),
            parameters=parameters,
            prefix="test",
        )

        test_metrics = evaluate_on_data_set(
            network=network,
            parameters=parameters,
            input_data=data_handle.test_features,
            labels=data_handle.test_labels,
            prefix="test",
            device=device,
            dtype=dtype,
        )
        metrics |= test_metrics

    train_metrics = evaluate_on_data_set(
        network=network,
        parameters=parameters,
        input_data=data_handle.x_train,
        labels=data_handle.y_train,
        prefix="train",
        device=device,
        dtype=dtype,
    )
    metrics |= train_metrics

    val_metrics = evaluate_on_data_set(
        network=network,
        parameters=parameters,
        input_data=data_handle.x_val,
        labels=data_handle.y_val,
        prefix="val",
        device=device,
        dtype=dtype,
    )
    metrics |= val_metrics

    # -------------------------------

    hyperparameters = {
        "algorithm_name": parameters.algorithm_name,
        "model_name": parameters.model_name,
        "constraint_coefficient_lambda": parameters.constraint_coefficient_lambda,
        "epochs": parameters.epochs,
        "learning_rate": parameters.learning_rate,
        "net_topology": ":".join(str(n_nodes) for n_nodes in parameters.net_topology),
        "activation_function": parameters.activation_function.__name__,
        "batch_size": parameters.batch_size,
        "weight_penalty": parameters.weight_penalty,
        "timestamp": parameters.timestamp,
        "dataset": parameters.data_directory.name,
        "enable_scheduler": parameters.enable_learning_rate_scheduler,
    }

    new_eval_dataframe = pd.DataFrame.from_records([hyperparameters | metrics])

    parameters.output_directory.mkdir(parents=True, exist_ok=True)
    eval_data_file = parameters.output_directory / "eval_data.csv"

    if eval_data_file.exists():
        existying_dataframe = pd.read_csv(
            eval_data_file, na_values=[" nan", " -nan", " inf", " -inf"], sep=","
        )
        updated_dataframe = pd.concat(
            (existying_dataframe, new_eval_dataframe), ignore_index=True
        )
        updated_dataframe.to_csv(eval_data_file, index=False)
    else:
        new_eval_dataframe.to_csv(eval_data_file, index=False)

    return hyperparameters, metrics
