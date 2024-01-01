import json
import logging
import multiprocessing
import os
import time
from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from constraint_aware_neural_networks.common.data_handle import Data_Handle
from constraint_aware_neural_networks.common.parameters import (
    NN_Parameters,
    NN_Parameters_JSON_Encoder,
)
from constraint_aware_neural_networks.util.evaluate import eval_network
from constraint_aware_neural_networks.util.plotting import plot_loss_history

logger = logging.getLogger(__name__)


def train_network(
    network: torch.nn.Module,
    parameters: NN_Parameters,
    data_handle: Data_Handle,
    use_gpu=False,
):
    """
    Main function for training neural networks.

    It contains the main training loop, keeps logging the metrics,
    and exports the model.
    """
    start_time = time.time()

    # -------------------------------
    # setup device info

    n_cpus = multiprocessing.cpu_count()
    n_cpus = min(n_cpus, 4)
    os.environ["NUM_CORES"] = str(n_cpus)
    os.environ["OMP_NUM_THREADS"] = str(n_cpus)

    if use_gpu:
        if not (torch.cuda.is_available()):
            logger.error("CUDA-device is not available!")
            raise RuntimeError
        dtype = torch.float
        device = torch.device("cuda:0")
    else:
        dtype = torch.float
        device = torch.device("cpu")
    logger.info(f"Using {device=}, {dtype=}")

    network.type(dtype).to(device)
    parameters.loss_function.type(dtype).to(device)

    # -------------------------------------

    x_train = torch.from_numpy(data_handle.x_train.to_numpy())
    y_train = torch.from_numpy(data_handle.y_train.to_numpy())
    x_val = torch.from_numpy(data_handle.x_val.to_numpy())
    y_val = torch.from_numpy(data_handle.y_val.to_numpy())

    x_train = x_train.type(dtype).to(device)
    y_train = y_train.type(dtype).to(device)
    x_val = x_val.type(dtype).to(device)
    y_val = y_val.type(dtype).to(device)

    # -------------------------------------

    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    if parameters.model_file_in is not None:
        try:
            network.load_state_dict(
                torch.load(parameters.model_file_in, map_location=device), strict=True
            )
            logger.info(
                f"successfully loaded neural network from {parameters.model_file_in}"
            )
        except Exception:
            logger.exception(
                f"Neural network could not be loaded from {parameters.model_file_in}!"
            )
            raise
    else:
        network.apply(init_weights)

    # --------------------------------------

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=parameters.learning_rate,
        eps=1.0e-6,
    )

    if parameters.enable_learning_rate_scheduler:
        total_steps: int
        if parameters.batch_size is not None:
            total_steps = parameters.epochs * ceil(
                x_train.shape[0] / parameters.batch_size
            )
        else:
            total_steps = parameters.epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=parameters.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=20.0,
            final_div_factor=1e4,
        )
    else:
        scheduler = None

    # --------------------------------------

    epoch_array = []
    train_losses = []
    val_losses = []
    train_constraint_losses = []
    val_constraint_losses = []
    l2_weight_losses = []
    epoch_durations = []

    parameters.network_output_directory.mkdir(parents=True, exist_ok=True)

    if parameters.enable_tensorboard_logging:
        parameters.log_path.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(parameters.log_path)

    start_time = time.time()
    epoch_start_time = time.time()

    best_val_loss = np.inf
    best_epoch = 0

    torch.save(network.state_dict(), parameters.network_file_out)

    # -----------------------------------------------------

    try:
        with logging_redirect_tqdm():
            for epoch in (pbar := tqdm(range(parameters.epochs))):
                epoch_start_time = time.time()
                # -----------------------------------------------
                # perform training step

                network.train()

                # shuffle data:
                idx = torch.randperm(x_train.shape[0], device=device)
                x_train = x_train[idx, :]
                y_train = y_train[idx, :]

                # split data in batches:
                if parameters.batch_size is not None:
                    x_batches = torch.split(x_train, parameters.batch_size, dim=0)
                    y_batches = torch.split(y_train, parameters.batch_size, dim=0)
                else:
                    x_batches = (x_train,)
                    y_batches = (y_train,)

                for _batch_iter, (x_batch, y_batch) in enumerate(
                    zip(x_batches, y_batches, strict=True)
                ):
                    prediction = network(x_batch)
                    loss = parameters.loss_function(prediction, y_batch)

                    # l2 regularization on weights:
                    l2_reg = torch.tensor(0.0)
                    for name, param in network.named_parameters():
                        if ("output_layer" not in name) and ("bias" not in name):
                            l2_reg = l2_reg + torch.pow(param, 2).sum()
                    loss = loss + parameters.weight_penalty * l2_reg

                    optimizer.zero_grad()
                    loss.backward()
                    if parameters.clipnorm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            network.parameters(), parameters.clipnorm
                        )
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                # -----------------------------------------------
                # perform eval step
                network.eval()
                epoch_array.append(epoch)

                # compute train loss for evaluation
                prediction = network(x_train)
                train_loss = float(parameters.loss_function(prediction, y_train))
                train_losses.append(train_loss)
                if parameters.constraint_loss_function is not None:
                    train_constraint_loss = float(
                        torch.mean(parameters.constraint_loss_function(prediction))
                    )
                    train_constraint_losses.append(train_constraint_loss)
                else:
                    train_constraint_loss = np.nan

                # compute validation loss for evaluation
                prediction = network(x_val)
                val_loss = float(parameters.loss_function(prediction, y_val))
                val_losses.append(val_loss)
                if parameters.constraint_loss_function is not None:
                    val_constraint_loss = float(
                        torch.mean(parameters.constraint_loss_function(prediction))
                    )
                    val_constraint_losses.append(val_constraint_loss)
                else:
                    val_constraint_loss = np.nan

                l2_weight_loss = 0.0
                for name, param in network.named_parameters():
                    if ("output_layer" not in name) and ("bias" not in name):
                        l2_weight_loss += float(torch.pow(param, 2).sum())
                l2_weight_losses.append(l2_weight_loss)

                epoch_duration = time.time() - epoch_start_time
                epoch_durations.append(epoch_duration)

                # -----------------------------------------------

                if parameters.enable_tensorboard_logging:
                    # write tensorboard summary
                    summary_writer.add_scalars(
                        "loss", {"train": train_loss, "val": val_loss}, epoch
                    )
                    if not np.isnan(train_constraint_loss) and not np.isnan(
                        val_constraint_loss
                    ):
                        summary_writer.add_scalars(
                            "constraint_loss_function",
                            {
                                "train": train_constraint_loss,
                                "val": val_constraint_loss,
                            },
                            epoch,
                        )
                    summary_writer.add_scalar(
                        "penalty/l2_weight_loss", l2_weight_loss, epoch
                    )
                    summary_writer.add_scalar(
                        "param/learning_rate",
                        float(optimizer.param_groups[0]["lr"]),
                        epoch,
                    )
                    summary_writer.add_scalar(
                        "time/epoch_duration",
                        epoch_duration,
                        epoch,
                    )

                if np.isnan(train_loss) or np.isnan(val_loss):
                    logger.error(
                        f"Error: Invalid loss encountered! {train_loss=}, {val_loss=}"
                    )
                    raise ValueError

                # save only the best solutions
                if val_loss < best_val_loss:
                    logger.debug(
                        f"loss decreased from {best_val_loss:.4e} "
                        f"to {val_loss:.4e} - save network."
                    )
                    best_val_loss = val_loss
                    best_epoch = epoch
                    torch.save(network.state_dict(), parameters.network_file_out)

                # stop if patience is reached
                patience = np.abs(epoch - best_epoch)
                if patience > parameters.patience:
                    logger.info("patience reached - stop training!")
                    break

                pbar.set_description(
                    f"ep = {epoch}, loss = {train_loss:.3e}, "
                    f"val_loss = {val_loss:.3e}, time/ep = {epoch_duration:.1e}s, "
                    f"patience = {patience}"
                )

    except KeyboardInterrupt:
        logger.warning("manually stopping the training...")

    # -----------------------------------------------------

    training_time = time.time() - start_time
    logger.info(
        f"Finished training after {training_time:.3e} s and {epoch_array[-1]} epochs."
    )

    # -----------------------------------------------------
    # save loss history
    training_history_dataframe = pd.DataFrame(
        {
            "epoch": epoch_array,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_constraint_loss": train_constraint_losses,
            "val_constraint_loss": val_constraint_losses,
            "l2_weight_loss": l2_weight_losses,
            "epoch_duration": epoch_durations,
        }
    )
    training_history_dataframe = training_history_dataframe.set_index("epoch")

    csv_filename = parameters.network_output_directory / "loss_data.csv.gz"
    training_history_dataframe.to_csv(csv_filename, compression="gzip")

    # -----------------------------------------------------
    # save parameters
    with (parameters.network_output_directory / "parameters.json").open(
        "w"
    ) as parameter_file:
        json.dump(
            parameters,
            parameter_file,
            cls=NN_Parameters_JSON_Encoder,
            sort_keys=True,
            indent=3,
        )

    # -----------------------------------------------------
    # plot loss history
    try:
        plot_loss_history(
            training_history_dataframe,
            parameters.network_output_directory,
            title=parameters.network_id,
        )
    except Exception:
        logger.exception("Plotting the loss history failed.")

    # -----------------------------------------------------
    # export the trained network

    # load last best network
    network.load_state_dict(torch.load(parameters.network_file_out))
    # set to eval mode
    network.eval()

    export_path = parameters.network_output_directory / "export"
    export_path.mkdir(parents=True, exist_ok=True)
    # save model in TorchScript/pytorch format
    network_scripted = torch.jit.script(network)
    network_scripted.save(export_path / f"{parameters.network_id}.pt")

    # save input-output labels:
    with (export_path / "in_labes.txt").open("w") as f:
        f.write("\n".join(parameters.feature_names))
    with (export_path / "out_labes.txt").open("w") as f:
        f.write("\n".join(parameters.label_names))

    # -----------------------------------------------------

    hyperparameters, eval_metrics = eval_network(
        network=network,
        parameters=parameters,
        data_handle=data_handle,
        device=device,
        dtype=dtype,
    )

    if parameters.enable_tensorboard_logging:
        # add prefix
        hyperparameters = {
            f"hparam/{key}": value for key, value in hyperparameters.items()
        }
        eval_metrics = {f"metric/{key}": value for key, value in eval_metrics.items()}

        summary_writer.add_hparams(
            hyperparameters, eval_metrics, run_name=parameters.network_id
        )
        summary_writer.flush()
        summary_writer.close()

    logger.info(f"Output saved at {parameters.network_output_directory}")
    return
