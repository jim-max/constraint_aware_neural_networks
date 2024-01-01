import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from constraint_aware_neural_networks.common.parameters import NN_Parameters

logger = logging.getLogger(__name__)


def plot_loss_history(
    loss_history: pd.DataFrame,
    network_output_directory: Path,
    title: str | None = None,
):
    """
    Plots the loss values over epochs and saves it as an image.
    """
    figure_size = 5

    plt.figure(figsize=(figure_size, figure_size))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.semilogy(
        loss_history.index.values,
        loss_history["train_loss"].values,
        alpha=0.5,
        linewidth=0.5,
        color="#377eb8",
        label="train loss",
    )
    plt.semilogy(
        loss_history.index.values,
        loss_history.cummin()["train_loss"].values,
        alpha=0.75,
        linewidth=0.5,
        color="#377eb8",
    )

    plt.semilogy(
        loss_history.index.values,
        loss_history["val_loss"].values,
        alpha=0.5,
        linewidth=0.5,
        color="#e41a1c",
        label="val loss",
    )
    plt.semilogy(
        loss_history.index.values,
        loss_history.cummin()["val_loss"].values,
        alpha=0.75,
        linewidth=0.5,
        color="#e41a1c",
    )

    try:
        plt.semilogy(
            loss_history.index,
            loss_history.rolling(20, center=True).mean()["train_constraint_loss"],
            label="constraint loss",
            alpha=0.33,
            linewidth=0.5,
            color="k",
            linestyle="--",
        )
    except Exception:
        logger.exception("plotting the train_constraint_loss failed!")
    try:
        plt.semilogy(
            loss_history.index,
            loss_history.rolling(20, center=True).mean()["val_constraint_loss"],
            label="val constraint loss",
            alpha=0.33,
            linewidth=0.5,
            color="k",
        )
    except Exception:
        logger.exception("plotting the val_constraint_loss failed!")

    plt.legend()

    if title:
        plt.title(title)

    val_losses = loss_history["val_loss"].to_numpy()
    val_losses = val_losses[np.isfinite(val_losses)]
    max_yval = np.percentile(val_losses, 99.5)
    plt.ylim(top=max_yval)

    # save plots
    network_output_directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        network_output_directory / "loss_history.jpg",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    return


def plot_predictions_vs_references(
    predictions: npt.NDArray,
    references: npt.NDArray,
    parameters: NN_Parameters,
    prefix: str | None = None,
):
    """
    Plots the prediction values vs. the reference values.
    """
    figure_size = 5.0
    plt.figure(figsize=(figure_size, parameters.output_dimension * figure_size))
    gs = gridspec.GridSpec(parameters.output_dimension, 1)
    for idx in range(parameters.output_dimension):
        ax = plt.subplot(gs[idx])
        ax.scatter(references[:, idx], predictions[:, idx])
        ax.set_xlabel("true values")
        ax.set_ylabel("predictions")
        ax.set_title(parameters.label_names[idx])
        ax.axis("equal")
        ax.set_xlim(plt.xlim())
        ax.set_ylim(plt.ylim())
        _ = ax.plot([-100, 100], [-100, 100])

    plt.tight_layout()

    # save plots
    parameters.network_output_directory.mkdir(parents=True, exist_ok=True)
    output_filename = "predictions_vs_true_values.jpg"
    if prefix is not None:
        output_filename = f"{prefix}_{output_filename}"
    plt.savefig(
        parameters.network_output_directory / output_filename,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
