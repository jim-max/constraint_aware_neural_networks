import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class Data_Handle:
    """
    Training and test data handle.

    Loads the training and test data, and normalizes it.

    .. note::
        The data handle does not support data that does not fit into memory.
    """

    def __init__(
        self,
        train_data_file: Path,
        test_data_file: None | Path,
        feature_names: list[str],
        label_names: list[str],
        validation_split_ratio: float = 0.2,
    ):
        self.label_names = label_names
        self.feature_names = feature_names
        self.validation_split_ratio = validation_split_ratio

        # ---------------------------------------------------
        # read data
        na_values = [" nan", " -nan", " inf", " -inf"]
        separator = ","

        logger.debug(f"loading training data {train_data_file}...")
        train_dataframe = pd.read_csv(
            train_data_file,
            na_values=na_values,
            sep=separator,
        )
        # extract feature/label data
        self.train_labels = train_dataframe[self.label_names]
        self.train_features = train_dataframe[self.feature_names]
        logger.info(f"loaded training data: \n{train_dataframe}")

        if test_data_file is not None:
            logger.debug(f"loading test data {test_data_file}...")
            test_dataframe = pd.read_csv(
                test_data_file,
                na_values=na_values,
                sep=separator,
            )
            # extract feature/label data
            self.test_labels = test_dataframe[self.label_names]
            self.test_features = test_dataframe[self.feature_names]
            logger.info(f"loaded test data: \n{test_dataframe}")
        else:
            self.test_labels = None
            self.test_features = None

        self.output_dimension = len(self.label_names)
        self.input_dimension = len(self.feature_names)

        # compute feature distribution:
        self.x_mean = self.train_features.mean()
        self.x_std = self.train_features.std()
        self.x_max = self.train_features.max()
        self.x_min = self.train_features.min()

        # compute label distribution:
        self.y_mean = self.train_labels.mean()
        self.y_std = self.train_labels.std()
        self.y_max = self.train_labels.max()
        self.y_min = self.train_labels.min()

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.train_features,
            self.train_labels,
            test_size=self.validation_split_ratio,
            random_state=1,
        )
