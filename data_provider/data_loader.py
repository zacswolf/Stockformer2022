import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler, dotdict
from utils.timefeatures import time_features

import warnings

warnings.filterwarnings("ignore")


def create_datasets(config):
    """
    Create train, validation, and test datasets using the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration settings for the DataInfo object and DatasetCustom objects.

    Returns:
        tuple: A tuple containing the train, validation, and test datasets as DatasetCustom objects.
    """
    data_info = DataInfo(config)
    train_dataset = DatasetCustom(data_info, "train")
    val_dataset = DatasetCustom(data_info, "val")
    test_dataset = DatasetCustom(data_info, "test")
    return train_dataset, val_dataset, test_dataset


class DataInfo:
    """
    A class to preprocess and manage data for train, validation, and test datasets.

    This class reads a raw stock data file, preprocesses it, and splits it into train, validation, and test sets.
    The class also handles data scaling and time encoding.
    """

    def __init__(self, config: dotdict):
        """
        Initialize the DataInfo object with a given configuration.

        Args:
            config (dotdict): A dictionary containing the configuration settings for the DataInfo object.
        """
        # Default values
        defaults = {
            "size": None,
            "features": "S",
            "target": "OT",
            "scale": True,
            "inverse_pred": False,
            "inverse_output": False,
            "cols": None,
            "date_start": None,
            "date_end": None,
            "date_test": None,
            "date_val": None,
            "t_embed": None,
        }
        config = dotdict({**defaults, **config})

        assert config.seq_len is not None
        assert config.label_len is not None
        assert config.pred_len is not None
        # assert flag in ["train", "test", "val"]
        assert config.freq is not None
        assert config.root_path is not None
        assert config.data_path is not None
        assert (
            (config.date_start is None)
            or (config.date_end is None)
            or (config.date_start < config.date_end)
        ), "date_start isn't before date_end"

        assert (
            (config.date_test is None)
            or (config.date_end is None)
            or (config.date_test < config.date_end)
        ), "date_test isn't before date_end"
        assert (
            (config.date_test is None)
            or (config.date_start is None)
            or (config.date_test > config.date_start)
        ), "date_test isn't after date_start"

        assert (config.date_val is None) or (
            config.date_test is not None
        ), "date_val is used without date_test"
        assert (
            (config.date_val is None)
            or (config.date_test is None)
            or (config.date_val < config.date_test)
        ), "date_val isn't before date_test"

        assert (
            (config.date_val is None)
            or (config.date_end is None)
            or (config.date_val < config.date_end)
        ), "date_val isn't before date_end"
        assert (
            (config.date_val is None)
            or (config.date_start is None)
            or (config.date_val > config.date_start)
        ), "date_val isn't after date_start"

        assert (config.label_len == 0) or (
            config.inverse_output == config.inverse_pred
        ), "If label length is non-zero then inverse_pred and inverse_output should be the same"

        self.config = config
        # self.flag = flag

        # self.timeenc = 0 if config.t_embed != "timeF" else 1
        if config.t_embed == "timeF":
            self.timeenc = 1
        elif config.t_embed == "time2vec_add" or config.t_embed == "time2vec_app":
            self.timeenc = 2
        else:
            self.timeenc = 0

        # type_map = {"train": 0, "val": 1, "test": 2}
        # self.set_type = type_map[flag]

        self.__read_data__()

    def __process_stock_data__(self, df_raw):
        """
        Process a stock dataset containing tickers and their traits in columns.

        This method receives a raw DataFrame (df_raw) with columns in the format
        'ticker_trait' and processes it into a DataFrame suitable for use in the data loader.
        The processed DataFrame ensures that the 'date' column is the first column,
        all tickers have the same traits in the same order, and the target ticker_trait is the last column.

        Parameters:
        ----------
        df_raw : pd.DataFrame
            A raw DataFrame containing the stock data with columns in the format 'ticker_trait'.

        Returns:
        -------
        df_processed : pd.DataFrame
            A processed DataFrame suitable for use in the data loader, with columns reordered
            according to the specified tickers, traits, and target ticker_trait.

        Raises:
        ------
        AssertionError
            If any of the following conditions are not met:
                - 'date' column exists in the raw DataFrame.
                - Specified tickers and traits are present in the raw DataFrame.
                - Every ticker has the same traits in the same order.
                - Target ticker_trait is a valid column in the raw DataFrame.
        """
        # Check if date column exists and is the first column
        assert "date" in df_raw.columns, "`date` not in data"
        assert df_raw.columns[0] == "date", "`date` should be the first column"

        rawdata_tickers = set()
        rawdata_traits = set()
        for column in df_raw.columns[1:]:
            ticker, trait = column.split("_")
            rawdata_tickers.add(ticker)
            rawdata_traits.add(trait)
        rawdata_tickers = list(rawdata_tickers)
        rawdata_traits = list(rawdata_traits)

        tickers = sorted(
            self.config.tickers if self.config.tickers else rawdata_tickers
        )
        traits = sorted(self.config.traits if self.config.traits else rawdata_traits)

        target_ticker, target_trait = self.config.target.split("_")

        assert target_ticker in tickers, "Target ticker not in tickers"
        assert target_trait in traits, "Target trait not in traits"

        tickers.remove(target_ticker)
        tickers.append(target_ticker)  # Move target ticker to the end
        traits.remove(target_trait)
        traits.append(target_trait)  # Move target trait to the end

        cols = ["date"]
        for ticker in tickers:
            for trait in traits:
                col = f"{ticker}_{trait}"
                assert col in df_raw.columns, f"Column '{col}' not in data"
                cols.append(col)

        return df_raw[cols], tickers, traits

    def __read_data__(self):
        """
        Read the raw data from a CSV file, preprocess it, and split it into train, validation, and test sets.
        """

        # Read data from CSV
        df_raw = pd.read_csv(os.path.join(self.config.root_path, self.config.data_path))
        df_raw = df_raw.astype(
            {c: np.float32 for c in df_raw.select_dtypes(include="float64").columns}
        )
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """

        # Process stock data
        df_processed, self.tickers, self.traits = self.__process_stock_data__(df_raw)

        # Check for infinity or NaN values in the dataset
        if np.isinf(df_processed[df_processed.columns[1:]].to_numpy()).any():
            raise Exception("There are inf's in the dataset")
        if np.isnan(df_processed[df_processed.columns[1:]].to_numpy()).any():
            raise Exception("There are nan's in the dataset")

        # Filter to datapoints in [date_start, date_end]
        if self.config.date_start is not None:
            df_processed = df_processed.loc[
                (df_processed["date"] >= self.config.date_start)
            ]
        if self.config.date_end is not None:
            df_processed = df_processed.loc[
                (df_processed["date"] <= self.config.date_end)
            ]

        # Prepare timestamps and data
        df_stamp = df_processed[["date"]]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        raw_dates = df_stamp.date.to_numpy(dtype=np.datetime64)
        data_stamp = np.float32(
            time_features(df_stamp, timeenc=self.timeenc, freq=self.config.freq)
        )

        # Determine train, val, and test set lengths
        num_train, num_val, num_test = self._calculate_split_lengths(df_processed)

        # Check if train, val, or test set has zero length
        if num_test == 0:
            raise Exception("Dataset loading issue: num_test==0, check date settings")
        elif num_val == 0:
            raise Exception("Dataset loading issue: num_val==0, check date settings")
        elif num_train == 0:
            raise Exception("Dataset loading issue: num_train==0, check date settings")

        df_data = df_processed[df_processed.columns[1:]]

        # # Scale data if necessary
        # if self.config.scale:
        #     train_start, train_end = self._compute_set_indices(
        #         "train", len(df_processed), num_train, num_val, num_test
        #     )
        #     train_data = df_data[train_start:train_end]
        #     self.scaler.fit(train_data.values, scale_mean=not self.config.no_scale_mean)
        #     data = torch.from_numpy(self.scaler.transform(df_data.values))
        # else:
        #     data = torch.from_numpy(df_data.values)
        scaled_data = self._scale_data(df_data, num_train, num_val, num_test)

        self.df_data = df_data
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.scaled_data = scaled_data
        self.data_stamp = data_stamp
        self.raw_dates = raw_dates
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test

        # # Specific to the data set type below
        # start, end = self._compute_set_indices(
        #     self.flag, df_processed, num_train, num_val, num_test
        # )
        # self.data_stamp = data_stamp[start:end]
        # # Set input and target data
        # self.data_x = data[start:end]
        # if self.config.inverse_pred:
        #     self.data_y = torch.from_numpy(df_data.values[start:end])
        # else:
        #     self.data_y = data[start:end]

    def get_dataset_info(self, flag: str):
        """
        Retrieve the dataset information (timestamps, input data, and target data) for a specified data split.

        Args:
            flag (str): The data split to retrieve information for ("train", "val", or "test").

        Returns:
            tuple: A tuple containing the data_stamp (timestamps), data_x (input data), and data_y (target data) for the specified data split.
        """
        assert flag in [
            "train",
            "val",
            "test",
        ], f"Invalid flag in get_dataset_info: {flag}"

        # Get boundary
        start, end = self._compute_set_indices(
            flag, len(self.df_data), self.num_train, self.num_val, self.num_test
        )

        data_stamp = self.data_stamp[start:end]
        raw_dates = self.raw_dates[start:end]

        data_x = self.scaled_data[start:end]

        if self.config.inverse_pred:
            data_y = torch.from_numpy(self.df_data.values[start:end])
        else:
            data_y = self.scaled_data[start:end]

        return data_stamp, raw_dates, data_x, data_y

    def _calculate_split_lengths(self, df_processed: pd.DataFrame):
        """
        Calculate the lengths of the train, validation, and test sets based on the configuration settings.

        Args:
            df_processed (pd.DataFrame): The preprocessed DataFrame containing the stock data.

        Returns:
            tuple: A tuple containing the lengths of the train, validation, and test sets.
        """
        if self.config.date_test is not None and self.config.date_val is not None:
            # num_test and num_val are specified
            num_test = len(
                df_processed.loc[df_processed["date"] >= self.config.date_test]
            )
            num_val = len(
                df_processed.loc[
                    (df_processed["date"] >= self.config.date_val)
                    & (df_processed["date"] < self.config.date_test)
                ]
            )
            num_train = len(df_processed) - num_val - num_test
        elif self.config.date_test is not None:
            # num_val is half of num_test which is specified
            num_test = len(
                df_processed.loc[(df_processed["date"] >= self.config.date_test)]
            )
            num_val = num_test // 2
            num_train = len(df_processed) - num_val - num_test
        else:
            # Default split
            print("Warning: using default dataset split")
            num_train = int(len(df_processed) * 0.7)
            num_test = int(len(df_processed) * 0.2)
            num_val = len(df_processed) - num_train - num_test
        return num_train, num_val, num_test

    def _compute_set_indices(
        self, flag: str, num_rows: int, num_train: int, num_val: int, num_test: int
    ):
        """
        Compute the start and end indices for a specified data split.

        Args:
            flag (str): The data split to compute indices for ("train", "val", or "test").
            num_rows (int): The total number of rows in the preprocessed DataFrame, len(df_processed).
            num_train (int): The length of the train set.
            num_val (int): The length of the validation set.
            num_test (int): The length of the test set.

        Returns:
            tuple: A tuple containing the start and end indices for the specified data split.
        """
        # num_rows is len(df_processed)
        assert flag in ["train", "val", "test"], f"Invalid flag: {flag}"
        if flag == "train":  # Train set
            start, end = 0, num_train
        elif flag == "val":  # Validation set
            start, end = num_train - self.config.seq_len, num_train + num_val
        else:  # Test set
            start, end = num_rows - num_test - self.config.seq_len, num_rows
        return start, end

    def _scale_data(self, df_data: pd.DataFrame, num_train, num_val, num_test):
        scale_method = 2

        if not self.config.scale:
            return torch.from_numpy(df_data.values)

        train_start, train_end = self._compute_set_indices(
            "train", len(df_data), num_train, num_val, num_test
        )

        # Don't modify original df_data
        df_data = df_data.copy(deep=True)
        train_data = df_data[train_start:train_end]

        if scale_method == 2:
            # Initialize scalars
            self.scalers = {
                f"{ticker}_{trait}": StandardScaler()
                for ticker in self.tickers
                for trait in self.traits
            }

            for ticker in self.tickers:
                for trait in self.traits:
                    column = f"{ticker}_{trait}"
                    self.scalers[column].fit(
                        train_data[column].values,
                        scale_mean=not self.config.no_scale_mean,
                    )
                    df_data[column] = self.scalers[column].transform(
                        df_data[column].values
                    )

            return torch.from_numpy(df_data.values)
        else:  # scale_method == 2
            self.scalar = StandardScaler()

            train_data = df_data[train_start:train_end]
            self.scaler.fit(train_data.values, scale_mean=not self.config.no_scale_mean)
            return torch.from_numpy(self.scaler.transform(df_data.values))

    def inverse_transform(self, data: np.ndarray):
        if not self.config.scale:
            return data

        scale_method = 2
        if scale_method == 2:
            # Convert the data back to a DataFrame
            columns = [
                f"{ticker}_{trait}" for ticker in self.tickers for trait in self.traits
            ]
            df_data_ = pd.DataFrame(data, columns=columns)

            # Inverse transform each (ticker, trait) pair individually
            for ticker in self.tickers:
                for trait in self.traits:
                    column = f"{ticker}_{trait}"
                    df_data_[column] = self.scalers[column].inverse_transform(
                        df_data_[column].values
                    )

            return df_data_.values
        else:  # scale_method == 2
            return self.scalar.inverse_transform(data)

    # def __getitem__(self, index):
    #     s_begin = index
    #     s_end = s_begin + self.config.seq_len
    #     r_begin = s_end - self.config.label_len
    #     r_end = r_begin + self.config.label_len + self.config.pred_len

    #     seq_x = self.data_x[s_begin:s_end]
    #     if self.config.inverse_pred:
    #         # this is where inverse_pred != inverse output gets wonky if label_len != 0
    #         # its because the label doesn't get inversed
    #         seq_y = np.concatenate(
    #             [
    #                 self.data_x[
    #                     r_begin : r_begin + self.config.label_len
    #                 ],  # Use non-scaled data_x
    #                 self.data_y[r_begin + self.config.label_len : r_end],
    #             ],
    #             axis=0,
    #         )
    #     else:
    #         seq_y = self.data_y[r_begin:r_end]
    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]

    #     return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    # def index_to_dates(self, index):
    #     # index is of length batch_size
    #     s_begin = index
    #     s_end = s_begin + self.config.seq_len
    #     r_begin = s_end - self.config.label_len
    #     r_end = r_begin + self.config.label_len + self.config.pred_len

    #     seq_x_raw_dates = self.raw_dates[
    #         np.add.outer(s_begin, np.arange(self.config.seq_len))
    #     ]
    #     seq_y_raw_dates = self.raw_dates[
    #         np.add.outer(
    #             r_begin, np.arange(self.config.label_len + self.config.pred_len)
    #         )
    #     ]
    #     # seq_x_raw_dates = self.raw_dates[np.r_[s_begin,s_end-1].reshape(-1, index.shape[0]).T]# self.raw_dates.iloc[np.r_[s_begin,s_end]]
    #     # seq_y_raw_dates = self.raw_dates[np.r_[r_begin,r_end-1].reshape(-1, index.shape[0]).T]# self.raw_dates.iloc[np.r_[r_begin,r_end]]

    #     return seq_x_raw_dates, seq_y_raw_dates

    # def __len__(self):
    #     return len(self.data_x) - self.config.seq_len - self.config.pred_len + 1

    # def inverse_transform(self, data):
    #     return self.scaler.inverse_transform(data)


class DatasetCustom(Dataset):
    """
    Custom dataset class that inherits from PyTorch's Dataset class. It is used to create train, validation,
    and test datasets for time series forecasting tasks.

    Attributes:
        data_info (DataInfo): An object containing the dataset's information and configurations.
        flag (str): A flag indicating the type of dataset (e.g. 'train', 'val', or 'test').
        config (Config): An object containing the dataset configuration parameters.
        data_stamp (numpy.ndarray): A 1D array containing the timestamps for the dataset.
        data_x (numpy.ndarray): A 2D array containing the input features for the dataset.
        data_y (numpy.ndarray): A 2D array containing the target output for the dataset.
    """

    def __init__(self, data_info: DataInfo, flag: str):
        """
        Initialize the DatasetCustom instance with the provided data_info object and the dataset flag.

        Args:
            data_info (DataInfo): An object containing the dataset's information and configurations.
            flag (str): A flag indicating the type of dataset (e.g. 'train', 'val', or 'test').
        """
        self.data_info = data_info
        self.flag = flag
        self.config = data_info.config

        # Set up specific properties for train, val, or test datasets using the provided data_info object
        (
            self.data_stamp,
            self.raw_dates,
            self.data_x,
            self.data_y,
        ) = data_info.get_dataset_info(flag)

    def __getitem__(self, index: int):
        """
        Get the input and target sequences for the given index.

        Args:
            index (int): The index of the sample to be retrieved.

        Returns:
            tuple: A tuple containing input sequence, target sequence, input timestamps, target timestamps, and index.
        """
        s_begin = index
        s_end = s_begin + self.config.seq_len
        r_begin = s_end - self.config.label_len
        r_end = r_begin + self.config.label_len + self.config.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.config.inverse_pred:
            # this is where inverse_pred != inverse output gets wonky if label_len != 0
            # its because the label doesn't get inversed
            seq_y = np.concatenate(
                [
                    self.data_x[
                        r_begin : r_begin + self.config.label_len
                    ],  # Use non-scaled data_x
                    self.data_y[r_begin + self.config.label_len : r_end],
                ],
                axis=0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def index_to_dates(self, index: np.ndarray):
        """
        Convert index to the corresponding sequence of dates for input and target sequences.

        Args:
            index (numpy.ndarray): A 1D array of indices.

        Returns:
            tuple: A tuple containing the input dates and target dates.
        """
        # index is of length batch_size
        s_begin = index
        s_end = s_begin + self.config.seq_len
        r_begin = s_end - self.config.label_len
        r_end = r_begin + self.config.label_len + self.config.pred_len

        seq_x_raw_dates = self.raw_dates[
            np.add.outer(s_begin, np.arange(self.config.seq_len))
        ]
        seq_y_raw_dates = self.raw_dates[
            np.add.outer(
                r_begin, np.arange(self.config.label_len + self.config.pred_len)
            )
        ]
        # seq_x_raw_dates = self.raw_dates[np.r_[s_begin,s_end-1].reshape(-1, index.shape[0]).T]# self.raw_dates.iloc[np.r_[s_begin,s_end]]
        # seq_y_raw_dates = self.raw_dates[np.r_[r_begin,r_end-1].reshape(-1, index.shape[0]).T]# self.raw_dates.iloc[np.r_[r_begin,r_end]]

        return seq_x_raw_dates, seq_y_raw_dates

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data_x) - self.config.seq_len - self.config.pred_len + 1

    def inverse_transform(self, data: np.ndarray):
        """
        Transform the scaled data back to its original scale using the scaler from the data_info object.

        Args:
            data (numpy.ndarray): The scaled data to be inverse-transformed.

        Returns:
            numpy.ndarray: The data transformed back to its original scale.
        """
        return self.data_info.inverse_transform(data)


# class Dataset_Pred(Dataset):
#     def __init__(self, config, flag="pred"):
#         # Default values
#         defaults = {
#             "size": None,
#             "features": "S",
#             "target": "OT",
#             "scale": True,
#             "inverse": False,
#             "cols": None,
#             "date_start": None,
#             "date_end": None,
#             "t_embed": None,
#         }
#         config = dotdict({**defaults, **config})

#         assert config.seq_len is not None
#         assert config.label_len is not None
#         assert config.pred_len is not None
#         assert flag in ["pred"]
#         assert config.freq is not None
#         assert config.root_path is not None
#         assert config.data_path is not None
#         assert (
#             (config.date_start is None)
#             or (config.date_end is None)
#             or (config.date_start < config.date_end)
#         ), "date_start isn't before date_end"

#         self.config = config
#         self.flag = flag
#         # self.timeenc = 0 if config.t_embed != "timeF" else 1
#         if config.t_embed == "timeF":
#             self.timeenc = 1
#         elif config.t_embed == "time2vec_add" or config.t_embed == "time2vec_app":
#             self.timeenc = 2
#         else:
#             self.timeenc = 0

#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.config.root_path, self.config.data_path))
#         """
#         df_raw.columns: ['date', ...(other features), target feature]
#         """

#         # Filter to datapoints in [date_start, date_end]
#         if self.config.date_start is not None:
#             df_raw = df_raw.loc[(df_raw["date"] >= self.config.date_start)]
#         if self.config.date_end is not None:
#             df_raw = df_raw.loc[(df_raw["date"] <= self.config.date_end)]

#         if self.config.cols:
#             cols = self.config.cols.copy()
#             cols.remove(self.config.target)
#         else:
#             cols = list(df_raw.columns)
#             cols.remove(self.config.target)
#             cols.remove("date")
#         df_raw = df_raw[["date"] + cols + [self.config.target]]

#         border1 = len(df_raw) - self.config.seq_len
#         border2 = len(df_raw)

#         if self.config.features == "M" or self.config.features == "MS":
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.config.features == "S":
#             df_data = df_raw[[self.config.target]]

#         if self.config.scale:
#             self.scaler.fit(df_data.values, scale_mean=not self.config.no_scale_mean)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         tmp_stamp = df_raw[["date"]][border1:border2]
#         tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)
#         pred_dates = pd.date_range(
#             tmp_stamp.date.values[-1],
#             periods=self.config.pred_len + 1,
#             freq=self.config.freq,
#         )

#         df_stamp = pd.DataFrame(columns=["date"])
#         df_stamp.date = pd.to_datetime(
#             list(tmp_stamp.date.values) + list(pred_dates[1:]), utc=True
#         )
#         self.raw_dates = df_stamp.date.to_numpy(dtype=np.datetime64)
#         # TODO: What is the deal with .freq[-1:]
#         self.data_stamp = np.float32(
#             time_features(df_stamp, timeenc=self.timeenc, freq=self.config.freq[-1:])
#         )

#         self.data_x = data[border1:border2]
#         if self.config.inverse:
#             self.data_y = df_data.values[border1:border2]
#         else:
#             self.data_y = data[border1:border2]

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.config.seq_len
#         r_begin = s_end - self.config.label_len
#         r_end = r_begin + self.config.label_len + self.config.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         if self.config.inverse:
#             seq_y = self.data_x[r_begin : r_begin + self.config.label_len]
#         else:
#             seq_y = self.data_y[r_begin : r_begin + self.config.label_len]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark, index

#     def index_to_dates(self, index):
#         # index is of length batch_size
#         s_begin = index
#         s_end = s_begin + self.config.seq_len
#         r_begin = s_end - self.config.label_len
#         r_end = r_begin + self.config.label_len + self.config.pred_len

#         seq_x_raw_dates = self.raw_dates[
#             np.add.outer(s_begin, np.arange(self.config.seq_len))
#         ]
#         seq_y_raw_dates = self.raw_dates[
#             np.add.outer(
#                 r_begin, np.arange(self.config.label_len + self.config.pred_len)
#             )
#         ]

#         return seq_x_raw_dates, seq_y_raw_dates

#     def __len__(self):
#         return len(self.data_x) - self.config.seq_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
