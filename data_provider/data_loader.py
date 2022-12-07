import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler, dotdict
from utils.timefeatures import time_features

import warnings

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="h",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        self.raw_dates = df_stamp.date.to_numpy(dtype=np.datetime64)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [
                    self.data_x[r_begin : r_begin + self.label_len],
                    self.data_y[r_begin + self.label_len : r_end],
                ],
                0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def index_to_dates(self, index):
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

        return seq_x_raw_dates, seq_y_raw_dates

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="t",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        self.raw_dates = df_stamp.date.to_numpy(dtype=np.datetime64)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [
                    self.data_x[r_begin : r_begin + self.label_len],
                    self.data_y[r_begin + self.label_len : r_end],
                ],
                0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def index_to_dates(self, index):
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

        return seq_x_raw_dates, seq_y_raw_dates

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, config, flag="train", freq="h", timeenc=0):
        # Default values
        defaults = {
            "size": None,
            "features": "S",
            "target": "OT",
            "scale": True,
            "inverse": False,
            "cols": None,
            "date_start": None,
            "date_end": None,
            "date_test": None,
        }
        config = dotdict({**defaults, **config})

        assert config.seq_len is not None
        assert config.label_len is not None
        assert config.pred_len is not None
        assert flag in ["train", "test", "val"]
        assert freq is not None
        assert timeenc is not None
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

        self.config = config

        self.freq = freq
        self.timeenc = timeenc

        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.config.root_path, self.config.data_path))
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        # Filter to datapoints in [date_start, date_end]
        if self.config.date_start is not None:
            df_raw = df_raw.loc[(df_raw["date"] >= self.config.date_start)]
        if self.config.date_end is not None:
            df_raw = df_raw.loc[(df_raw["date"] <= self.config.date_end)]

        if self.config.cols:
            cols = self.config.cols.copy()
            assert self.config.target in cols, "Target not in cols"
            cols.remove(self.config.target)
        else:
            cols = list(df_raw.columns)
            assert self.config.target in cols, "Target not in data"
            cols.remove(self.config.target)
            assert "date" in cols, "`date` not in data"
            cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.config.target]]

        if self.config.test_date is None:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
        else:
            num_test = len(df_raw.loc[(df_raw["date"] >= self.config.date_test)])
            num_vali = num_test // 2
            num_train = len(df_raw) - num_vali - num_test

        border1s = [
            0,
            num_train - self.config.seq_len,
            len(df_raw) - num_test - self.config.seq_len,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.config.features == "M" or self.config.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.config.features == "S":
            df_data = df_raw[[self.config.target]]

        if self.config.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        self.raw_dates = df_stamp.date.to_numpy(dtype=np.datetime64)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.config.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.config.seq_len
        r_begin = s_end - self.config.label_len
        r_end = r_begin + self.config.label_len + self.config.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.config.inverse:
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

    def index_to_dates(self, index):
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
        return len(self.data_x) - self.config.seq_len - self.config.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, config, flag="pred", freq="15min", timeenc=0):
        # Default values
        defaults = {
            "size": None,
            "features": "S",
            "target": "OT",
            "scale": True,
            "inverse": False,
            "cols": None,
            "date_start": None,
            "date_end": None,
        }
        config = dotdict({**defaults, **config})

        assert config.seq_len is not None
        assert config.label_len is not None
        assert config.pred_len is not None
        assert flag in ["pred"]
        assert freq is not None
        assert timeenc is not None
        assert config.root_path is not None
        assert config.data_path is not None
        assert (
            (config.date_start is None)
            or (config.date_end is None)
            or (config.date_start < config.date_end)
        ), "date_start isn't before date_end"

        self.config = config

        self.freq = freq
        self.timeenc = timeenc

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.config.root_path, self.config.data_path))
        """
        df_raw.columns: ['date', ...(other features), target feature]
        """

        # Filter to datapoints in [date_start, date_end]
        if self.config.date_start is not None:
            df_raw = df_raw.loc[(df_raw["date"] >= self.config.date_start)]
        if self.config.date_end is not None:
            df_raw = df_raw.loc[(df_raw["date"] <= self.config.date_end)]

        if self.config.cols:
            cols = self.config.cols.copy()
            cols.remove(self.config.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.config.target)
            cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.config.target]]

        border1 = len(df_raw) - self.config.seq_len
        border2 = len(df_raw)

        if self.config.features == "M" or self.config.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.config.features == "S":
            df_data = df_raw[[self.config.target]]

        if self.config.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[["date"]][border1:border2]
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.config.pred_len + 1, freq=self.freq
        )

        df_stamp = pd.DataFrame(columns=["date"])
        df_stamp.date = pd.to_datetime(
            list(tmp_stamp.date.values) + list(pred_dates[1:]), utc=True
        )
        self.raw_dates = df_stamp.date.to_numpy(dtype=np.datetime64)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.config.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.config.seq_len
        r_begin = s_end - self.config.label_len
        r_end = r_begin + self.config.label_len + self.config.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.config.inverse:
            seq_y = self.data_x[r_begin : r_begin + self.config.label_len]
        else:
            seq_y = self.data_y[r_begin : r_begin + self.config.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def index_to_dates(self, index):
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

        return seq_x_raw_dates, seq_y_raw_dates

    def __len__(self):
        return len(self.data_x) - self.config.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
