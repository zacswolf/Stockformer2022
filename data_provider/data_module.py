import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from data_provider.data_loader import create_datasets
from utils.tools import dotdict
import pytorch_lightning as pl


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, config: dotdict, num_workers: int = 0):
        super().__init__()
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.config = config

        # pl makes self.batch_size special
        self.batch_size = config.batch_size
        self.num_workers = num_workers

        assert (
            not config.inverse
        ) or config.scale, "Can't enable inverse without enabling scale"

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: str | None = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        self.data_train, self.data_val, self.data_test = create_datasets(self.config)
        # self.data_pred = Dataset_Pred(self.config, flag="pred")
        print(
            f"LOADED DATASETS for {stage}: train: {len(self.data_train)}\tval: {len(self.data_val)}\ttest: {len(self.data_test)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=not self.config.dont_shuffle_train,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        # assert self.batch_size <= len(
        #     self.data_val
        # ), f"Batch size larger than val data set, batch size: {self.batch_size}, val size: {len(self.data_val)}"
        return [
            DataLoader(
                self.data_val,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.data_test,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                self.data_train,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.data_val,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.data_test,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
        ]

    def predict_dataloader(self):
        return (
            DataLoader(
                self.data_train,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.data_val,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.data_test,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
            # DataLoader(
            #     self.data_pred,
            #     batch_size=self.config.batch_size,
            #     shuffle=False,
            #     drop_last=False,
            #     num_workers=self.num_workers,
            # ),
        )
